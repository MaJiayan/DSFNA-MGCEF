import os
import re
import torch.nn as nn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DebertaTokenizer,
    DebertaModel,
    BitsAndBytesConfig,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

from torch.amp import autocast, GradScaler
from tqdm import tqdm

DBMoudle_path = "/root/autodl-tmp/huggingface_cache/models--microsoft--deberta-base/snapshots/0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


deberta_model_name = "microsoft/deberta-base"

cache_dir = "/root/autodl-tmp/huggingface_cache"

# 1. 数据预处理
def load_data():
    train_df = pd.read_csv("/buzz_train.csv")
    val_df = pd.read_csv("/buzz_validation.csv")
    test_df = pd.read_csv("/buzz_test.csv")
    all_df = pd.read_csv("/buzz_TandV.csv")

    return train_df, test_df, val_df, all_df


def process_text(texts):
    # 读取停用词表并转换为数组
    with open('/stop.txt', 'r') as f:
        stop_words = [line.strip() for line in f if line.strip()]

    # 对停用词进行词干化处理
    ps = PorterStemmer()
    stemmed_stop_words = list(set([ps.stem(word) for word in stop_words]))

    # 文本预处理（分词+词干化）
    stemmed_texts = [" ".join([ps.stem(word) for word in text.split()]) for text in texts]

    # 创建TF-IDF向量化器（带停用词过滤）
    tfidf = TfidfVectorizer(
        max_features=80,
        stop_words=stemmed_stop_words
    )

    # 生成TF-IDF矩阵
    tfidf_matrix = tfidf.fit_transform(stemmed_texts)

    # 提取关键词
    keywords = [", ".join(tfidf.get_feature_names_out()[doc.nonzero()[1]])
                for doc in tfidf_matrix]

    print(len(keywords))
    print(keywords)

    return keywords


def preprocess_text(text):
    """清洗文本：移除特殊符号、处理编码、标准化格式"""
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text))
    try:
        cleaned_text = cleaned_text.encode("iso-8859-1").decode("utf-8")
    except Exception:
        pass
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text.lower()


class RumorDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_encoding = self.tokenizer(
            self.prompts[idx],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        response_encoding = self.tokenizer(
            self.responses[idx],
            max_length=384,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = torch.cat([
            prompt_encoding.input_ids[0],
            response_encoding.input_ids[0][1:]
        ])

        attention_mask = torch.cat([
            prompt_encoding.attention_mask[0],
            response_encoding.attention_mask[0][1:]
        ])

        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_encoding.input_ids[0]):] = input_ids[len(prompt_encoding.input_ids[0]):]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }




# 5. 特征提取
class DeBERTaFeatureExtractor:
    def __init__(self):
        self.tokenizer = DebertaTokenizer.from_pretrained(DBMoudle_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = DebertaModel.from_pretrained(
            DBMoudle_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        # 添加CLFE模块（卷积局部特征增强）
        self.clfe_module = CLFEModule(hidden_size=self.model.config.hidden_size)

        for layer in self.model.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.model.eval()

    def get_features(self, texts):
        texts = [preprocess_text(t) for t in texts]
        texts = [str(t) if not pd.isnull(t) else '' for t in texts]
        features = []

        batch_size = 8
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.model.device)

                # 获取嵌入输出
                embedding_output = self.model.embeddings(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs.get('token_type_ids'),
                    position_ids=None
                )

                # 应用CLFE模块进行局部特征增强
                enhanced_embeddings = self.clfe_module(embedding_output)

                # 将增强后的嵌入传递给编码器
                encoder_outputs = self.model.encoder(
                    hidden_states=enhanced_embeddings,
                    attention_mask=inputs['attention_mask'].unsqueeze(1).unsqueeze(2)
                )

                cls_vectors = encoder_outputs.last_hidden_state[:, 0, :]

                batch_features = torch.nn.functional.layer_norm(cls_vectors, (cls_vectors.size(-1),))

                batch_features = batch_features.detach().cpu().float()
                if batch_features.shape[0] > 1:
                    batch_features = (batch_features - batch_features.mean(0)) / (batch_features.std(0) + 1e-8)
                    batch_features = torch.nan_to_num(
                        batch_features,
                        nan=0.0,
                        posinf=10000,
                        neginf=-10000
                    )
                # 执行截断操作
                batch_features = torch.clamp(batch_features, min=-10000, max=10000)
                features.append(batch_features.numpy())

                # 显存释放
                del inputs, embedding_output, enhanced_embeddings, encoder_outputs, batch_features
                torch.cuda.empty_cache()

        return np.concatenate(features) if features else np.array([])


# CLFE模块定义
class CLFEModule(nn.Module):
    def __init__(self, hidden_size):
        super(CLFEModule, self).__init__()
        self.hidden_size = hidden_size

        # 三个并行的卷积层，卷积核大小分别为3,4,5
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1  # 保持序列长度不变
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=4,
            padding=1  # 保持序列长度不变
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=5,
            padding=2  # 保持序列长度不变
        )

        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        # hidden_states形状: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.shape

        hidden_states_t = hidden_states.transpose(1, 2)

        # 三个并行的卷积层
        conv1_out = self.conv1(hidden_states_t)
        conv2_out = self.conv2(hidden_states_t)
        conv3_out = self.conv3(hidden_states_t)

        # 转置回原始形状并逐元素相加
        conv1_out = conv1_out.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        conv2_out = conv2_out.transpose(1, 2)
        conv3_out = conv3_out.transpose(1, 2)

        # 融合三个卷积输出
        conv_output = conv1_out + conv2_out + conv3_out

        # 残差连接 + GELU + LayerNorm
        enhanced_output = hidden_states + self.gelu(conv_output)
        enhanced_output = self.layer_norm(enhanced_output)

        return enhanced_output


# 6. 分类模型
class FakeNewsClassifier(torch.nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained(
            DBMoudle_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        for layer in self.deberta.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # 使用CLS向量作为特征
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # 分类
        return self.classifier(cls_embedding)


# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    all_probs = []  # 存储所有批次的概率值
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.squeeze(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if len(all_probs) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc_roc": 0}

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs)
    }


# 7. 训练流程
def main():
    # 加载数据
    train_df, test_df, val_df, all_df = load_data()
    print("加载数据完成")

    # 数据增强
    checkpoint_path = "/best_class/training_checkpoint_buzz.pth"
    augument_path = "/buzz_augumentation.csv"
    deepseek_local_path = "/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/afbda8b347ec881666061fa67447046fc5164ec8"
    deepseek_tokenizer = AutoTokenizer.from_pretrained(
        deepseek_local_path,
        local_files_only=True,
        trust_remote_code=True
    )

    deepseek_model = AutoModelForCausalLM.from_pretrained(
        deepseek_local_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    # 设置pad_token
    if deepseek_tokenizer.pad_token is None:
        deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token
    augumented_df = pd.read_csv(augument_path)


    full_train = pd.concat([train_df, augumented_df], ignore_index=True)
    if isinstance(full_train, pd.DataFrame):
        full_train = full_train.sample(frac=1).reset_index(drop=True)
    elif isinstance(full_train, np.ndarray):
        shuffled_indices = np.random.permutation(len(full_train))
        full_train = full_train[shuffled_indices]
    else:
        print("不支持的数据类型，无法打乱顺序")

    full_train["text"] = full_train["text"].astype(str).fillna("")
    test_df["text"] = test_df["text"].astype(str).fillna("")
    val_df["text"] = val_df["text"].astype(str).fillna("")

    # 特征提取
    tokenizer = DebertaTokenizer.from_pretrained(DBMoudle_path)

    # 数据预处理函数
    def encode_texts(texts, max_length=512):
        return tokenizer(
            [preprocess_text(t) for t in texts],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    # 编码数据集
    train_encodings = encode_texts(full_train["text"])
    val_encodings = encode_texts(val_df["text"])
    test_encodings = encode_texts(test_df["text"])

    # 构建Dataset
    class TextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = torch.FloatTensor(labels).squeeze()

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]
            }

    train_dataset = TextDataset(train_encodings, full_train["label"].values)
    val_dataset = TextDataset(val_encodings, val_df["label"].values)
    test_dataset = TextDataset(test_encodings, test_df["label"].values)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 训练分类器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FakeNewsClassifier().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.deberta.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.05)

    criterion = torch.nn.BCEWithLogitsLoss()

    start_epoch = 0
    early_stop_counter = 0

    # 检查是否存在检查点
    if os.path.exists(checkpoint_path):
        print("\n发现训练检查点，正在加载...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint['best_metrics']
        early_stop_counter = checkpoint['early_stop_counter']
        print(f"已恢复训练状态，将从第{start_epoch}轮开始继续训练")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)


    scaler = GradScaler()


    for epoch in range(start_epoch, 30):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(**inputs).squeeze(-1)
            loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # 更新学习率
        scheduler.step()

        print(f"训练状态已保存至 {checkpoint_path}")
        # 验证评估
        metrics = evaluate_model(model, val_loader, device)
        if np.isnan(metrics['auc_roc']).any():
            print("Metrics contains NaN values. Skipping this epoch.")
            continue
        print(f"\nEpoch {epoch + 1} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")

        # 保存最佳模型
        best_metrics = {
            "f1": float(metrics['f1']),
            "auc_roc": float(metrics['auc_roc'])
        }

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_metrics': best_metrics,
            'early_stop_counter': early_stop_counter
        }, checkpoint_path)

    # ================== 6. 最终测试 ==================
    model.eval()

    test_probs = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
            test_probs.append(probs)
            test_labels.append(labels)

    # 过滤空数组并拼接
    test_probs = [arr for arr in test_probs if arr.size > 0]
    if not test_probs:
        print("警告：未生成任何测试概率，结果不可信！")
        test_probs = np.array([])
    else:
        test_probs = np.concatenate(test_probs)

    test_labels = np.concatenate(test_labels)

    # 计算最终指标
    test_preds = (test_probs > 0.5).astype(int) if test_probs.size > 0 else np.array([])
    test_metrics = {
        "accuracy": accuracy_score(test_labels, test_preds) if test_probs.size > 0 else 0,
        "precision": precision_score(test_labels, test_preds, zero_division=0) if test_probs.size > 0 else 0,
        "recall": recall_score(test_labels, test_preds, zero_division=0) if test_probs.size > 0 else 0,
        "f1": f1_score(test_labels, test_preds, zero_division=0) if test_probs.size > 0 else 0,
        "auc_roc": roc_auc_score(test_labels, test_probs) if test_probs.size > 0 else 0
    }

    print("\n=== Final Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k.upper():<10}: {v:.4f}")


if __name__ == "__main__":
    main()