import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random

# 设置随机种子
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ==================== 1. 数据准备 ====================
data = [
    ("hello", "你好"),
    ("how are you", "你好吗"),
    ("i am fine", "我很好"),
    ("thank you", "谢谢"),
    ("good morning", "早上好"),
    ("good night", "晚安"),
    ("what is your name", "你叫什么名字"),
    ("my name is tom", "我叫汤姆"),
    ("i love you", "我爱你"),
    ("see you later", "再见"),
]


# 构建词表
def build_vocab(sentences):
    word_counts = Counter()
    for sent in sentences:
        for word in sent.lower().split():
            word_counts[word] += 1
    word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
    for word in word_counts:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word
    return word2idx, idx2word


en_word2idx, en_idx2word = build_vocab([pair[0] for pair in data])
zh_word2idx, zh_idx2word = build_vocab([pair[1] for pair in data])

print(f"英文词表大小: {len(en_word2idx)}")
print(f"中文词表大小: {len(zh_word2idx)}")
print()

# 序列转换
MAX_LEN = 10


def prepare_sequence(seq, word2idx, add_sos=False):
    indices = []
    if add_sos:
        indices.append(word2idx['<SOS>'])
    indices.extend([word2idx.get(word, word2idx['<UNK>']) for word in seq.lower().split()])
    indices.append(word2idx['<EOS>'])
    # 填充
    if len(indices) < MAX_LEN:
        indices += [word2idx['<PAD>']] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return torch.tensor(indices, dtype=torch.long)


# 准备数据
X = torch.stack([prepare_sequence(en, en_word2idx) for en, _ in data])
Y = torch.stack([prepare_sequence(zh, zh_word2idx, add_sos=True) for _, zh in data])

print(f"输入形状: {X.shape}, 输出形状: {Y.shape}")
print()


# ==================== 2. Bahdanau Attention ====================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        """
        query: (batch_size, hidden_size)
        keys: (batch_size, seq_len, hidden_size)
        """
        # query: (batch_size, 1, hidden_size)
        query = query.unsqueeze(1)

        # 计算注意力分数
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2)  # (batch_size, seq_len)

        # 注意力权重
        attn_weights = torch.softmax(scores, dim=-1)

        # 上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), keys)
        context = context.squeeze(1)  # (batch_size, hidden_size)

        return context, attn_weights


# ==================== 3. Encoder ====================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


# ==================== 4. Decoder ====================
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_token, decoder_hidden, encoder_outputs):
        """
        input_token: (batch_size,)
        decoder_hidden: (1, batch_size, hidden_size)
        encoder_outputs: (batch_size, seq_len, hidden_size)
        """
        # 嵌入输入token
        embedded = self.dropout(self.embedding(input_token))  # (batch_size, embedding_size)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, embedding_size)

        # 计算注意力
        query = decoder_hidden.squeeze(0)  # (batch_size, hidden_size)
        context, attn_weights = self.attention(query, encoder_outputs)
        context = context.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # 拼接嵌入和上下文
        rnn_input = torch.cat([embedded, context], dim=-1)  # (batch_size, 1, embedding_size + hidden_size)

        # GRU
        output, decoder_hidden = self.gru(rnn_input, decoder_hidden)

        # 生成输出
        prediction = self.fc(output.squeeze(1))  # (batch_size, vocab_size)

        return prediction, decoder_hidden, attn_weights


# ==================== 5. Seq2Seq模型 ====================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        # 存储输出
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        # 编码
        encoder_outputs, encoder_hidden = self.encoder(src)

        # 解码器初始输入（SOS token）
        decoder_input = trg[:, 0]  # (batch_size,)
        decoder_hidden = encoder_hidden

        for t in range(1, max_len):
            # 解码一步
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            outputs[:, t, :] = decoder_output

            # 决定下一个输入
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = trg[:, t] if teacher_force else top1

        return outputs


# ==================== 6. 训练函数 ====================
def train_model():
    # 超参数
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 500
    BATCH_SIZE = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    encoder = Encoder(len(en_word2idx), EMBEDDING_SIZE, HIDDEN_SIZE)
    decoder = Decoder(len(zh_word2idx), EMBEDDING_SIZE, HIDDEN_SIZE, len(zh_word2idx))
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=zh_word2idx['<PAD>'])

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 数据集
    dataset = list(zip(X, Y))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg)

            # 计算损失
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
            test_translation(model, device)
            print()

    return model


# ==================== 7. 翻译函数 ====================
def translate_sentence(model, sentence, device):
    model.eval()

    with torch.no_grad():
        # 编码
        src = prepare_sequence(sentence, en_word2idx).unsqueeze(0).to(device)
        encoder_outputs, encoder_hidden = model.encoder(src)

        # 解码
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([zh_word2idx['<SOS>']]).to(device)

        translated = []
        attentions = []

        for _ in range(MAX_LEN):
            decoder_output, decoder_hidden, attn_weights = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            attentions.append(attn_weights.cpu().numpy())
            top1 = decoder_output.argmax(1).item()

            if top1 == zh_word2idx['<EOS>']:
                break

            if top1 not in [zh_word2idx['<PAD>'], zh_word2idx['<SOS>'], zh_word2idx['<UNK>']]:
                translated.append(zh_idx2word[top1])

            decoder_input = torch.tensor([top1]).to(device)

    model.train()
    return ' '.join(translated)


def test_translation(model, device):
    test_sentences = ["hello", "how are you", "i love you", "good night", "what is your name"]

    for sent in test_sentences:
        translation = translate_sentence(model, sent, device)
        print(f"'{sent}' -> '{translation}'")


# ==================== 8. 主程序 ====================
print("=" * 50)
print("开始训练 Bahdanau Attention Seq2Seq 模型")
print("=" * 50)

model = train_model()

print("\n" + "=" * 50)
print("最终翻译测试:")
print("=" * 50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_translation(model, device)

# 交互式翻译
print("\n" + "=" * 50)
print("交互式翻译 (输入 'quit' 退出):")
print("=" * 50)

while True:
    text = input("\n请输入英文: ").strip()
    if text.lower() == 'quit':
        break
    if text:
        translation = translate_sentence(model, text, device)
        print(f"翻译结果: {translation}")