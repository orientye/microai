# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html

import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionWiseFFN(4, 8)
ffn.eval()
result = ffn(torch.ones((2, 3, 5)))[0]

print(f"输出形状: {result.shape}")
print(f"输出内容:")
print(result)

print(f"dense1 权重形状: {ffn.dense1.weight.shape}")  # torch.Size([4, 5])
print(f"dense2 权重形状: {ffn.dense2.weight.shape}")  # torch.Size([8, 4])

ln = nn.LayerNorm(2) #创建一个层归一化层 对每个样本的所有特征进行归一化（横向归一化）
bn = nn.LazyBatchNorm1d() #创建一个批归一化层（懒加载版本，不需要预先指定特征数，会在第一次前向传播时自动推断） 对每个特征在所有样本上进行归一化（纵向归一化）
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance from X in the training mode
print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        """
        X：原始输入（残差连接的“直连”路径）
        Y：经过子层（如自注意力或前馈网络）处理后的输出
        self.dropout(Y) + X：将子层输出（经过dropout后）与原始输入相加，这就是残差连接
        """
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm(4, 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(torch.ones(shape), torch.ones(shape)), shape)


class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        # 1. 多头自注意力层：学习序列内部的依赖关系
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        # 2. 第一个残差连接和层归一化（用于注意力层之后）
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        # 3. 逐位前馈网络：对每个位置的特征进行非线性变换
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        # 4. 第二个残差连接和层归一化（用于前馈网络之后）
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        """
        首先计算自注意力：使用相同的X作为查询、键和值，valid_lens控制有效长度。输出形状与X相同(batch_size, seq_len, num_hiddens)。
        然后通过addnorm1：将注意力输出与原始输入X做残差连接，然后进行层归一化和可能的dropout，输出Y。
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        #计算前馈网络：self.ffn(Y)将Y变换为相同形状。然后通过addnorm2：将前馈网络输出与Y做残差连接，再归一化。返回最终输出，形状不变。
        return self.addnorm2(Y, self.ffn(Y))

X = torch.ones((2, 100, 24)) # 全一张量，表示 batch_size=2，序列长度=100，嵌入维度=24（即 num_hiddens=24）。
valid_lens = torch.tensor([3, 2]) # 有效长度张量 [3, 2]，表示第一个样本只有前 3 个位置有效，第二个样本只有前 2 个位置有效，其余为填充。
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval() # 将模块切换到评估模式，主要影响 Dropout 和 BatchNorm 等层，例如 Dropout 在 eval 模式下不生效。
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)

class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        # 1. 词嵌入层：将单词索引转换为连续向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 2. 位置编码：为向量注入位置信息（Transformer 本身不识别顺序）
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 3. 堆叠多个编码器块
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        # 第一步：词嵌入，并乘以特征维度的平方根进行缩放（为了和位置编码匹配）
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 准备存储每一层的注意力权重
        self.attention_weights = [None] * len(self.blks)
        # 逐层通过编码器块
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 保存该层计算出的注意力权重
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

# 词汇表大小 200，特征/隐藏维度 24，前馈内部维度 48，头数 8，块数 2，dropout 0.5。
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens),
                (2, 100, 24))

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        """
        self.i: 当前块在堆栈中的索引（用于缓存KV值）。
        self.attention1: 掩蔽自注意力（Masked Self - Attention），防止模型看到未来信息。
        self.addnorm1, 2, 3: 残差连接 + 层归一化（Add & Norm）。
        self.attention2: 编码器 - 解码器交叉注意力（Encoder - Decoder Attention）。
        self.ffn: 逐位前馈网络（Position - Wise Feed - Forward Network）。
        
        self.attention1 (掩蔽自注意力 - Masked Self-Attention)
            Query, Key, Value 全来自解码器本身。
            目的：让解码器在生成当前词时，回顾之前已经生成出来的词。
            约束：它必须是“掩蔽”的。在训练时，它不能看到“未来”的词。
        self.attention2 (编码器-解码器注意力 - Cross Attention)
            Query 来自解码器，Key 和 Value 来自编码器的输出。
            目的：让解码器在生成词时，去“查阅”输入句子（源语言）的信息。
            联系：它是连接翻译源（如英文）和翻译目标（如中文）的桥梁。
            
        forward 函数里的调用：
            # self-attention: 三个输入全是 key_values (即 X 的历史累积)
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens) 
            # encoder-decoder attention: Query 是 Y，但 Key/Value 是 enc_outputs
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        
        特性	attention1 (Self)	attention2 (Cross)
        Query (Q)	    当前解码器的状态	上一层自注意力的输出
        Key (K)	        解码器已生成的历史信息	编码器的最终输出
        Value (V)	    解码器已生成的历史信息	编码器的最终输出
        掩码 (Mask)	    dec_valid_lens (防止看未来)	enc_valid_lens (忽略填充字符)
        
        attention1 负责“内部消化”：处理生成过程中的序列依赖。
        attention2 负责“外部对齐”：处理输入与输出之间的对应关系。
        """
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    """
    前向传播。
    enc_outputs, enc_valid_lens: 从 state 中提取编码器的输出和有效长度。
    KV 缓存逻辑:
        如果 state[2][self.i] 为空，说明是训练阶段或预测的第一步，KV 值即为当前 X。
        如果不为空，则将历史 KV 值与当前的 X 在时间维度（dim=1）上拼接，实现增量解码。
    训练掩码:
        if self.training: 生成自增序列 [1, 2, ..., num_steps] 作为有效长度，确保注意力机制实现“因果掩码”（即第t个词只能看到前t个词）。
    计算流:
        执行自注意力。
        执行第一个 AddNorm。
        执行交叉注意力（Query 来自解码器，Key/Value 来自编码器）。
        执行第二个 AddNorm。
        执行前馈网络并进行最后的 AddNorm。
    """
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

"""
decoder_blk = ...: 创建一个 24 隐藏单元、8 头的解码器块。
X = torch.ones(...): 模拟一个 batch 为 2，序列长度 100，特征维度 24 的输入。
state = [...]: 初始化状态，包含编码器输出、长度和用于 KV 缓存的空列表。
d2l.check_shape(...): 验证输出形状是否与输入 X 一致。
"""
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)

class TransformerDecoder(d2l.AttentionDecoder):
    """
    self.embedding: 将词索引转换为词向量，并乘以 √dmodel进行缩放。
    self.pos_encoding: 添加位置编码，赋予模型处理序列顺序的能力。
    self.blks: 使用 nn.Sequential 堆叠 num_blks 个解码器块。
    self.dense: 一个全连接层，将隐藏特征映射回词表大小（Vocab Size），用于预测下一个词。
    """
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    #初始化解码器状态，创建一个长度为层数的列表 [None] * num_blks，用来在推理阶段存储每一层的 KV 缓存。
    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    """
    self._attention_weights: 初始化一个二维列表，用于记录每一层的两种注意力权重（方便后续可视化）。
    循环计算: 遍历所有块，逐层更新 X 和 state。
    权重提取: 在循环中提取并保存 attention1 和 attention2 的权重。
    """
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, 'data')
_CKPT_PATH = os.path.join(_DATA_DIR, 'mt_seq2seq_eng_zh.pt')
_CORPUS = os.path.join(_DATA_DIR, 'eng-zh.txt')
with open(_CORPUS, encoding='utf-8') as _cf:
    _N = sum(1 for line in _cf if line.strip() and '\t' in line)
if _N == 0:
    raise FileNotFoundError(f'No valid tab-separated pairs in {_CORPUS}')
if _N == 1:
    _NUM_TRAIN, _NUM_VAL = 1, 0
else:
    _NUM_VAL = min(2000, max(1, _N // 10), _N - 1)
    _NUM_TRAIN = _N - _NUM_VAL
_MIN_FREQ = 2 if _N >= 1000 else 1

data = d2l.MTEngZh(
    batch_size=128,
    num_steps=48,
    num_train=_NUM_TRAIN,
    num_val=_NUM_VAL,
    root=_DATA_DIR,
    filename='eng-zh.txt',
    min_freq=_MIN_FREQ,
)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
encoder = TransformerEncoder(
    len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
    num_blks, dropout)
decoder = TransformerDecoder(
    len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
    num_blks, dropout)
model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.001)
# Training plots (matplotlib + IPython) dominate wall time in scripts; turn off.
model.board.display = False
# At most one loss aggregation per epoch (cheap bookkeeping vs default ~2/epoch).
model.plot_train_per_epoch = 1
model.plot_valid_per_epoch = 1

_NUM_GPUS = min(1, d2l.num_gpus())


def _ckpt_meta():
    return {
        'corpus': os.path.basename(_CORPUS),
        'n_lines': _N,
        'num_train': data.num_train,
        'num_val': data.num_val,
        'num_steps': data.num_steps,
        'min_freq': data.min_freq,
        'src_vocab': len(data.src_vocab),
        'tgt_vocab': len(data.tgt_vocab),
        'num_hiddens': num_hiddens,
        'num_blks': num_blks,
        'ffn_num_hiddens': ffn_num_hiddens,
        'num_heads': num_heads,
    }


def _materialize_lazy_layers(m, dat, dev):
    """One forward so LazyLinear / lazy attention layers have parameters."""
    m.eval()
    batch = next(iter(dat.train_dataloader()))
    batch = [d2l.to(t, dev) for t in batch]
    with torch.no_grad():
        m(batch[0], batch[1], batch[2])
    m.train()


_device = d2l.try_gpu()
_meta = _ckpt_meta()
_checkpoint_loaded = False
print(f'Training will use num_gpus={_NUM_GPUS}, device={_device}')

if os.path.isfile(_CKPT_PATH):
    try:
        pack = torch.load(_CKPT_PATH, map_location=_device)
    except Exception as e:
        pack = None
        print(f'Checkpoint read failed ({e}); will train from scratch.')
    if pack is not None:
        state = pack['state_dict'] if isinstance(pack, dict) and 'state_dict' in pack else pack
        old_meta = pack.get('meta') if isinstance(pack, dict) else None
        if old_meta == _meta:
            try:
                model.to(_device)
                _materialize_lazy_layers(model, data, _device)
                model.load_state_dict(state, strict=True)
                _checkpoint_loaded = True
                print(f'Loaded checkpoint (skipped training): {_CKPT_PATH}')
            except Exception as e:
                print(f'Checkpoint load failed ({e}); will train from scratch.')
        else:
            print('Checkpoint meta mismatch (corpus / vocab / hyperparams); retraining.')

if not _checkpoint_loaded:
    trainer = d2l.Trainer(max_epochs=15, gradient_clip_val=1, num_gpus=_NUM_GPUS)
    trainer.fit(model, data)
    model.to(_device)
    torch.save({'state_dict': model.state_dict(), 'meta': _meta}, _CKPT_PATH)
    print(f'Saved checkpoint: {_CKPT_PATH}')

model.eval()

engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
zhs = ['走吧。', '我迷路了。', '他很冷静。', '我到家了。']
preds, _ = model.predict_step(
    data.build(engs, zhs), _device, data.num_steps)
for en, zh, p in zip(engs, zhs, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    ref = " ".join([c for c in zh.replace(" ", "")])
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), ref, k=2):.3f}')

_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [zhs[-1]]), _device, data.num_steps, True)
enc_attention_weights = torch.cat(model.encoder.attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = enc_attention_weights.reshape(shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))

d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
shape = (-1, 2, num_blks, num_heads, data.num_steps)
dec_attention_weights = dec_attention_weights_filled.reshape(shape)
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)

d2l.check_shape(dec_self_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
d2l.check_shape(dec_inter_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))

d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

plt.show()