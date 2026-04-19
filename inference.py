import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencc import OpenCC


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "models/transformer_en_zh_best.pt"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = "models/transformer_en_zh_last.pt"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

MAX_DECODE_LEN = 60
BEAM_SIZE = 8
LENGTH_PENALTY_ALPHA = 0.9
REPEAT_PENALTY = 1.5
UNK_PENALTY = 0.4
NO_REPEAT_NGRAM_SIZE = 3
MIN_DECODE_LEN = 3

cc = OpenCC("t2s")


def tokenize_en(text):
    text = text.strip().lower()
    return re.findall(r"\w+|[^\w\s]", text)


class Vocab:
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi

    def __len__(self):
        return len(self.itos)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 兼容旧 checkpoint 的存储格式: [max_len, 1, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]  (当前模型是 batch_first=True)
        """
        pe = self.pe[:x.size(1)].transpose(0, 1)  # -> [1, seq_len, d_model]
        x = x + pe
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        pad_idx,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=1000)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.fc_out.weight = self.tgt_embedding.weight

    def generate_square_subsequent_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, tgt_input):
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt_input == self.pad_idx)
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1), tgt_input.device)

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt_input) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)


def encode_source(text, src_vocab):
    tokens = tokenize_en(text)
    ids = [BOS_IDX] + [src_vocab.stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def decode_ids_to_text(ids, tgt_vocab):
    tokens = []
    for idx in ids:
        if idx == EOS_IDX:
            break
        if idx in (PAD_IDX, BOS_IDX):
            continue
        token = tgt_vocab.itos[idx]
        if token not in SPECIAL_TOKENS:
            tokens.append(token)
    return cc.convert("".join(tokens))


def has_repeat_ngram(token_ids, next_id, n):
    if n <= 0:
        return False

    candidate = token_ids + [next_id]
    if len(candidate) < n * 2:
        return False

    last_ngram = tuple(candidate[-n:])
    seen = set(tuple(candidate[i:i + n]) for i in range(len(candidate) - n))
    return last_ngram in seen


def too_many_same_tail(token_ids, next_id, repeat_limit=2):
    candidate = token_ids + [next_id]
    if len(candidate) < repeat_limit + 1:
        return False
    return all(x == next_id for x in candidate[-repeat_limit:])


@torch.no_grad()
def beam_search_decode(model, src_tensor, tgt_vocab):
    beams = [([BOS_IDX], 0.0, False)]

    for step_idx in range(MAX_DECODE_LEN):
        candidates = []

        for token_ids, score, finished in beams:
            if finished:
                candidates.append((token_ids, score, True))
                continue

            tgt_input = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            logits = model(src_tensor, tgt_input)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)

            if UNK_IDX < log_probs.size(0):
                log_probs[UNK_IDX] += math.log(UNK_PENALTY)

            if step_idx < MIN_DECODE_LEN and EOS_IDX < log_probs.size(0):
                log_probs[EOS_IDX] = -1e9

            for prev_id in token_ids[-4:]:
                if prev_id < log_probs.size(0):
                    log_probs[prev_id] += math.log(1.0 / REPEAT_PENALTY)

            topk_log_probs, topk_ids = torch.topk(log_probs, BEAM_SIZE * 8)

            kept = 0
            for k in range(BEAM_SIZE * 8):
                next_id = topk_ids[k].item()

                if next_id != EOS_IDX and has_repeat_ngram(token_ids, next_id, NO_REPEAT_NGRAM_SIZE):
                    continue

                if next_id != EOS_IDX and too_many_same_tail(token_ids, next_id, repeat_limit=2):
                    continue

                next_tokens = token_ids + [next_id]
                next_score = score + topk_log_probs[k].item()
                next_finished = (next_id == EOS_IDX)

                candidates.append((next_tokens, next_score, next_finished))
                kept += 1

                if kept >= BEAM_SIZE * 3:
                    break

        if not candidates:
            break

        def rank(item):
            token_ids, score, _ = item
            length = max(1, len(token_ids) - 1)
            lp = ((5 + length) / 6) ** LENGTH_PENALTY_ALPHA
            return score / lp

        candidates.sort(key=rank, reverse=True)
        beams = candidates[:BEAM_SIZE]

        if all(x[2] for x in beams):
            break

    best = max(
        beams,
        key=lambda x: x[1] / (((5 + max(1, len(x[0]) - 1)) / 6) ** LENGTH_PENALTY_ALPHA),
    )
    return decode_ids_to_text(best[0][1:], tgt_vocab)


def translate(text, model, src_vocab, tgt_vocab):
    src_tensor = encode_source(text, src_vocab)
    return beam_search_decode(model, src_tensor, tgt_vocab)


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    src_vocab = Vocab(checkpoint["src_vocab"]["itos"], checkpoint["src_vocab"]["stoi"])
    tgt_vocab = Vocab(checkpoint["tgt_vocab"]["itos"], checkpoint["tgt_vocab"]["stoi"])
    config = checkpoint["config"]

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        pad_idx=config["pad_idx"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_sentences = [
        "Although it was raining heavily, we decided to continue the trip.",
        "If I had known you were coming, I would have prepared dinner.",
        "The book that you gave me yesterday is surprisingly insightful.",
        "She not only speaks French fluently, but also writes it beautifully.",
        "By the time we arrived at the station, the last train had already left.",
        "The company is expected to announce its quarterly earnings next Monday.",
        "Please let me know whether you can attend the meeting before Friday.",
        "He pretended not to notice that everyone was laughing at him.",
        "What matters most is not where you start, but where you end up.",
        "In order to reduce carbon emissions, many cities are expanding public transport.",
        "I am feeling under the weather today.",
        "Don't beat around the bush—just tell me the truth.",
        "It never rains but it pours.",
        "Once in a blue moon, we go out for a fancy dinner.",
        "I thought the exam would be easy, but it turned out to be a nightmare.",
    ]

    print(f"Using device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("Translation results (Simplified Chinese):")
    print("-" * 60)

    for sentence in test_sentences:
        zh = translate(sentence, model, src_vocab, tgt_vocab)
        print(f"EN: {sentence}")
        print(f"ZH: {zh}")
        print("-" * 60)


if __name__ == "__main__":
    main()
