import csv
import math
import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
VAL_FILE = os.path.join(DATA_DIR, "val.tsv")

MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

BEST_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "transformer_en_zh_best.pt")
LAST_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "transformer_en_zh_last.pt")
LOG_PATH = os.path.join(RESULT_DIR, "train_log.txt")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

MAX_SAMPLES_TRAIN = None
MAX_SAMPLES_VAL = None

MAX_SRC_LEN = 30
MAX_TGT_LEN = 50

MIN_FREQ = 1
BATCH_SIZE = 64
NUM_EPOCHS = 30
BASE_LR = 1e-3
WARMUP_STEPS = 4000
EARLY_STOP_PATIENCE = 6
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.2


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_en(text):
    text = text.strip().lower()
    return re.findall(r"\w+|[^\w\s]", text)


def tokenize_zh(text):
    return list(text.strip())


def read_parallel_data(file_path, max_samples=None):
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue

            src_text, tgt_text = row
            src_tokens = tokenize_en(src_text)
            tgt_tokens = tokenize_zh(tgt_text)

            if not src_tokens or not tgt_tokens:
                continue
            if len(src_tokens) > MAX_SRC_LEN or len(tgt_tokens) > MAX_TGT_LEN:
                continue

            pairs.append((src_tokens, tgt_tokens))

            if max_samples is not None and len(pairs) >= max_samples:
                break

    return pairs


class Vocab:
    def __init__(self, tokens_list=None, min_freq=1, itos=None, stoi=None):
        if itos is not None and stoi is not None:
            self.itos = itos
            self.stoi = stoi
            return

        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)

        self.itos = list(SPECIAL_TOKENS)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(BOS_IDX)
        for token in tokens:
            ids.append(self.stoi.get(token, UNK_IDX))
        if add_eos:
            ids.append(EOS_IDX)
        return ids

    def serialize(self):
        return {"itos": self.itos, "stoi": self.stoi}


def load_vocab_from_checkpoint(vocab_obj):
    return Vocab(itos=vocab_obj["itos"], stoi=vocab_obj["stoi"])


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.examples = []
        for src_tokens, tgt_tokens in pairs:
            src_ids = src_vocab.encode(src_tokens, add_bos=True, add_eos=True)
            tgt_ids = tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True)
            self.examples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_max_len = max(len(x) for x in src_batch)
    tgt_max_len = max(len(x) for x in tgt_batch)

    padded_src = []
    padded_tgt = []

    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        padded_src.append(src_ids + [PAD_IDX] * (src_max_len - len(src_ids)))
        padded_tgt.append(tgt_ids + [PAD_IDX] * (tgt_max_len - len(tgt_ids)))

    return (
        torch.tensor(padded_src, dtype=torch.long),
        torch.tensor(padded_tgt, dtype=torch.long),
    )


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


def build_vocab_from_pairs(pairs):
    src_tokens = [src for src, _ in pairs]
    tgt_tokens = [tgt for _, tgt in pairs]
    return Vocab(src_tokens, MIN_FREQ), Vocab(tgt_tokens, MIN_FREQ)


def save_checkpoint(path, model, src_vocab, tgt_vocab, epoch, best_val_loss, global_step):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "src_vocab": src_vocab.serialize(),
            "tgt_vocab": tgt_vocab.serialize(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "config": {
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_encoder_layers": NUM_ENCODER_LAYERS,
                "num_decoder_layers": NUM_DECODER_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": DROPOUT,
                "pad_idx": PAD_IDX,
            },
        },
        path,
    )


def append_log(line):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_resume_checkpoint():
    if os.path.exists(BEST_CHECKPOINT_PATH):
        return BEST_CHECKPOINT_PATH
    return None


def build_scheduler(optimizer):
    def lr_lambda(step):
        step = max(1, step)
        if step <= WARMUP_STEPS:
            return step / WARMUP_STEPS
        return (WARMUP_STEPS ** 0.5) / (step ** 0.5)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, global_step):
    model.train()
    total_loss = 0.0

    for src, tgt in tqdm(loader, desc="Training", leave=False):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        global_step += 1
        total_loss += loss.item()

    return total_loss / len(loader), global_step


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for src, tgt in tqdm(loader, desc="Validating", leave=False):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_in)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    set_seed(42)
    print(f"Using device: {DEVICE}")

    train_pairs = read_parallel_data(TRAIN_FILE, MAX_SAMPLES_TRAIN)
    val_pairs = read_parallel_data(VAL_FILE, MAX_SAMPLES_VAL)
    print(f"Train pairs loaded: {len(train_pairs)}")
    print(f"Val pairs loaded: {len(val_pairs)}")

    checkpoint = None
    src_vocab = None
    tgt_vocab = None
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    resume_path = find_resume_checkpoint()
    if resume_path is not None:
        print(f"Resume from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        src_vocab = load_vocab_from_checkpoint(checkpoint["src_vocab"])
        tgt_vocab = load_vocab_from_checkpoint(checkpoint["tgt_vocab"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        global_step = checkpoint.get("global_step", 0)
    else:
        print("No checkpoint found, train from scratch.")

    if src_vocab is None or tgt_vocab is None:
        src_vocab, tgt_vocab = build_vocab_from_pairs(train_pairs)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded.")

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = Adam(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = build_scheduler(optimizer)

    if global_step > 0:
        for _ in range(global_step):
            scheduler.step()

    print(f"Start epoch: {start_epoch + 1}")
    print(f"Best val loss so far: {best_val_loss if best_val_loss != float('inf') else 'inf'}")
    print(f"Warmup steps: {WARMUP_STEPS}")

    no_improve = 0

    for epoch in range(start_epoch + 1, start_epoch + NUM_EPOCHS + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, global_step
        )
        val_loss = eval_one_epoch(model, val_loader, criterion)

        lr_now = optimizer.param_groups[0]["lr"]
        line = f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_now:.8f}"
        print(line)
        append_log(line)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                src_vocab,
                tgt_vocab,
                epoch,
                best_val_loss,
                global_step,
            )
            print(f"Best model updated: {BEST_CHECKPOINT_PATH}")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{EARLY_STOP_PATIENCE}")

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            src_vocab,
            tgt_vocab,
            epoch,
            best_val_loss,
            global_step,
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training finished.")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Last checkpoint: {LAST_CHECKPOINT_PATH}")
    print(f"Log path: {LOG_PATH}")


if __name__ == "__main__":
    main()
