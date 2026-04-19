import csv
import os
import random
import re
from tqdm import tqdm

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

SENTENCES_FILE = os.path.join(RAW_DIR, "sentences.csv")
LINKS_FILE = os.path.join(RAW_DIR, "links.csv")

ALL_FILE = os.path.join(PROCESSED_DIR, "eng_cmn.tsv")
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.tsv")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.tsv")

RANDOM_SEED = 42
MAX_SAMPLES = None
VAL_RATIO = 0.1

MIN_EN_WORDS = 1
MAX_EN_WORDS = 25
MIN_ZH_CHARS = 1
MAX_ZH_CHARS = 40


def normalize_english(text):
    text = text.strip().lower()
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-zA-Z0-9?.!,;:'\"()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_chinese(text):
    text = text.strip()
    text = text.replace("　", "")
    text = re.sub(r"\s+", "", text)
    return text


def english_word_count(text):
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def chinese_char_count(text):
    return len(text)


def is_valid_pair(en_text, zh_text):
    if not en_text or not zh_text:
        return False

    en_len = english_word_count(en_text)
    zh_len = chinese_char_count(zh_text)

    if en_len < MIN_EN_WORDS or en_len > MAX_EN_WORDS:
        return False
    if zh_len < MIN_ZH_CHARS or zh_len > MAX_ZH_CHARS:
        return False

    if any(token in en_text for token in ["http://", "https://", "www."]):
        return False

    words = re.findall(r"[a-zA-Z0-9']+", en_text)
    if len(words) > 3 and len(set(words)) == 1:
        return False

    return True


def load_sentences(sentences_file):
    eng_sentences = {}
    cmn_sentences = {}

    print("Loading sentences.csv ...")
    with open(sentences_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader):
            if len(row) < 3:
                continue
            sent_id, lang, text = row[0], row[1], row[2]

            if lang == "eng":
                text = normalize_english(text)
                if text:
                    eng_sentences[sent_id] = text
            elif lang == "cmn":
                text = normalize_chinese(text)
                if text:
                    cmn_sentences[sent_id] = text

    print(f"English sentences: {len(eng_sentences)}")
    print(f"Chinese sentences: {len(cmn_sentences)}")
    return eng_sentences, cmn_sentences


def load_links(links_file):
    print("Loading links.csv ...")
    links = []
    with open(links_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader):
            if len(row) < 2:
                continue
            links.append((row[0], row[1]))
    print(f"Total links: {len(links)}")
    return links


def build_parallel_pairs(eng_sentences, cmn_sentences, links):
    print("Building English-Chinese pairs ...")
    pairs = []
    seen = set()

    for left_id, right_id in tqdm(links):
        en_text = None
        zh_text = None

        if left_id in eng_sentences and right_id in cmn_sentences:
            en_text = eng_sentences[left_id]
            zh_text = cmn_sentences[right_id]
        elif right_id in eng_sentences and left_id in cmn_sentences:
            en_text = eng_sentences[right_id]
            zh_text = cmn_sentences[left_id]

        if en_text is None or zh_text is None:
            continue

        if not is_valid_pair(en_text, zh_text):
            continue

        key = (en_text, zh_text)
        if key in seen:
            continue

        seen.add(key)
        pairs.append(key)

    print(f"Valid unique pairs: {len(pairs)}")
    return pairs


def save_tsv(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for en_text, zh_text in pairs:
            writer.writerow([en_text, zh_text])


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    eng_sentences, cmn_sentences = load_sentences(SENTENCES_FILE)
    links = load_links(LINKS_FILE)
    pairs = build_parallel_pairs(eng_sentences, cmn_sentences, links)

    random.shuffle(pairs)

    if MAX_SAMPLES is not None and len(pairs) > MAX_SAMPLES:
        pairs = pairs[:MAX_SAMPLES]

    val_size = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    save_tsv(ALL_FILE, pairs)
    save_tsv(TRAIN_FILE, train_pairs)
    save_tsv(VAL_FILE, val_pairs)

    print()
    print("Data preparation finished.")
    print(f"All pairs saved to:   {ALL_FILE}")
    print(f"Train pairs saved to: {TRAIN_FILE}")
    print(f"Val pairs saved to:   {VAL_FILE}")
    print(f"Train size: {len(train_pairs)}")
    print(f"Val size:   {len(val_pairs)}")

    print("\nSample pairs:")
    for i, (en_text, zh_text) in enumerate(train_pairs[:10], start=1):
        print(f"{i}. {en_text} -> {zh_text}")


if __name__ == "__main__":
    main()
