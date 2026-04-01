# tokenizer.py
# Simple word-level tokenizer with basic normalization and vocab save/load.

import re
import json
from collections import Counter, defaultdict

class SimpleTokenizer:
    def __init__(self, unk_token="<UNK>", pad_token="<PAD>", min_freq=1):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def clean(self, text):
        text = text.lower()

        # Replace numbers with a single <num> token
        text = re.sub(r"\d+", "", text)

        # Normalize apostrophes and split contractions
        text = text.replace("’", "'")
        text = text.replace("n't", " not")
        text = text.replace("'re", " are")
        text = text.replace("'s", " is")
        text = text.replace("'ll", " will")
        text = text.replace("'ve", " have")
        text = text.replace("'m", " am")
        text = text.replace("'", "")

        # Remove ALL punctuation (except spaces)
        text = re.sub(r"[^a-z\s]", " ", text)

        # Remove single-letter garbage words except a/i
        text = re.sub(r"\b[b-hj-z]\b", " ", text)

        # Remove duplicate spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text



    def build_vocab(self, texts, max_vocab=None):
        # texts: iterable of strings
        counter = Counter()
        for t in texts:
            cleaned = self.clean(t)
            tokens = cleaned.split()
            counter.update(tokens)
        # Start with special tokens
        vocab = [self.pad_token, self.unk_token]
        # include tokens by frequency
        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if tok in vocab:
                continue
            vocab.append(tok)
            if max_vocab and len(vocab) >= max_vocab:
                break
        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.idx2word = {i:w for w,i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        cleaned = self.clean(text)
        return [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in cleaned.split()]

    def decode(self, indices):
        return " ".join(self.idx2word.get(i, self.unk_token) for i in indices)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        self.word2idx = j["word2idx"]
        # Correct way to rebuild idx2word
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
