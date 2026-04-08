
"""
BiDAF-style QA model for SQuAD with either:
1) contextualized BERT embeddings, or
2) static GloVe embeddings loaded from one or more embedding files.

Examples:
    python nlpp4.py --embedding_type bert --train_samples 2000 --val_samples 500 --epochs 2

    python nlpp4.py \
        --embedding_type glove \
        --glove_paths glove.6B.100d.txt other_glove_100d.txt \
        --glove_dim 100 \
        --train_samples 2000 --val_samples 500 --epochs 2
"""

from __future__ import annotations
import argparse
import collections
import json
import os
import random
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
import evaluate
from transformers import BertModel, AutoTokenizer


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def build_vocab(samples: List[Dict[str, Any]], min_freq: int = 1) -> Dict[str, int]:
    counter: collections.Counter = collections.Counter()
    for sample in samples:
        counter.update(sample["context"].lower().split())
        counter.update(sample["question"].lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def load_glove_from_multiple(glove_paths: List[str], word2idx: Dict[str, int], emb_dim: int) -> np.ndarray:
    """
    Load vectors from multiple GloVe-like text files.
    Earlier files get priority. Later files only fill words that are still missing.
    All files must use the same embedding dimension.
    """
    embeddings = np.random.normal(scale=0.02, size=(len(word2idx), emb_dim)).astype(np.float32)
    embeddings[word2idx["<pad>"]] = 0.0

    found_words = set()
    total_loaded = 0

    for glove_path in glove_paths:
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")

        file_loaded = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < emb_dim + 1:
                    continue

                word = parts[0]
                if word not in word2idx or word in found_words:
                    continue

                vector = np.asarray(parts[1:], dtype=np.float32)
                if vector.shape[0] != emb_dim:
                    continue

                embeddings[word2idx[word]] = vector
                found_words.add(word)
                file_loaded += 1
                total_loaded += 1

        print(f"Loaded {file_loaded} vocab items from {glove_path}")

    print(f"Loaded GloVe vectors for {total_loaded}/{len(word2idx)} vocab items from {len(glove_paths)} file(s)")
    return embeddings


# -----------------------------
# Dataset preprocessing
# -----------------------------

@dataclass
class QAFeature:
    input_context: torch.Tensor
    input_question: torch.Tensor
    start_pos: int
    end_pos: int
    context_tokens: List[str]
    example_id: str
    answers: Dict[str, List[Any]]
    context_text: str


class SquadBertDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer_name: str = "bert-base-uncased",
        max_context_len: int = 384,
        max_question_len: int = 64,
    ):
        self.data = hf_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.features = [self._build_feature(ex) for ex in self.data]

    def _build_feature(self, ex: Dict[str, Any]) -> QAFeature:
        context = ex["context"]
        question = ex["question"]
        answers = ex["answers"]
        example_id = ex["id"]

        enc_context = self.tokenizer(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_context_len,
            return_offsets_mapping=True,
        )
        enc_question = self.tokenizer(
            question,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_question_len,
            return_offsets_mapping=False,
        )

        context_ids = enc_context["input_ids"]
        question_ids = enc_question["input_ids"]
        offsets = enc_context["offset_mapping"]
        context_tokens = self.tokenizer.convert_ids_to_tokens(context_ids)

        answer_start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end_char = answer_start_char + len(answer_text)

        start_pos = 0
        end_pos = 0
        found_start = False

        for i, (start_char, end_char) in enumerate(offsets):
            if start_char <= answer_start_char < end_char and not found_start:
                start_pos = i
                found_start = True
            if start_char < answer_end_char <= end_char:
                end_pos = i
                break

        if not found_start:
            start_pos = 0
        if end_pos < start_pos:
            end_pos = start_pos

        return QAFeature(
            input_context=torch.tensor(context_ids, dtype=torch.long),
            input_question=torch.tensor(question_ids, dtype=torch.long),
            start_pos=start_pos,
            end_pos=end_pos,
            context_tokens=context_tokens,
            example_id=example_id,
            answers=answers,
            context_text=context,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> QAFeature:
        return self.features[idx]


class SquadStaticDataset(Dataset):
    def __init__(self, hf_split, word2idx: Dict[str, int], max_context_len: int = 384, max_question_len: int = 64):
        self.data = hf_split
        self.word2idx = word2idx
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.features = [self._build_feature(ex) for ex in self.data]

    def _encode_words(self, text: str, max_len: int) -> List[int]:
        toks = text.lower().split()[:max_len]
        return [self.word2idx.get(tok, self.word2idx["<unk>"]) for tok in toks]

    def _build_feature(self, ex: Dict[str, Any]) -> QAFeature:
        context = ex["context"]
        question = ex["question"]
        answers = ex["answers"]
        example_id = ex["id"]

        context_words = context.split()[: self.max_context_len]
        context_ids = self._encode_words(context, self.max_context_len)
        question_ids = self._encode_words(question, self.max_question_len)

        answer_start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end_char = answer_start_char + len(answer_text)

        start_pos = 0
        end_pos = 0
        current = 0
        for i, word in enumerate(context_words):
            start = current
            end = current + len(word)
            if start <= answer_start_char < end:
                start_pos = i
            if start < answer_end_char <= end:
                end_pos = i
                break
            current = end + 1

        if end_pos < start_pos:
            end_pos = start_pos

        return QAFeature(
            input_context=torch.tensor(context_ids, dtype=torch.long),
            input_question=torch.tensor(question_ids, dtype=torch.long),
            start_pos=start_pos,
            end_pos=end_pos,
            context_tokens=context_words,
            example_id=example_id,
            answers=answers,
            context_text=context,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> QAFeature:
        return self.features[idx]


@dataclass
class Batch:
    context_ids: torch.Tensor
    question_ids: torch.Tensor
    context_mask: torch.Tensor
    question_mask: torch.Tensor
    start_positions: torch.Tensor
    end_positions: torch.Tensor
    context_tokens: List[List[str]]
    example_ids: List[str]
    answers: List[Dict[str, List[Any]]]
    context_texts: List[str]


class QACollator:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, features: List[QAFeature]) -> Batch:
        context_ids = pad_sequence([f.input_context for f in features], batch_first=True, padding_value=self.pad_id)
        question_ids = pad_sequence([f.input_question for f in features], batch_first=True, padding_value=self.pad_id)

        context_mask = (context_ids != self.pad_id).long()
        question_mask = (question_ids != self.pad_id).long()

        start_positions = torch.tensor([f.start_pos for f in features], dtype=torch.long)
        end_positions = torch.tensor([f.end_pos for f in features], dtype=torch.long)

        return Batch(
            context_ids=context_ids,
            question_ids=question_ids,
            context_mask=context_mask,
            question_mask=question_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            context_tokens=[f.context_tokens for f in features],
            example_ids=[f.example_id for f in features],
            answers=[f.answers for f in features],
            context_texts=[f.context_text for f in features],
        )


# -----------------------------
# BiDAF-style models
# -----------------------------

class BiDAFAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, context: torch.Tensor, question: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        similarity = torch.bmm(context, question.transpose(1, 2))
        q_mask = question_mask.unsqueeze(1).expand_as(similarity)
        similarity = similarity.masked_fill(q_mask == 0, -1e9)
        c2q_attn = torch.softmax(similarity, dim=-1)
        attended_question = torch.bmm(c2q_attn, question)

        combined = torch.cat(
            [context, attended_question, context * attended_question, context - attended_question],
            dim=-1,
        )
        return combined


class BertBiDAF(nn.Module):
    def __init__(self, bert_name: str = "bert-base-uncased", hidden_dim: int = 128, dropout: float = 0.1, freeze_bert: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_dim = self.bert.config.hidden_size
        self.context_proj = nn.Linear(bert_dim, hidden_dim)
        self.question_proj = nn.Linear(bert_dim, hidden_dim)
        self.attention = BiDAFAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.modeling = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        return outputs.last_hidden_state

    def forward(self, context_ids: torch.Tensor, question_ids: torch.Tensor, context_mask: torch.Tensor, question_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context_emb = self.context_proj(self.encode(context_ids, context_mask))
        question_emb = self.question_proj(self.encode(question_ids, question_mask))
        context_emb = self.dropout(context_emb)
        question_emb = self.dropout(question_emb)
        fused = self.attention(context_emb, question_emb, question_mask)
        modeled, _ = self.modeling(fused)
        modeled = self.dropout(modeled)
        start_logits = self.start_head(modeled).squeeze(-1)
        end_logits = self.end_head(modeled).squeeze(-1)
        start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(context_mask == 0, -1e9)
        return start_logits, end_logits


class GloveBiDAF(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        emb_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )
        self.context_encoder = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.question_encoder = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.attention = BiDAFAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.modeling = nn.LSTM(
            input_size=(hidden_dim * 2) * 4,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, context_ids: torch.Tensor, question_ids: torch.Tensor, context_mask: torch.Tensor, question_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context_emb = self.embedding(context_ids)
        question_emb = self.embedding(question_ids)
        context_enc, _ = self.context_encoder(context_emb)
        question_enc, _ = self.question_encoder(question_emb)
        context_enc = self.dropout(context_enc)
        question_enc = self.dropout(question_enc)
        fused = self.attention(context_enc, question_enc, question_mask)
        modeled, _ = self.modeling(fused)
        modeled = self.dropout(modeled)
        start_logits = self.start_head(modeled).squeeze(-1)
        end_logits = self.end_head(modeled).squeeze(-1)
        start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(context_mask == 0, -1e9)
        return start_logits, end_logits


# -----------------------------
# Train / Eval
# -----------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0

    for batch in loader:
        context_ids = batch.context_ids.to(device)
        question_ids = batch.question_ids.to(device)
        context_mask = batch.context_mask.to(device)
        question_mask = batch.question_mask.to(device)
        start_positions = batch.start_positions.to(device)
        end_positions = batch.end_positions.to(device)

        start_logits, end_logits = model(context_ids, question_ids, context_mask, question_mask)
        loss_start = loss_fn(start_logits, start_positions)
        loss_end = loss_fn(end_logits, end_positions)
        loss = 0.5 * (loss_start + loss_end)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def decode_best_span(start_logits: torch.Tensor, end_logits: torch.Tensor) -> Tuple[int, int]:
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    seq_len = start_logits.size(0)
    best_score = -1.0
    best_span = (0, 0)

    for i in range(seq_len):
        max_j = min(seq_len, i + 30)
        for j in range(i, max_j):
            score = (start_probs[i] * end_probs[j]).item()
            if score > best_score:
                best_score = score
                best_span = (i, j)
    return best_span


def evaluate_model(model, loader, device):
    model.eval()
    squad_metric = evaluate.load("squad")
    predictions = []
    references = []

    with torch.no_grad():
        for batch in loader:
            context_ids = batch.context_ids.to(device)
            question_ids = batch.question_ids.to(device)
            context_mask = batch.context_mask.to(device)
            question_mask = batch.question_mask.to(device)

            start_logits, end_logits = model(context_ids, question_ids, context_mask, question_mask)

            for i in range(context_ids.size(0)):
                s_idx, e_idx = decode_best_span(start_logits[i].cpu(), end_logits[i].cpu())
                tokens = batch.context_tokens[i]
                if len(tokens) == 0:
                    pred_text = ""
                else:
                    s_idx = min(s_idx, len(tokens) - 1)
                    e_idx = min(e_idx, len(tokens) - 1)
                    if e_idx < s_idx:
                        e_idx = s_idx
                    pred_tokens = tokens[s_idx : e_idx + 1]
                    pred_text = " ".join(pred_tokens).replace(" ##", "")

                predictions.append({"id": batch.example_ids[i], "prediction_text": pred_text})
                references.append({"id": batch.example_ids[i], "answers": batch.answers[i]})

    results = squad_metric.compute(predictions=predictions, references=references)
    return results, predictions[:5]




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_type", choices=["bert", "glove"], default="bert")
    parser.add_argument("--bert_name", default="bert-base-uncased")
    parser.add_argument("--glove_path", nargs="+", default="glove.6B.100d.txt")
    parser.add_argument("--glove_dim", type=int, default=100)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_context_len", type=int, default=384)
    parser.add_argument("--max_question_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--save_dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("squad")
    train_split = dataset["train"].select(range(args.train_samples))
    val_split = dataset["validation"].select(range(args.val_samples))

    run_config = {
        "embedding_type": args.embedding_type,
        "bert_name": args.bert_name,
        "glove_paths": args.glove_paths,
        "glove_dim": args.glove_dim,
        "hidden_dim": args.hidden_dim,
        "max_context_len": args.max_context_len,
        "max_question_len": args.max_question_len,
        "freeze_bert": args.freeze_bert,
    }

    if args.embedding_type == "bert":
        train_ds = SquadBertDataset(train_split, tokenizer_name=args.bert_name, max_context_len=args.max_context_len, max_question_len=args.max_question_len)
        val_ds = SquadBertDataset(val_split, tokenizer_name=args.bert_name, max_context_len=args.max_context_len, max_question_len=args.max_question_len)
        collator = QACollator(pad_id=0)
        model = BertBiDAF(bert_name=args.bert_name, hidden_dim=args.hidden_dim, freeze_bert=args.freeze_bert)
    else:
        train_list = [train_split[i] for i in range(len(train_split))]
        word2idx = build_vocab(train_list)
        embedding_matrix = load_glove_from_multiple(args.glove_paths, word2idx, args.glove_dim)

        train_ds = SquadStaticDataset(train_split, word2idx=word2idx, max_context_len=args.max_context_len, max_question_len=args.max_question_len)
        val_ds = SquadStaticDataset(val_split, word2idx=word2idx, max_context_len=args.max_context_len, max_question_len=args.max_question_len)
        collator = QACollator(pad_id=0)
        model = GloveBiDAF(embedding_matrix=embedding_matrix, hidden_dim=args.hidden_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, sample_preds = evaluate_model(model, val_loader, device)
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "exact_match": val_metrics["exact_match"],
            "f1": val_metrics["f1"],
        }
        history.append(log_row)
        print(json.dumps(log_row, indent=2))
        print("Sample predictions:")
        for item in sample_preds[:3]:
            print(item)

    metrics_path = os.path.join(args.save_dir, f"{args.run_name}_{args.embedding_type}_metrics.json")
    model_path = os.path.join(args.save_dir, f"{args.run_name}_{args.embedding_type}.pt")
    config_path = os.path.join(args.save_dir, f"{args.run_name}_{args.embedding_type}_config.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    torch.save(model.state_dict(), model_path)
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved config to: {config_path}")

    if args.embedding_type == "glove":
        word2idx_path = os.path.join(args.save_dir, f"{args.run_name}_{args.embedding_type}_word2idx.pt")
        embedding_matrix_path = os.path.join(args.save_dir, f"{args.run_name}_{args.embedding_type}_embedding_matrix.pt")
        torch.save(word2idx, word2idx_path)
        torch.save(torch.tensor(embedding_matrix, dtype=torch.float32), embedding_matrix_path)
        print(f"Saved word2idx to: {word2idx_path}")
        print(f"Saved embedding matrix to: {embedding_matrix_path}")



if __name__ == "__main__":
    main()

