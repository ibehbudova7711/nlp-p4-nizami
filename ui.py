import os
import re
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer

from nlpp4 import BertBiDAF, GloveBiDAF


st.set_page_config(page_title="BiDAF QA Demo", layout="wide")

BERT_NAME = "/Users/aliyevamehriban/bert_local"
MAX_CONTEXT_LEN = 384
MAX_QUESTION_LEN = 64
HIDDEN_DIM = 128
DROPOUT = 0.1


def clean_bert_tokens(tokens: List[str]) -> str:
    text = " ".join(tokens)
    text = text.replace(" ##", "")
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    return text.strip()


def decode_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_len: int = 30,
) -> Tuple[int, int, float]:
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    seq_len = start_logits.size(0)

    best_score = -1.0
    best_span = (0, 0)
    for i in range(seq_len):
        max_j = min(seq_len, i + max_answer_len)
        for j in range(i, max_j):
            score = (start_probs[i] * end_probs[j]).item()
            if score > best_score:
                best_score = score
                best_span = (i, j)
    return best_span[0], best_span[1], best_score


@st.cache_resource(show_spinner=False)
def load_bert_model(model_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BERT .pt file not found: {model_path}")

    model = BertBiDAF(
        bert_name=BERT_NAME,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        freeze_bert=False,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME, use_fast=True)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_glove_resources(model_path: str, emb_path: str, vocab_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GloVe .pt file not found: {model_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    word2idx = torch.load(vocab_path, map_location="cpu")
    if not isinstance(word2idx, dict):
        raise ValueError("The vocab file must contain a Python dict named word2idx.")

    embedding_matrix = torch.load(emb_path, map_location="cpu")
    if isinstance(embedding_matrix, torch.Tensor):
        embedding_matrix = embedding_matrix.cpu().numpy()
    elif isinstance(embedding_matrix, list):
        embedding_matrix = np.asarray(embedding_matrix, dtype=np.float32)
    elif not isinstance(embedding_matrix, np.ndarray):
        raise ValueError("The embedding file must contain a numpy array, list, or torch tensor.")

    model = GloveBiDAF(
        embedding_matrix=embedding_matrix,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, word2idx


def encode_glove_text(text: str, word2idx: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, List[str]]:
    tokens = text.split()[:max_len]
    ids = [word2idx.get(tok.lower(), word2idx.get("<unk>", 1)) for tok in tokens]
    if not ids:
        ids = [word2idx.get("<pad>", 0)]
    return torch.tensor([ids], dtype=torch.long), tokens


def predict_with_bert(model: torch.nn.Module, tokenizer, question: str, context: str, device: str):
    enc_context = tokenizer(
        context,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_CONTEXT_LEN,
        return_offsets_mapping=True,
    )
    enc_question = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_QUESTION_LEN,
        return_offsets_mapping=False,
    )

    context_ids = torch.tensor([enc_context["input_ids"]], dtype=torch.long).to(device)
    question_ids = torch.tensor([enc_question["input_ids"]], dtype=torch.long).to(device)
    context_mask = (context_ids != 0).long().to(device)
    question_mask = (question_ids != 0).long().to(device)

    with torch.no_grad():
        start_logits, end_logits = model(context_ids, question_ids, context_mask, question_mask)

    s_idx, e_idx, score = decode_best_span(start_logits[0].cpu(), end_logits[0].cpu())
    context_tokens = tokenizer.convert_ids_to_tokens(enc_context["input_ids"])
    if not context_tokens:
        return "", 0, 0, 0.0, []

    s_idx = min(s_idx, len(context_tokens) - 1)
    e_idx = min(e_idx, len(context_tokens) - 1)
    if e_idx < s_idx:
        e_idx = s_idx

    answer_tokens = context_tokens[s_idx : e_idx + 1]
    answer_text = clean_bert_tokens(answer_tokens)
    return answer_text, s_idx, e_idx, score, context_tokens


def predict_with_glove(model: torch.nn.Module, word2idx: Dict[str, int], question: str, context: str, device: str):
    context_ids, context_tokens = encode_glove_text(context, word2idx, max_len=MAX_CONTEXT_LEN)
    question_ids, _ = encode_glove_text(question, word2idx, max_len=MAX_QUESTION_LEN)

    context_ids = context_ids.to(device)
    question_ids = question_ids.to(device)
    context_mask = (context_ids != 0).long().to(device)
    question_mask = (question_ids != 0).long().to(device)

    with torch.no_grad():
        start_logits, end_logits = model(context_ids, question_ids, context_mask, question_mask)

    s_idx, e_idx, score = decode_best_span(start_logits[0].cpu(), end_logits[0].cpu())
    if not context_tokens:
        return "", 0, 0, 0.0, []

    s_idx = min(s_idx, len(context_tokens) - 1)
    e_idx = min(e_idx, len(context_tokens) - 1)
    if e_idx < s_idx:
        e_idx = s_idx

    answer_text = " ".join(context_tokens[s_idx : e_idx + 1]).strip()
    return answer_text, s_idx, e_idx, score, context_tokens


def highlight_span(tokens: List[str], start_idx: int, end_idx: int) -> str:
    html_parts = []
    for i, tok in enumerate(tokens):
        safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
        if start_idx <= i <= end_idx:
            html_parts.append(
                f"<span style='background-color:#fff3a3;padding:2px 4px;border-radius:4px'>{safe_tok}</span>"
            )
        else:
            html_parts.append(safe_tok)
    return " ".join(html_parts)


st.title("Question Answering with BiDAF")
st.caption("Run the BERT-based and GloVe-based BiDAF models saved as .pt files.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.header("Model files")
st.sidebar.write(f"Running on: **{device}**")

bert_model_path = st.sidebar.text_input("BERT model .pt path", value="/Users/aliyevamehriban/Downloads/bert-output/run_bert.pt")
glove_model_path = st.sidebar.text_input("GloVe model .pt path", value="run_glove-2.pt")
glove_emb_path = st.sidebar.text_input("GloVe embedding matrix path", value="run_glove_embedding_matrix.pt")
glove_vocab_path = st.sidebar.text_input("GloVe word2idx path", value="run_glove_word2idx.pt")

example_context = "Baku is the capital and largest city of Azerbaijan."
example_question = "What is the capital of Azerbaijan?"

col1, col2 = st.columns(2)
with col1:
    question = st.text_input("Question", value=example_question)
with col2:
    compare_mode = st.checkbox("Run both models", value=True)

context = st.text_area("Context", value=example_context, height=220)
run_btn = st.button("Get answer", type="primary")

single_model_choice = st.radio(
    "Choose one model when compare mode is off",
    ["BiDAF + BERT embeddings", "BiDAF + GloVe embeddings"],
    horizontal=True,
    disabled=compare_mode,
)

if run_btn:
    if not question.strip() or not context.strip():
        st.error("Please enter both a question and a context.")
    else:
        results = []

        if compare_mode or single_model_choice == "BiDAF + BERT embeddings":
            try:
                with st.spinner("Loading BERT BiDAF model..."):
                    bert_model, bert_tokenizer = load_bert_model(bert_model_path, device)
                with st.spinner("Running BERT BiDAF..."):
                    answer, s_idx, e_idx, score, toks = predict_with_bert(
                        bert_model, bert_tokenizer, question, context, device
                    )
                results.append(("BiDAF + BERT embeddings", answer, s_idx, e_idx, score, toks))
            except Exception as e:
                st.error(f"BERT model failed: {e}")

        if compare_mode or single_model_choice == "BiDAF + GloVe embeddings":
            try:
                with st.spinner("Loading GloVe BiDAF model..."):
                    glove_model, word2idx = load_glove_resources(
                        glove_model_path, glove_emb_path, glove_vocab_path, device
                    )
                with st.spinner("Running GloVe BiDAF..."):
                    answer, s_idx, e_idx, score, toks = predict_with_glove(
                        glove_model, word2idx, question, context, device
                    )
                results.append(("BiDAF + GloVe embeddings", answer, s_idx, e_idx, score, toks))
            except Exception as e:
                st.error(f"GloVe model failed: {e}")

        if results:
            st.subheader("Results")
            for name, answer, s_idx, e_idx, score, toks in results:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.write(f"Predicted answer: **{answer or '[empty]'}**")
                    st.write(f"Start index: `{s_idx}` | End index: `{e_idx}` | Span score: `{score:.6f}`")
                    if toks:
                        st.markdown("**Highlighted context tokens**")
                        st.markdown(highlight_span(toks, s_idx, e_idx), unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "**This version expects `.pt` files**  \n"
    "- BERT model file: a `state_dict` saved with `torch.save(model.state_dict(), path)`  \n"
    "- GloVe model file: a `state_dict` saved with `torch.save(model.state_dict(), path)`  \n"
)
st.code("streamlit run ui_pt.py", language="bash")
