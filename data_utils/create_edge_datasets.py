"""
create_edge_datasets.py

Builds datasets for edge-probing-style tasks:
  - POS tagging (UD GUM)
  - Dependency relations (UD GUM)
  - NER (BIO, from GUM Entity in MISC)
  - Coreference (mention-pair classification from GUM Entity in MISC)
  - Constituents (simple BIO chunks from UPOS family)
  - SRL (Universal Propositions English-EWT)
  - SPR (semantic proto-roles; UDEWT wide/per-arg + PB)
  - Relation classification (SemEval-2010 Task 8 ZIP)

Outputs to data/.
"""

from __future__ import annotations
import os
import re
import io
import csv
import tarfile
import zipfile
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set

import requests
import pandas as pd
from tqdm import tqdm

try:
    from conllu import parse_incr
except Exception:
    parse_incr = None

# --------------------------
# Global config (set in main)
# --------------------------
MAX_ROWS: Optional[int] = 20000
RNG_SEED: int = 1337

def _assert_target_alignment(df: pd.DataFrame, label_for_log: str):
    """Fail loudly if `Target Index` does not point at `Word Form` under whitespace
    splitting -- the exact bug that made the extractor probe the wrong token."""
    if not {"Sentence", "Target Index", "Word Form"}.issubset(df.columns) or df.empty:
        return
    ok = tot = 0
    for _, r in df.iterrows():
        toks = str(r["Sentence"]).split()
        ti = int(r["Target Index"])
        tot += 1
        if 0 <= ti < len(toks) and toks[ti] == str(r["Word Form"]):
            ok += 1
    rate = ok / max(tot, 1)
    logging.info(f"ALIGN: {label_for_log}: split()[Target Index]==Word Form for {rate:.1%}")
    if rate < 0.999:
        raise ValueError(f"{label_for_log}: token alignment {rate:.1%} < 99.9% -- "
                         f"Target Index does not match Sentence.split(); extraction would probe wrong tokens.")


def _write_csv_capped(df: pd.DataFrame, out_path: Path, label_for_log: str):
    """Write df to csv, capping to MAX_ROWS with deterministic sampling if needed."""
    global MAX_ROWS, RNG_SEED
    _assert_target_alignment(df, label_for_log)
    n = len(df)
    if MAX_ROWS is not None and n > MAX_ROWS:
        # deterministic sampling
        df = df.sample(n=MAX_ROWS, random_state=RNG_SEED).reset_index(drop=True)
        logging.info(f"CAP: {label_for_log}: sampled {MAX_ROWS} rows from {n} -> {out_path.name}")
    else:
        logging.info(f"{label_for_log}: {n} rows -> {out_path.name}")
    df.to_csv(out_path, index=False)

# --------------------------
# Constants / URLs
# --------------------------

UD_REPO = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master"
UD_FILENAMES = {
    "train": "en_gum-ud-train.conllu",
    "dev":   "en_gum-ud-dev.conllu",
    "test":  "en_gum-ud-test.conllu",
}

UP_EWT_RAW_URL = "https://raw.githubusercontent.com/UniversalPropositions/UP-1.0/master/UP_English-EWT"
UP_FILENAMES = {
    "train": "en_ewt-up-train.conllu",
    "dev":   "en_ewt-up-dev.conllu",
    "test":  "en_ewt-up-test.conllu",
}

SPR_URLS = {
    "pb": "https://decomp.io/projects/semantic-proto-roles/protoroles_eng_pb.tar.gz",
    "udewt": "https://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz",
}

SEMEVAL2010_ZIP_URL = "https://github.com/JoelNiklaus/SemEval2010Task8/raw/main/SemEval2010_task8_all_data.zip"
SEMEVAL2010_TRAIN_REL = "SemEval2010_task8_training/TRAIN_FILE.TXT"
SEMEVAL2010_TEST_REL  = "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------
# Helpers
# --------------------------

def http_get(url: str, dest: Path, text: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logging.info(f"Downloading {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    if text:
        dest.write_text(resp.text, encoding="utf-8")
    else:
        dest.write_bytes(resp.content)
    return dest


def download_zip(url: str, dest_zip: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    http_get(url, dest_zip, text=False)
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(extract_to)
    return extract_to


def render_sentence(tokenlist) -> str:
    parts = []
    for tok in tokenlist:
        if not isinstance(tok["id"], int):
            continue
        form = tok["form"]
        misc = tok.get("misc") or {}
        parts.append(form)
        if not (misc.get("SpaceAfter") == "No"):
            parts.append(" ")
    return ("".join(parts)).rstrip()


def safe_get_misc(tok, key: str) -> Optional[str]:
    misc = tok.get("misc") or {}
    return misc.get(key)


def is_int_token(tok) -> bool:
    return isinstance(tok["id"], int)


def sent_text_and_tokens(tokenlist) -> Tuple[str, List[dict]]:
    # Sentence is stored as space-joined token forms so that `Sentence.split()`
    # recovers the exact UD tokens that `Target Index` / span offsets refer to.
    # (Detokenized text -- e.g. "Rude." -- would misalign whitespace-splitting with
    # the UD tokenization used for the indices; that silently probed the wrong token.)
    tokens = [tok for tok in tokenlist if is_int_token(tok)]
    for tok in tokens:
        tok["form"] = re.sub(r"\s+", "", str(tok["form"])) or "_"
    return " ".join(tok["form"] for tok in tokens), tokens


def doc_id_from_meta(tokenlist) -> Optional[str]:
    md = tokenlist.metadata or {}
    return md.get("newdoc id") or md.get("newdoc_id") or md.get("sent_id")


def sent_id_from_meta(tokenlist, fallback_idx: int) -> str:
    md = tokenlist.metadata or {}
    return md.get("sent_id") or f"{doc_id_from_meta(tokenlist) or 'doc'}::s{fallback_idx}"


# --------------------------
# MISC Entity parser (GUM)
# --------------------------

ENTITY_OPEN_RE = re.compile(r"^\((\d+)-([A-Za-z_]+)-")
ENTITY_CLOSE_RE = re.compile(r"^(\d+)\)$")

def extract_entity_spans_for_sentence(tokenlist) -> List[dict]:
    spans: List[dict] = []
    open_map: Dict[str, List[Tuple[int, str]]] = {}
    tokens = [tok for tok in tokenlist if is_int_token(tok)]
    for i, tok in enumerate(tokens):
        ent_field = safe_get_misc(tok, "Entity")
        if not ent_field:
            continue
        parts = str(ent_field).split("|")
        for p in parts:
            p = p.strip()
            m_open = ENTITY_OPEN_RE.match(p)
            m_close = ENTITY_CLOSE_RE.match(p)
            if m_open:
                g = m_open.group(1)
                etype = m_open.group(2)
                open_map.setdefault(g, []).append((i, etype))
            elif m_close:
                g = m_close.group(1)
                if g in open_map and open_map[g]:
                    start, etype = open_map[g].pop()
                    spans.append({"group": g, "etype": etype, "start": start, "end": i})
    return spans


def build_ner_bio_labels(tokens: List[dict], spans: List[dict]) -> List[str]:
    n = len(tokens)
    labels = ["O"] * n
    cover: List[List[Tuple[int,int,str]]] = [[] for _ in range(n)]
    for sp in spans:
        for i in range(sp["start"], sp["end"] + 1):
            cover[i].append((sp["start"], sp["end"], sp["etype"]))
    for i in range(n):
        if not cover[i]:
            continue
        s, e, et = min(cover[i], key=lambda t: (t[1]-t[0]+1, t[0]))
        labels[i] = "B-" + et if i == s else "I-" + et
    return labels


# --------------------------
# UD GUM -> POS / DEP / NER / Constituents / Coref
# --------------------------

def download_ud_gum(split: str) -> Path:
    assert split in UD_FILENAMES
    url = f"{UD_REPO}/{UD_FILENAMES[split]}"
    dest = DATA_DIR / UD_FILENAMES[split]
    return http_get(url, dest, text=True)


def conllu_iter(path: Path):
    if parse_incr is None:
        raise RuntimeError("conllu is not installed. pip install conllu")
    with path.open("r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            yield tokenlist


# tiny chunk family for a light "constituents" BIO
CHUNK_FAMILY = {
    "NOUN":"NP","PROPN":"NP","PRON":"NP","DET":"NP",
    "VERB":"VP","AUX":"VP",
    "ADP":"PP",
    "ADJ":"ADJP",
    "ADV":"ADVP",
    "NUM":"NUMP",
    "PART":"PRT",
}

def chunk_label_from_upos(upos: Optional[str]) -> Optional[str]:
    return CHUNK_FAMILY.get(upos or "")


def build_pos_dep_ner_const(split: str):
    src = download_ud_gum(split)
    pos_rows, dep_rows, ner_rows, const_rows = [], [], [], []

    for si, sent in enumerate(conllu_iter(src)):
        sentence_text, tokens = sent_text_and_tokens(sent)

        # POS + DEP
        for idx, tok in enumerate(tokens):
            upos = tok.get("upostag")
            deprel = tok.get("deprel")
            form = tok.get("form")
            lemma = tok.get("lemma")
            head = tok.get("head")

            if upos:
                pos_rows.append({
                    "Sentence": sentence_text,
                    "Target Index": idx,
                    "Label": upos,
                    "Word Form": form,
                    "Lemma": lemma,
                    "Source Type": f"UD_GUM_POS_{split}",
                })
            if deprel and head is not None:
                # Span1 is the dependent, Span2 is the head
                dep_rows.append({
                    "Sentence": sentence_text,
                    "Span1 Start": idx,
                    "Span1 End": idx + 1,
                    "Span2 Start": head - 1, # CONLLU is 1-indexed
                    "Span2 End": head,
                    "Label": deprel,
                    "Source Type": f"UD_GUM_DEP_{split}",
                })

        # NER BIO from Entity MISC
        spans = extract_entity_spans_for_sentence(sent)
        ner_tags = build_ner_bio_labels(tokens, spans)
        for idx, tok in enumerate(tokens):
            ner_rows.append({
                "Sentence": sentence_text,
                "Target Index": idx,
                "Label": ner_tags[idx],
                "Word Form": tok.get("form"),
                "Lemma": tok.get("lemma"),
                "Source Type": f"UD_GUM_NER_{split}",
            })

        # Constituents BIO from UPOS family
        fams = [chunk_label_from_upos(tok.get("upostag")) for tok in tokens]
        tags = []
        prev_f = None
        for fam in fams:
            if fam is None:
                tags.append("O")
                prev_f = None
            else:
                tags.append("B-"+fam if fam != prev_f else "I-"+fam)
                prev_f = fam
        for idx, tok in enumerate(tokens):
            const_rows.append({
                "Sentence": sentence_text,
                "Target Index": idx,
                "Label": tags[idx],
                "Word Form": tok.get("form"),
                "Lemma": tok.get("lemma"),
                "Source Type": f"UD_GUM_CONSTITUENTS_{split}",
            })

    _write_csv_capped(pd.DataFrame(pos_rows),   DATA_DIR / f"ud_gum_pos_{split}.csv",   f"Wrote POS for {split}")
    _write_csv_capped(pd.DataFrame(dep_rows),   DATA_DIR / f"ud_gum_dep_{split}.csv",   f"Wrote DEP for {split}")
    _write_csv_capped(pd.DataFrame(ner_rows),   DATA_DIR / f"ud_gum_ner_{split}.csv",   f"Wrote NER for {split}")
    _write_csv_capped(pd.DataFrame(const_rows),DATA_DIR / f"ud_gum_constituents_{split}.csv", f"Wrote Constituents for {split}")


def build_coref_pairs(split: str, negative_multiplier: float = 1.0, max_pairs_per_doc: int = 4000):
    src = download_ud_gum(split)
    rows = []

    docs: Dict[str, Dict[str, Any]] = {}
    running_doc_id = None
    sent_counter_in_doc = 0
    doc_token_offset = 0

    for si, sent in enumerate(conllu_iter(src)):
        sentence_text, tokens = sent_text_and_tokens(sent)
        doc_id = doc_id_from_meta(sent) or f"doc_{split}"
        sent_id = sent_id_from_meta(sent, si)

        if doc_id != running_doc_id:
            running_doc_id = doc_id
            sent_counter_in_doc = 0
            doc_token_offset = 0
            docs.setdefault(doc_id, {"sents": [], "spans": []})

        docs[doc_id]["sents"].append(sentence_text)

        spans = extract_entity_spans_for_sentence(sent)
        for sp in spans:
            docs[doc_id]["spans"].append({
                "group": sp["group"],
                "etype": sp["etype"],
                "sent_id": sent_counter_in_doc,
                "start": sp["start"] + doc_token_offset,
                "end": sp["end"] + doc_token_offset,
            })

        doc_token_offset += len(tokens)
        sent_counter_in_doc += 1

    rng = random.Random(RNG_SEED)
    for doc_id, data in docs.items():
        spans = data["spans"]
        chain_map: Dict[str, List[int]] = {}
        for i, sp in enumerate(spans):
            chain_map.setdefault(sp["group"], []).append(i)

        pos_pairs: List[Tuple[int, int]] = []
        for chain, idxs in chain_map.items():
            if len(idxs) < 2:  # singleton
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pos_pairs.append((idxs[i], idxs[j]))
            if len(pos_pairs) > max_pairs_per_doc:
                pos_pairs = rng.sample(pos_pairs, max_pairs_per_doc)

        all_indices = list(range(len(spans)))
        neg_pairs = set()
        target_negs = int(len(pos_pairs) * negative_multiplier)
        trials = 0
        while len(neg_pairs) < target_negs and trials < target_negs * 20:
            i, j = rng.sample(all_indices, 2)
            if spans[i]["group"] != spans[j]["group"]:
                neg_pairs.add((min(i, j), max(i, j)))
            trials += 1

        doc_text = " ".join(data["sents"]).strip()
        for i, j in pos_pairs:
            a, b = spans[i], spans[j]
            rows.append({
                "Text": doc_text,
                "Sentence": doc_text,
                "Span1 Start": a["start"], "Span1 End": a["end"] + 1,
                "Span2 Start": b["start"], "Span2 End": b["end"] + 1,
                "Label": 1,
                "Source Type": f"UD_GUM_COREF_{split}",
                "Doc ID": doc_id,
                "Sent1 ID": a["sent_id"], "Sent2 ID": b["sent_id"],
            })
        for i, j in neg_pairs:
            a, b = spans[i], spans[j]
            rows.append({
                "Text": doc_text,
                "Sentence": doc_text,
                "Span1 Start": a["start"], "Span1 End": a["end"] + 1,
                "Span2 Start": b["start"], "Span2 End": b["end"] + 1,
                "Label": 0,
                "Source Type": f"UD_GUM_COREF_{split}",
                "Doc ID": doc_id,
                "Sent1 ID": a["sent_id"], "Sent2 ID": b["sent_id"],
            })

    out = DATA_DIR / f"ud_gum_coref_pairs_{split}.csv"
    _write_csv_capped(pd.DataFrame(rows), out, f"Wrote COREF pairs for {split}")


# --------------------------
# SemEval-2010 from official ZIP
# --------------------------

E1_OPEN, E1_CLOSE = "<e1>", "</e1>"
E2_OPEN, E2_CLOSE = "<e2>", "</e2>"

def _strip_markers_and_get_spans(text: str) -> Tuple[str, Tuple[int,int], Tuple[int,int]]:
    clean_chars = []
    i = 0
    stack = []
    e1_range = None
    e2_range = None
    while i < len(text):
        if text.startswith(E1_OPEN, i):
            i += len(E1_OPEN); stack.append(("e1", len(clean_chars))); continue
        if text.startswith(E1_CLOSE, i):
            i += len(E1_CLOSE); _, s = stack.pop(); e1_range = (s, len(clean_chars)); continue
        if text.startswith(E2_OPEN, i):
            i += len(E2_OPEN); stack.append(("e2", len(clean_chars))); continue
        if text.startswith(E2_CLOSE, i):
            i += len(E2_CLOSE); _, s = stack.pop(); e2_range = (s, len(clean_chars)); continue
        clean_chars.append(text[i]); i += 1
    clean = "".join(clean_chars)
    clean_norm = re.sub(r"\s+", " ", clean).strip()
    if not e1_range or not e2_range:
        raise ValueError("Missing entity markers")
    e1_text = clean[e1_range[0]:e1_range[1]]
    e2_text = clean[e2_range[0]:e2_range[1]]
    def first_span(nt, frag):
        m = re.search(re.escape(re.sub(r"\s+", " ", frag).strip()), nt)
        return (m.start(), m.end()) if m else (0, 0)
    return clean_norm, first_span(clean_norm, e1_text), first_span(clean_norm, e2_text)


def _iter_official_blocks(path: Path):
    with path.open("r", encoding="iso-8859-1") as f:
        lines = f.readlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "":
            i += 1; continue
        block = lines[i:i+4]
        if len(block) < 2: break
        yield block
        i += 4


def _parse_header_line(line: str) -> Tuple[str, str]:
    parts = line.rstrip("\n").split("\t", 1)
    ex_id = parts[0].strip()
    sent_raw = parts[1].strip()
    sent = sent_raw[1:-1] if (len(sent_raw)>=2 and sent_raw[0]=='"' and sent_raw[-1]=='"') else sent_raw.strip('"')
    return ex_id, sent


def _char_to_word_span(text: str, cs: int, ce: int) -> Tuple[int, int]:
    """Map a character span [cs, ce) in a space-normalized string to word indices
    (start inclusive, end exclusive) so it aligns with `Sentence.split()`."""
    words = text.split()
    pos, ws, we = 0, None, None
    for i, w in enumerate(words):
        w0 = text.index(w, pos)
        w1 = w0 + len(w)
        pos = w1
        if w1 > cs and w0 < ce:                 # word overlaps the char span
            if ws is None:
                ws = i
            we = i + 1
    return (ws, we) if ws is not None else (0, 1)


def _parse_semeval_official(path: Path) -> List[dict]:
    rows = []
    for block in _iter_official_blocks(path):
        header = block[0]
        relation = block[1].strip()
        try:
            ex_id, sentence_w_markers = _parse_header_line(header)
            clean, e1_span, e2_span = _strip_markers_and_get_spans(sentence_w_markers)
        except Exception:
            continue
        # spans come back as CHARACTER offsets; extraction indexes words, so convert
        s1, e1 = _char_to_word_span(clean, e1_span[0], e1_span[1])
        s2, e2 = _char_to_word_span(clean, e2_span[0], e2_span[1])
        rows.append({
            "Text": clean, "Sentence": clean,
            "Span1 Start": s1, "Span1 End": e1,
            "Span2 Start": s2, "Span2 End": e2,
            "Label": relation, "Source Type": "SemEval2010",
            "Doc ID": ex_id, "Sent1 ID": 0, "Sent2 ID": 0
        })
    return rows


def build_relations_from_official_zip():
    cache_zip = DATA_DIR / "SemEval2010_task8_all_data.zip"
    extract_dir = DATA_DIR / "semeval2010_zip"
    download_zip(SEMEVAL2010_ZIP_URL, cache_zip, extract_dir)

    train_path = extract_dir / "SemEval2010_task8_all_data" / SEMEVAL2010_TRAIN_REL
    test_path  = extract_dir / "SemEval2010_task8_all_data" / SEMEVAL2010_TEST_REL
    if not train_path.exists() or not test_path.exists():
        train_path = None; test_path = None
        for p in extract_dir.rglob("*"):
            if p.name.upper() == "TRAIN_FILE.TXT": train_path = p
            if p.name.upper() == "TEST_FILE_FULL.TXT": test_path = p
    if not train_path or not test_path:
        raise FileNotFoundError(f"Could not locate SemEval files under: {extract_dir}")

    for split, path in (("train", train_path), ("test", test_path)):
        rows = _parse_semeval_official(path)
        out = DATA_DIR / f"semeval2010_relations_{split}.csv"
        _write_csv_capped(pd.DataFrame(rows), out, f"Wrote SemEval-2010 {split}")


# --------------------------
# UP-EWT SRL (robust)
# --------------------------

ARG_BIO_RE = re.compile(r"^(B|I)-(.+)$", re.IGNORECASE)

def download_up_ewt(split: str) -> Path:
    url = f"{UP_EWT_RAW_URL}/{UP_FILENAMES[split]}"
    dest = DATA_DIR / UP_FILENAMES[split]
    return http_get(url, dest, text=True)


def parse_conllup_srl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        sent_lines: List[str] = []
        columns_hint: List[str] = []
        doc_id = ""; sent_id = ""
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# global.columns"):
                    rhs = line.split("=", 1)[1].strip()
                    columns_hint = rhs.split()
                if line.startswith("# newdoc id"):
                    doc_id = line.split("=", 1)[1].strip()
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                sent_lines.append(line); continue
            if line == "":
                if any(l and not l.startswith("#") for l in sent_lines):
                    rows.extend(_conllup_sentence_to_srl_rows(sent_lines, columns_hint, doc_id, sent_id))
                sent_lines = []; continue
            sent_lines.append(line)
        if sent_lines and any(l and not l.startswith("#") for l in sent_lines):
            rows.extend(_conllup_sentence_to_srl_rows(sent_lines, columns_hint, doc_id, sent_id))
    return rows


def _is_neutral(val: str) -> bool:
    v = (val or "").strip()
    return (not v) or v in {"O", "_", "-", "*", ""}


def _conllup_sentence_to_srl_rows(lines: List[str], columns_hint: List[str],
                                  doc_id: str, sent_id: str) -> List[dict]:
    # Collect token rows, pad to per-sentence max columns
    raw_tokens: List[List[str]] = []
    for ln in lines:
        if not ln or ln.startswith("#"): continue
        cols = ln.split("\t")
        if len(cols) < 10: continue
        raw_tokens.append(cols)
    if not raw_tokens: return []

    max_cols = max(len(t) for t in raw_tokens)
    tokens = [(t + ["O"] * (max_cols - len(t))) if len(t) < max_cols else t for t in raw_tokens]

    forms = [t[1] for t in tokens]
    sentence_text = " ".join(forms).strip()

    extra_cols = list(range(10, max_cols))
    if not extra_cols: return []

    # Identify predicate/role columns:
    pred_cols: List[int] = []
    for j in extra_cols:
        col_vals = [t[j].strip() for t in tokens]
        if all(_is_neutral(v) for v in col_vals):
            continue
        # accept columns with any V/B-V/I-V or any non-neutral role string
        has_v = any(v in {"V", "B-V", "I-V"} or v.upper() in {"B-V", "I-V"} for v in col_vals)
        has_role = any((ARG_BIO_RE.match(v) or (not _is_neutral(v) and v.upper() != "V")) for v in col_vals)
        if has_v or has_role:
            pred_cols.append(j)
    if not pred_cols:  # nothing looks like SRL here
        return []

    out_rows: List[dict] = []
    for j in pred_cols:
        col_vals = [t[j].strip() for t in tokens]
        # predicate index: first explicit V, else first VERB/AUX, else 0
        pred_idx = None
        for i, v in enumerate(col_vals):
            if v in {"V", "B-V", "I-V"} or v.upper() in {"B-V", "I-V"}:
                pred_idx = i; break
        if pred_idx is None:
            for i, t in enumerate(tokens):
                upos = (t[3] if len(t) > 3 else "").upper()
                if upos in {"VERB", "AUX"}:
                    pred_idx = i; break
        if pred_idx is None:
            pred_idx = 0

        # Extract spans (BIO or singleton labels)
        i, n = 0, len(tokens)
        while i < n:
            lab = col_vals[i]
            m = ARG_BIO_RE.match(lab)
            if m and m.group(1).upper() == "B":
                role = m.group(2)
                start = i; i += 1
                while i < n:
                    lab2 = col_vals[i]; m2 = ARG_BIO_RE.match(lab2)
                    if not (m2 and m2.group(1).upper() == "I" and m2.group(2) == role): break
                    i += 1
                end = i
                out_rows.append({
                    "Sentence": sentence_text,
                    "Predicate Index": pred_idx,
                    "Arg Start": start, "Arg End": end,
                    "Label": role, "Source Type": "UP_EWT_SRL",
                    "Doc ID": doc_id, "Sent ID": sent_id,
                })
            elif (not _is_neutral(lab)) and lab.upper() != "V":
                # non-BIO singleton (e.g., "ARG1", "ARGM-TMP")
                role = lab
                out_rows.append({
                    "Sentence": sentence_text,
                    "Predicate Index": pred_idx,
                    "Arg Start": i, "Arg End": i+1,
                    "Label": role, "Source Type": "UP_EWT_SRL",
                    "Doc ID": doc_id, "Sent ID": sent_id,
                })
                i += 1
            else:
                i += 1
    return out_rows


def build_srl_up(split: str):
    src = download_up_ewt(split)
    rows = parse_conllup_srl(src)
    out = DATA_DIR / f"up_ewt_srl_{split}.csv"
    _write_csv_capped(pd.DataFrame(rows), out, f"Wrote SRL (UP-EWT) for {split}")


# --------------------------
# SPR (UD-EWT + PB)
# --------------------------

SPR1_PROPERTIES = {
    "awareness", "change_of_location", "change_of_state", "changes_possession",
    "created", "destroyed", "existed_after", "existed_before",
    "existed_during", "exists_as_physical", "instigation",
    "location_of_event", "makes_physical_contact", "manipulated_by_another",
    "predicate_changed_argument", "sentient", "stationary", "volition",
}

def download_and_extract(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    basename = url.split("/")[-1]
    archive_path = dest_dir / basename
    http_get(url, archive_path, text=False)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    return dest_dir

def _find_spr_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in {".tsv", ".csv"}]

def _parse_table(path: Path) -> Optional[pd.DataFrame]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, engine="python")
        return df if not df.empty else None
    except Exception:
        try:
            return pd.read_csv(path, sep=sep, engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
        except Exception:
            return None

def _looks_like_udewt_wide(df: pd.DataFrame) -> bool:
    has_core = {"predicate_index"}.issubset(df.columns) and \
               ("arg_span" in df.columns or {"Arg.Tokens.Begin","Arg.Tokens.End"}.issubset(df.columns))
    prop_hit = len(SPR1_PROPERTIES.intersection(df.columns)) >= 3
    return bool(has_core and prop_hit)

def _looks_like_per_arg(df: pd.DataFrame) -> bool:
    return {"Property","Response"}.issubset(df.columns) and (
        "arg_span" in df.columns or
        {"Arg.Tokens.Begin","Arg.Tokens.End"}.issubset(df.columns) or
        "Arg.Span" in df.columns or
        "Arg.Pos" in df.columns or
        "Arg.Head" in df.columns or
        "Arg_Head" in df.columns
    )

_ARG_POS_CANDIDATES = ("Arg.Pos", "Arg.Head", "Arg_Head", "Arg.Head.Index")

def _parse_headish(val) -> Optional[int]:
    """Parse various head index encodings into an int token index."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "":
        return None
    # plain integer
    if re.fullmatch(r"\d+", s):
        return int(s)
    # colon separated like '0:1' -> use the RHS
    m = re.fullmatch(r"\d+:(\d+)", s)
    if m:
        return int(m.group(1))
    # bracketed list like '[3,4]' -> take first
    m = re.fullmatch(r"\[(\d+(?:\s*,\s*\d+)*)\]", s)
    if m:
        first = m.group(1).split(",")[0].strip()
        if first.isdigit():
            return int(first)
    # CSV/semicolon -> take first number
    m = re.match(r"^\s*(\d+)", s)
    if m:
        return int(m.group(1))
    return None

def _norm_sentence(row: pd.Series, sent_col: Optional[str]) -> str:
    if sent_col and sent_col in row and isinstance(row[sent_col], str):
        val = row[sent_col]
        # some dumps store token list JSON in 'words' / 'sentence'
        try:
            js = json.loads(val)
            if isinstance(js, list):
                return " ".join(js).strip()
        except Exception:
            pass
        return val.strip()
    return " ".join(str(v) for v in row.values if isinstance(v, str)).strip()

def build_spr():
    all_rows = []

    for source, url in SPR_URLS.items():
        root = download_and_extract(url, DATA_DIR / f"spr_{source}")
        files = _find_spr_files(root)
        if not files:
            logging.warning(f"No SPR files found under {root}")
            continue

        candidates: List[pd.DataFrame] = []
        for p in files:
            df = _parse_table(p)
            if df is None or df.empty:
                continue
            cols = set(df.columns)
            if not (("Property" in cols) or (len(SPR1_PROPERTIES.intersection(cols)) >= 3)):
                continue
            if _looks_like_udewt_wide(df) or _looks_like_per_arg(df):
                candidates.append(df)

        if not candidates:
            logging.warning(f"[SPR {source}] Could not identify argument span tables; skipping.")
            continue

        df = pd.concat(candidates, ignore_index=True)

        # ---- detect core columns ----
        def pick(*cands):
            for c in cands:
                if c and c in df.columns:
                    return c
            return None

        sent_col = pick("sentence", "sent", "text", "words")
        pred_col = pick("predicate_index", "pred_idx", "pred_id", "pred", "Pred.Token", "predicate")

        # ---- derive spans (several schema variants) ----
        if "Arg.Tokens.Begin" in df.columns and "Arg.Tokens.End" in df.columns:
            df["Arg Start"] = pd.to_numeric(df["Arg.Tokens.Begin"], errors="coerce").astype("Int64")
            df["Arg End"]   = pd.to_numeric(df["Arg.Tokens.End"],   errors="coerce").astype("Int64")
            needs_fix = (df["Arg End"] <= df["Arg Start"])
            df.loc[needs_fix, "Arg End"] = df.loc[needs_fix, "Arg End"] + 1
        elif "arg_span" in df.columns:
            s = df["arg_span"].astype(str)
            df["Arg Start"] = pd.to_numeric(s.str.split("..").str[0], errors="coerce").astype("Int64")
            df["Arg End"]   = pd.to_numeric(s.str.split("..").str[1], errors="coerce").astype("Int64") + 1
        elif "Arg.Span" in df.columns:
            s = df["Arg.Span"].astype(str)
            df["Arg Start"] = pd.to_numeric(s.str.split("..").str[0], errors="coerce").astype("Int64")
            df["Arg End"]   = pd.to_numeric(s.str.split("..").str[1], errors="coerce").astype("Int64") + 1
        else:
            head_col = pick(*_ARG_POS_CANDIDATES)
            if head_col and head_col in df.columns:
                df["Arg Start"] = df[head_col].apply(_parse_headish).astype("Int64")
                df["Arg End"]   = df["Arg Start"] + 1
            else:
                logging.warning(f"[SPR {source}] No recognizable span columns; skipping.")
                continue

        df = df[df["Arg Start"].notna() & df["Arg End"].notna()].copy()
        if df.empty:
            logging.warning(f"[SPR {source}] All rows had invalid spans; skipping.")
            continue

        if pred_col and pred_col in df.columns:
            df["Predicate Index"] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)
        else:
            df["Predicate Index"] = 0

        df["Sentence"] = df.apply(lambda r: _norm_sentence(r, sent_col), axis=1)
        out_chunks = []

        if _looks_like_udewt_wide(df):
            used_cols = {"Sentence","Predicate Index","Arg Start","Arg End",
                         sent_col, pred_col, "arg_span","Arg.Tokens.Begin","Arg.Tokens.End"}
            prop_cols = [c for c in df.columns if (c not in used_cols) and (c in SPR1_PROPERTIES)]
            for prop in prop_cols:
                sub = df[["Sentence","Predicate Index","Arg Start","Arg End", prop]].copy()
                sub = sub.rename(columns={prop: "Value"})
                def to_float(x):
                    try: return float(str(x).replace(",", ""))
                    except Exception: return None
                vals = sub["Value"].apply(to_float)
                sub["Label"] = (vals >= 0.5).astype(int) if vals.notna().any() else sub["Value"]
                sub["Property"] = prop
                sub["Source Type"] = f"SPR_{source}"
                out_chunks.append(sub)

        if _looks_like_per_arg(df) and "Property" in df.columns:
            long_df = df.copy()
            long_df["Property"] = long_df["Property"].astype(str)
            def resp_to_val(x):
                s = str(x).strip().lower()
                if s in {"", "nan", "none"}: return None
                if s in {"y","yes","true"}: return 1.0
                if s in {"n","no","false"}: return 0.0
                try: return float(s)
                except Exception: return None
            resp_col = "Response" if "Response" in long_df.columns else "Value" if "Value" in long_df.columns else None
            if resp_col:
                long_df["Value"] = long_df[resp_col].apply(resp_to_val)
            else:
                long_df["Value"] = None
            long_df["Label"] = long_df["Value"].apply(lambda v: int(v >= 0.5) if isinstance(v, (int,float)) else None)
            sub = long_df[["Sentence","Predicate Index","Arg Start","Arg End","Property","Value","Label"]].copy()
            sub["Source Type"] = f"SPR_{source}"
            out_chunks.append(sub)

        if out_chunks:
            merged = pd.concat(out_chunks, ignore_index=True)
            all_rows.append(merged)
        else:
            logging.warning(f"[SPR {source}] No usable property columns found.")

    if all_rows:
        merged_all = pd.concat(all_rows, ignore_index=True)
        # Filter to only SPR1 properties
        spr1_df = merged_all[merged_all['Property'].isin(SPR1_PROPERTIES)].copy()
        _write_csv_capped(spr1_df, DATA_DIR / "spr_all_properties.csv", "Wrote merged SPR (SPR1 only) file")

TASKS = ["pos", "dep", "ner", "coref", "constituents", "srl", "spr", "relation"]

def main():
    global MAX_ROWS, RNG_SEED
    ap = argparse.ArgumentParser(description="Create edge-probing datasets (UD GUM, UP-EWT, SPR, SemEval2010).")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    ap.add_argument("--splits", nargs="+", choices=["train","dev","test"], default=["train","dev","test"])
    ap.add_argument("--coref_neg_mult", type=float, default=1.0)
    ap.add_argument("--max_rows", type=int, default=20000, help="Cap each output CSV to at most this many rows (per file). Use 0 to disable.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed for caps.")
    args = ap.parse_args()

    RNG_SEED = args.seed
    MAX_ROWS = None if (args.max_rows is None or args.max_rows <= 0) else int(args.max_rows)

    if any(t in args.tasks for t in ("pos","dep","ner","constituents","coref")):
        for split in args.splits:
            if any(t in args.tasks for t in ("pos","dep","ner","constituents")):
                build_pos_dep_ner_const(split)
            if "coref" in args.tasks:
                build_coref_pairs(split, negative_multiplier=args.coref_neg_mult)

    if "relation" in args.tasks:
        build_relations_from_official_zip()

    if "srl" in args.tasks:
        for split in args.splits:
            build_srl_up(split)

    if "spr" in args.tasks:
        build_spr()

    logging.info("All requested datasets have been generated.")


if __name__ == "__main__":
    main()