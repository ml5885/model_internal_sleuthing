"""
create_multilingual_edge_datasets.py

Builds edge-probing datasets for multilingual experiments across five languages:
  Chinese (zh), Turkish (tr), French (fr), Russian (ru), German (de)

Tasks:
  - POS tagging (Universal Dependencies treebanks)
  - Dependency parsing (Universal Dependencies treebanks)
  - Constituency parsing (UD-derived chunks, same method as English)
  - NER (various sources per language, loaded via HuggingFace or custom)
  - SRL (various sources per language)
  - Coreference (various sources per language)
  - Relation extraction (REDFM for fr/de, DuIE for zh)

Outputs to data/ with naming convention: {source}_{lang}_{task}_{split}.csv
"""

from __future__ import annotations
import io
import os
import re
import json
import random
import logging
import argparse
import zipfile
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

try:
    from conllu import parse_incr
except ImportError:
    parse_incr = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MAX_ROWS: Optional[int] = 20000
RNG_SEED: int = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Language / dataset configuration
# ============================================================

LANG_NAMES = {
    "zh": "Chinese", "tr": "Turkish", "fr": "French",
    "ru": "Russian", "de": "German",
}

# Universal Dependencies treebanks (used for POS, DEP, Constituents)
UD_CONFIGS = {
    "zh": {"repo": "UD_Chinese-GSDSimp", "prefix": "zh_gsdsimp-ud"},
    "tr": {"repo": "UD_Turkish-BOUN", "prefix": "tr_boun-ud"},
    "fr": {"repo": "UD_French-GSD", "prefix": "fr_gsd-ud"},
    "ru": {"repo": "UD_Russian-SynTagRus", "prefix": "ru_syntagrus-ud"},
    "de": {"repo": "UD_German-HDT", "prefix": "de_hdt-ud"},
}

# NER datasets (HuggingFace hub IDs or custom loaders)
NER_CONFIGS = {
    "zh": {
        # parquet mirror of MSRA NER (levow/msra_ner is a script-only repo that
        # `datasets`>=3 can no longer load). NB: the mirror's tag order differs
        # from the original repo's.
        "hf_dataset": "PassbyGrocer/msra-ner",
        "token_field": "tokens", "tag_field": "ner_tags",
        "tag_names": ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"],
    },
    "tr": {
        "hf_dataset": "turkish-nlp-suite/turkish-wikiNER",
        "token_field": "tokens", "tag_field": "tags",
        "tags_are_strings": True,  # Tags are already string BIO labels
    },
    "fr": {
        "hf_dataset": "danrun/WikiNER-fr-gold",
        "token_field": "tokens", "tag_field": "ner_tags",
        "tag_names": ["O", "B-PER", "I-PER", "E-PER", "S-PER",
                      "B-LOC", "I-LOC", "E-LOC", "S-LOC",
                      "B-ORG", "I-ORG", "E-ORG", "S-ORG",
                      "B-MISC", "I-MISC", "E-MISC", "S-MISC"],
        "skip_splits": ["dev", "test"],  # Only has train split
        "convert_bioes_to_bio": True,
    },
    "de": {
        # GermanEval/germeval_14 is script-only on HF; fetch the raw GermEval-2014
        # TSVs the script itself points at (Google Drive, small direct downloads).
        "gdrive_tsv": {
            "train": "https://drive.google.com/uc?export=download&id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P",
            "dev": "https://drive.google.com/uc?export=download&id=1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm",
            "test": "https://drive.google.com/uc?export=download&id=1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH",
        },
    },
    "ru": {"source": "nerel"},  # Custom loader for NEREL
}

# REDFM relation extraction (HuggingFace, with language configs)
REDFM_LANGS = {"fr": "fr", "de": "de"}

# SRL datasets - Universal Propositions provides multilingual SRL in CoNLL-U+ format.
# Turkish and Russian are not available in UP.
SRL_CONFIGS = {
    "zh": {"up_repo": "UP_Chinese", "up_prefix": "zh-up"},
    "fr": {"up_repo": "UP_French", "up_prefix": "fr-up"},
    "de": {"up_repo": "UP_German", "up_prefix": "de-up"},
}

# CorefUD treebanks (from local zip file)
# Each language may have multiple treebanks; we pick the largest / most standard one.
COREFUD_CONFIGS = {
    "tr": {"treebank": "CorefUD_Turkish-ITCC", "prefix": "tr_itcc-corefud"},
    "fr": {"treebank": "CorefUD_French-Democrat", "prefix": "fr_democrat-corefud"},
    "de": {"treebank": "CorefUD_German-PotsdamCC", "prefix": "de_potsdamcc-corefud"},
}
COREFUD_ZIP = "Coreference in Universal Dependencies 1.4 (CorefUD 1.4).zip"
COREFUD_INNER_ZIP = "CorefUD-1.4-public.zip"

# RuCoCo Russian coreference (from local zip file)
RUCOCO_ZIP = "v1.0.0.zip"

# DuIE 2.0 Chinese relation extraction (from local zip file)
DUIE_ZIP = "archive.zip"

TASKS = ["pos", "dep", "constituents", "ner", "srl", "coref", "relation"]
LANGUAGES = ["zh", "tr", "fr", "ru", "de"]


# ============================================================
# Shared helpers
# ============================================================

def _assert_target_alignment(df: pd.DataFrame, label_for_log: str):
    """Fail loudly if `Target Index` does not point at `Word Form` under whitespace
    splitting -- guards against the token-vs-detokenized misalignment bug."""
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
                         f"extraction would probe the wrong tokens.")


def _write_csv_capped(df: pd.DataFrame, out_path: Path, label_for_log: str):
    _assert_target_alignment(df, label_for_log)
    n = len(df)
    if MAX_ROWS is not None and n > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RNG_SEED).reset_index(drop=True)
        logging.info(f"CAP: {label_for_log}: sampled {MAX_ROWS} rows from {n} -> {out_path.name}")
    else:
        logging.info(f"{label_for_log}: {n} rows -> {out_path.name}")
    df.to_csv(out_path, index=False)


def http_get(url: str, dest: Path, text: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logging.info(f"Downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    if text:
        dest.write_text(resp.text, encoding="utf-8")
    else:
        dest.write_bytes(resp.content)
    return dest


def conllu_iter(path: Path):
    if parse_incr is None:
        raise RuntimeError("conllu is not installed. pip install conllu")
    with path.open("r", encoding="utf-8") as f:
        yield from parse_incr(f)


def is_int_token(tok) -> bool:
    return isinstance(tok["id"], int)


def render_sentence(tokenlist) -> str:
    parts = []
    for tok in tokenlist:
        if not isinstance(tok["id"], int):
            continue
        parts.append(tok["form"])
        misc = tok.get("misc") or {}
        if not (misc.get("SpaceAfter") == "No"):
            parts.append(" ")
    return "".join(parts).rstrip()


def sent_text_and_tokens(tokenlist):
    # Space-join token forms so Sentence.split() recovers the UD tokens that
    # Target Index / span offsets index into. Collapse whitespace *inside* a form
    # (e.g. the French thousands-separator token "10 000") first, so a single
    # token cannot split into two and shift every downstream index.
    tokens = [tok for tok in tokenlist if is_int_token(tok)]
    for tok in tokens:
        tok["form"] = re.sub(r"\s+", "", str(tok["form"])) or "_"
    return " ".join(tok["form"] for tok in tokens), tokens


# Chunk family mapping (same as English)
CHUNK_FAMILY = {
    "NOUN": "NP", "PROPN": "NP", "PRON": "NP", "DET": "NP",
    "VERB": "VP", "AUX": "VP",
    "ADP": "PP",
    "ADJ": "ADJP",
    "ADV": "ADVP",
    "NUM": "NUMP",
    "PART": "PRT",
}


# ============================================================
# UD-based tasks: POS, DEP, Constituents
# ============================================================

def _download_ud_conllu(lang: str, split: str) -> list[Path]:
    """Download UD CoNLL-U file(s) for a language/split. Returns list of paths
    (some large treebanks split train into multiple files)."""
    cfg = UD_CONFIGS[lang]
    repo = cfg["repo"]
    prefix = cfg["prefix"]
    base_url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/master"
    cache_dir = DATA_DIR / f"raw_ud_{lang}"

    # Try the standard single-file pattern first
    filename = f"{prefix}-{split}.conllu"
    dest = cache_dir / filename
    try:
        return [http_get(f"{base_url}/{filename}", dest, text=True)]
    except requests.HTTPError:
        pass

    # Large treebanks split train files. Try multiple naming conventions:
    # Russian SynTagRus: train-a, train-b, train-c
    # German HDT: train-a-1, train-a-2, train-b-1, train-b-2
    paths = []
    for suffix in ["a", "b", "c", "d", "e", "f"]:
        # Try simple suffix (e.g., train-a)
        filename = f"{prefix}-{split}-{suffix}.conllu"
        dest = cache_dir / filename
        try:
            paths.append(http_get(f"{base_url}/{filename}", dest, text=True))
        except requests.HTTPError:
            # Try numbered sub-splits (e.g., train-a-1, train-a-2)
            found_sub = False
            for num in range(1, 5):
                filename = f"{prefix}-{split}-{suffix}-{num}.conllu"
                dest = cache_dir / filename
                try:
                    paths.append(http_get(f"{base_url}/{filename}", dest, text=True))
                    found_sub = True
                except requests.HTTPError:
                    break
            if not found_sub and not paths:
                continue
            elif not found_sub:
                break

    if not paths:
        raise FileNotFoundError(
            f"Could not download UD {split} for {lang} from {base_url}."
        )
    return paths


def build_ud_tasks(lang: str, split: str):
    """Build POS, DEP, and Constituents datasets from a UD treebank."""
    paths = _download_ud_conllu(lang, split)
    pos_rows, dep_rows, const_rows = [], [], []
    source_prefix = UD_CONFIGS[lang]["repo"].replace("UD_", "")

    for path in paths:
        for sent in conllu_iter(path):
            sentence_text, tokens = sent_text_and_tokens(sent)
            if not tokens:
                continue

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
                        "Source Type": f"UD_{source_prefix}_POS_{split}",
                    })

                if deprel and head is not None:
                    head_idx = head - 1  # CoNLL-U is 1-indexed
                    dep_rows.append({
                        "Sentence": sentence_text,
                        "Span1 Start": idx,
                        "Span1 End": idx + 1,
                        "Span2 Start": head_idx,
                        "Span2 End": head_idx + 1,
                        "Label": deprel,
                        "Source Type": f"UD_{source_prefix}_DEP_{split}",
                    })

            # Constituents (BIO from UPOS family)
            fams = [CHUNK_FAMILY.get(tok.get("upostag") or "") for tok in tokens]
            prev_f = None
            tags = []
            for fam in fams:
                if fam is None:
                    tags.append("O")
                    prev_f = None
                else:
                    tags.append("B-" + fam if fam != prev_f else "I-" + fam)
                    prev_f = fam
            for idx, tok in enumerate(tokens):
                const_rows.append({
                    "Sentence": sentence_text,
                    "Target Index": idx,
                    "Label": tags[idx],
                    "Word Form": tok.get("form"),
                    "Lemma": tok.get("lemma"),
                    "Source Type": f"UD_{source_prefix}_CONSTITUENTS_{split}",
                })

    prefix = DATASET_REGISTRY[(lang, "pos")]
    _write_csv_capped(pd.DataFrame(pos_rows), DATA_DIR / f"{prefix}_pos_{split}.csv",
                      f"POS {lang}/{split}")
    _write_csv_capped(pd.DataFrame(dep_rows), DATA_DIR / f"{prefix}_dep_{split}.csv",
                      f"DEP {lang}/{split}")
    _write_csv_capped(pd.DataFrame(const_rows), DATA_DIR / f"{prefix}_constituents_{split}.csv",
                      f"Constituents {lang}/{split}")


# ============================================================
# NER from HuggingFace datasets
# ============================================================

def _resolve_tag_names(ds, tag_field: str, configured_names: list[str] | None) -> list[str] | None:
    """Get BIO tag string names from HuggingFace dataset features."""
    if configured_names is not None:
        return configured_names
    features = ds.features
    if tag_field in features:
        feat = features[tag_field]
        # Sequence(ClassLabel(...))
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            return feat.feature.names
    return None


def _bioes_to_bio(label: str) -> str:
    """Convert BIOES/BILOU tag to BIO-2 scheme."""
    if label == "O":
        return "O"
    prefix, etype = label[0], label[2:]
    if prefix == "S":
        return f"B-{etype}"
    if prefix == "E":
        return f"I-{etype}"
    return label  # B and I stay the same


def _build_ner_germeval_tsv(lang: str, split: str, urls: dict):
    """Build NER rows from GermEval-2014 style TSVs (idx, token, outer BIO, inner BIO;
    '#' comment lines start a sentence block, blank lines end it)."""
    url = urls.get(split)
    if not url:
        return
    dest = DATA_DIR / f"raw_ner_{lang}" / f"{split}.tsv"
    path = http_get(url, dest, text=True)

    rows, tokens, tags = [], [], []

    def flush():
        if not tokens:
            return
        sentence = " ".join(tokens)
        for idx, (tok, tag) in enumerate(zip(tokens, tags)):
            rows.append({"Sentence": sentence, "Target Index": idx, "Label": tag,
                         "Word Form": tok, "Lemma": tok,
                         "Source Type": f"NER_{lang}_{split}"})
        tokens.clear(); tags.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        tok = re.sub(r"\s+", "", cols[1]) or "_"   # keep split() alignment
        tokens.append(tok)
        tags.append(cols[2].strip() or "O")
    flush()

    if rows:
        prefix = DATASET_REGISTRY[(lang, "ner")]
        _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_ner_{split}.csv",
                          f"NER {lang}/{split}")
    else:
        logging.warning(f"NER {lang}/{split}: no rows from TSV.")


def build_ner_hf(lang: str, split: str):
    """Build NER dataset from a HuggingFace token classification dataset."""
    cfg = NER_CONFIGS[lang]
    if "gdrive_tsv" in cfg:
        return _build_ner_germeval_tsv(lang, split, cfg["gdrive_tsv"])
    if load_dataset is None:
        raise RuntimeError("datasets library not installed. pip install datasets")

    if "hf_dataset" not in cfg:
        logging.warning(f"NER for {lang}: no HuggingFace dataset configured, skipping.")
        return

    if split in cfg.get("skip_splits", []):
        logging.info(f"NER {lang}: split '{split}' not available, skipping.")
        return

    hf_name = cfg["hf_dataset"]
    token_field = cfg["token_field"]
    tag_field = cfg["tag_field"]
    tags_are_strings = cfg.get("tags_are_strings", False)
    convert_bioes = cfg.get("convert_bioes_to_bio", False)

    hf_split_map = {"train": "train", "dev": "validation", "test": "test"}
    hf_split = hf_split_map.get(split, split)

    ds = None
    try:
        ds = load_dataset(hf_name, split=hf_split, trust_remote_code=True)
    except Exception:
        pass

    # Fallback: load parquet directly (e.g., WikiNER-fr-gold has a broken loader)
    if ds is None:
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
            parquet_path = hf_hub_download(repo_id=hf_name, filename="data/train-00000-of-00001.parquet",
                                           repo_type="dataset")
            table = pq.read_table(parquet_path)
            ds = table.to_pandas()
            # Convert DataFrame to list-of-dicts iterable
            ds = ds.to_dict("records")
            tags_are_strings = False  # parquet datasets use int tags
        except Exception as e2:
            logging.warning(f"NER {lang}: could not load {hf_name} (split={hf_split}): {e2}")
            return

    tag_names = cfg.get("tag_names")
    if tag_names is None and not tags_are_strings:
        if hasattr(ds, "features"):
            tag_names = _resolve_tag_names(ds, tag_field, None)

    rows = []
    for example in ds:
        tokens = example[token_field]
        tags = example[tag_field]
        if not tokens or not tags:
            continue

        tokens = [re.sub(r"\s+", "", str(t)) or "_" for t in tokens]  # split() alignment
        sentence = " ".join(tokens)
        for idx, (tok, tag) in enumerate(zip(tokens, tags)):
            if tags_are_strings:
                label = str(tag)
            elif tag_names is not None:
                label = tag_names[tag] if isinstance(tag, int) else str(tag)
            else:
                label = str(tag)

            if convert_bioes:
                label = _bioes_to_bio(label)

            rows.append({
                "Sentence": sentence,
                "Target Index": idx,
                "Label": label,
                "Word Form": str(tok),
                "Lemma": str(tok),
                "Source Type": f"NER_{lang}_{split}",
            })

    if not rows:
        logging.warning(f"NER {lang}/{split}: no rows produced.")
        return

    prefix = DATASET_REGISTRY[(lang, "ner")]
    _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_ner_{split}.csv",
                      f"NER {lang}/{split}")


def build_ner_nerel(split: str):
    """Build Russian NER from NEREL GitHub repo (CoNLL-like .ann + .txt files)."""
    # NEREL uses BRAT standoff format: .txt files with text, .ann files with annotations
    repo_url = "https://github.com/nerel-ds/NEREL"
    raw_base = "https://raw.githubusercontent.com/nerel-ds/NEREL/master"

    # NEREL has train/dev/test directories
    split_dir_map = {"train": "train", "dev": "dev", "test": "test"}
    split_dir = split_dir_map.get(split)
    if not split_dir:
        return

    # NEREL data is under NEREL-v1.1/{train,dev,test}/
    api_url = f"https://api.github.com/repos/nerel-ds/NEREL/contents/NEREL-v1.1/{split_dir}"
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        files = resp.json()
    except Exception as e:
        logging.warning(f"NEREL: could not list files for {split}: {e}")
        return

    # Get pairs of .txt and .ann files
    txt_files = {f["name"].replace(".txt", ""): f["download_url"]
                 for f in files if f["name"].endswith(".txt")}
    ann_files = {f["name"].replace(".ann", ""): f["download_url"]
                 for f in files if f["name"].endswith(".ann")}

    common_docs = sorted(set(txt_files) & set(ann_files))
    if not common_docs:
        logging.warning(f"NEREL {split}: no matching .txt/.ann pairs found.")
        return

    rows = []
    cache_dir = DATA_DIR / "raw_nerel" / split_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    for doc_id in common_docs:
        txt_path = cache_dir / f"{doc_id}.txt"
        ann_path = cache_dir / f"{doc_id}.ann"

        try:
            http_get(txt_files[doc_id], txt_path, text=True)
            http_get(ann_files[doc_id], ann_path, text=True)
        except Exception:
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        # Parse BRAT annotations
        spans = []
        for line in ann_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            ann_info = parts[1]
            # Format: "EntityType start end" (may have discontinuous spans with ;)
            info_parts = ann_info.split()
            if len(info_parts) < 3:
                continue
            etype = info_parts[0]
            try:
                char_start = int(info_parts[1])
                # Handle discontinuous spans - take the last end
                char_end = int(info_parts[-1])
            except ValueError:
                continue
            spans.append((char_start, char_end, etype))

        # Tokenize by whitespace and map character offsets to token indices
        tokens = text.split()
        char_to_tok = {}
        pos = 0
        for i, tok in enumerate(tokens):
            start = text.index(tok, pos)
            for c in range(start, start + len(tok)):
                char_to_tok[c] = i
            pos = start + len(tok)

        # Build BIO labels
        labels = ["O"] * len(tokens)
        for char_start, char_end, etype in sorted(spans, key=lambda s: s[0]):
            tok_start = char_to_tok.get(char_start)
            tok_end = char_to_tok.get(char_end - 1)
            if tok_start is None or tok_end is None:
                continue
            for ti in range(tok_start, tok_end + 1):
                labels[ti] = f"B-{etype}" if ti == tok_start else f"I-{etype}"

        sentence = " ".join(tokens)
        for idx, (tok, label) in enumerate(zip(tokens, labels)):
            rows.append({
                "Sentence": sentence,
                "Target Index": idx,
                "Label": label,
                "Word Form": tok,
                "Lemma": tok,
                "Source Type": f"NEREL_NER_{split}",
            })

    if rows:
        prefix = DATASET_REGISTRY[("ru", "ner")]
        _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_ner_{split}.csv",
                          f"NER ru/NEREL {split}")
    else:
        logging.warning(f"NEREL {split}: no rows produced.")


# ============================================================
# SRL from Universal Propositions (CoNLL-U+ format)
# ============================================================

ARG_BIO_RE = re.compile(r"^(B|I)-(.+)$", re.IGNORECASE)


def _is_neutral(val: str) -> bool:
    v = (val or "").strip()
    return (not v) or v in {"O", "_", "-", "*", ""}


def _conllup_sentence_to_srl_rows(lines: list[str], columns_hint: list[str],
                                  doc_id: str, sent_id: str,
                                  source_type: str) -> list[dict]:
    """Parse a CoNLL-U+ sentence block into SRL rows. Shared with English SRL logic."""
    raw_tokens = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        cols = ln.split("\t")
        if len(cols) < 10:
            continue
        # Keep only integer-ID rows: multi-word-token ranges ("6-7 im") and empty
        # nodes ("8.1") would duplicate surface forms and shift every index.
        if not cols[0].isdigit():
            continue
        raw_tokens.append(cols)
    if not raw_tokens:
        return []

    max_cols = max(len(t) for t in raw_tokens)
    tokens = [(t + ["O"] * (max_cols - len(t))) if len(t) < max_cols else t for t in raw_tokens]
    forms = [t[1] for t in tokens]
    sentence_text = " ".join(forms).strip()

    extra_cols = list(range(10, max_cols))
    if not extra_cols:
        return []

    pred_cols = []
    for j in extra_cols:
        col_vals = [t[j].strip() for t in tokens]
        if all(_is_neutral(v) for v in col_vals):
            continue
        has_v = any(v in {"V", "B-V", "I-V"} or v.upper() in {"B-V", "I-V"} for v in col_vals)
        has_role = any((ARG_BIO_RE.match(v) or (not _is_neutral(v) and v.upper() != "V")) for v in col_vals)
        if has_v or has_role:
            pred_cols.append(j)
    if not pred_cols:
        return []

    out_rows = []
    for j in pred_cols:
        col_vals = [t[j].strip() for t in tokens]
        pred_idx = None
        for i, v in enumerate(col_vals):
            if v in {"V", "B-V", "I-V"} or v.upper() in {"B-V", "I-V"}:
                pred_idx = i
                break
        if pred_idx is None:
            for i, t in enumerate(tokens):
                upos = (t[3] if len(t) > 3 else "").upper()
                if upos in {"VERB", "AUX"}:
                    pred_idx = i
                    break
        if pred_idx is None:
            pred_idx = 0

        i, n = 0, len(tokens)
        while i < n:
            lab = col_vals[i]
            m = ARG_BIO_RE.match(lab)
            if m and m.group(1).upper() == "B":
                role = m.group(2)
                start = i
                i += 1
                while i < n:
                    m2 = ARG_BIO_RE.match(col_vals[i])
                    if not (m2 and m2.group(1).upper() == "I" and m2.group(2) == role):
                        break
                    i += 1
                out_rows.append({
                    "Sentence": sentence_text,
                    "Predicate Index": pred_idx,
                    "Arg Start": start, "Arg End": i,
                    "Label": role, "Source Type": source_type,
                    "Doc ID": doc_id, "Sent ID": sent_id,
                })
            elif (not _is_neutral(lab)) and lab.upper() != "V":
                out_rows.append({
                    "Sentence": sentence_text,
                    "Predicate Index": pred_idx,
                    "Arg Start": i, "Arg End": i + 1,
                    "Label": lab, "Source Type": source_type,
                    "Doc ID": doc_id, "Sent ID": sent_id,
                })
                i += 1
            else:
                i += 1
    return out_rows


def parse_conllup_srl(path: Path, source_type: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        sent_lines: list[str] = []
        columns_hint: list[str] = []
        doc_id = ""
        sent_id = ""
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                # Some UP releases omit the "=" in comment lines
                # (e.g. German: "# sent_id train-s1/de").
                def _comment_value(prefix):
                    rest = line[len(prefix):].strip()
                    return rest[1:].strip() if rest.startswith("=") else rest
                if line.startswith("# global.columns"):
                    columns_hint = _comment_value("# global.columns").split()
                if line.startswith("# newdoc id"):
                    doc_id = _comment_value("# newdoc id")
                if line.startswith("# sent_id"):
                    sent_id = _comment_value("# sent_id")
                sent_lines.append(line)
                continue
            if line == "":
                if any(l and not l.startswith("#") for l in sent_lines):
                    rows.extend(_conllup_sentence_to_srl_rows(
                        sent_lines, columns_hint, doc_id, sent_id, source_type))
                sent_lines = []
                continue
            sent_lines.append(line)
        if sent_lines and any(l and not l.startswith("#") for l in sent_lines):
            rows.extend(_conllup_sentence_to_srl_rows(
                sent_lines, columns_hint, doc_id, sent_id, source_type))
    return rows


def build_srl_up(lang: str, split: str):
    """Build SRL dataset from Universal Propositions (if available for this language)."""
    cfg = SRL_CONFIGS.get(lang)
    if not cfg:
        logging.info(f"SRL {lang}: no UP repo configured, skipping.")
        return

    repo = cfg["up_repo"]
    prefix = cfg["up_prefix"]
    base_url = f"https://raw.githubusercontent.com/UniversalPropositions/UP-1.0/master/{repo}"
    filename = f"{prefix}-{split}.conllu"
    url = f"{base_url}/{filename}"
    dest = DATA_DIR / f"raw_up_{lang}" / filename

    try:
        path = http_get(url, dest, text=True)
    except requests.HTTPError:
        logging.warning(f"SRL {lang}: could not download {url}")
        return

    rows = parse_conllup_srl(path, f"UP_{lang}_SRL")
    if not rows:
        logging.warning(f"SRL {lang}/{split}: no SRL rows found in {filename}")
        return

    prefix = DATASET_REGISTRY[(lang, "srl")]
    _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_srl_{split}.csv",
                      f"SRL {lang}/{split}")


# ============================================================
# Relation extraction from REDFM (HuggingFace)
# ============================================================

def _char_to_token_map(text: str) -> dict[int, int]:
    """Build a mapping from character offset to whitespace-token index."""
    tokens = text.split()
    mapping = {}
    pos = 0
    for i, tok in enumerate(tokens):
        idx = text.find(tok, pos)
        if idx == -1:
            idx = pos
        for c in range(idx, idx + len(tok)):
            mapping[c] = i
        pos = idx + len(tok)
    return mapping


def build_relations_redfm(lang: str, split: str):
    """Build relation extraction dataset from REDFM.

    Fetches the raw per-language jsonl straight from the hub (the repo is
    script-only, which `datasets`>=3 refuses to load). Each line has 'text' and
    'relations' whose subject/object dicts carry char-offset 'boundaries' and
    whose predicate carries a human-readable 'surfaceform'.
    """
    if lang not in REDFM_LANGS:
        return

    url = (f"https://huggingface.co/datasets/Babelscape/REDFM/resolve/main/"
           f"data/{split}.{REDFM_LANGS[lang]}.jsonl")
    dest = DATA_DIR / f"raw_redfm_{lang}" / f"{split}.jsonl"
    try:
        path = http_get(url, dest, text=True)
    except requests.HTTPError as e:
        logging.warning(f"Relations {lang}: could not fetch REDFM {split}: {e}")
        return

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        # normalize whitespace so char offsets map cleanly to split() tokens
        text = example.get("text", "")
        relations = example.get("relations", [])
        doc_id = example.get("docid", "")
        if not text or not relations:
            continue

        c2t = _char_to_token_map(text)

        for rel in relations:
            subj, obj, pred = rel.get("subject"), rel.get("object"), rel.get("predicate")
            if not (isinstance(subj, dict) and isinstance(obj, dict) and isinstance(pred, dict)):
                continue
            sb, ob = subj.get("boundaries"), obj.get("boundaries")
            label = pred.get("surfaceform") or pred.get("uri")
            if not sb or not ob or not label:
                continue

            s1_start = c2t.get(int(sb[0]), 0)
            s1_end = c2t.get(max(int(sb[1]) - 1, 0), 0) + 1
            s2_start = c2t.get(int(ob[0]), 0)
            s2_end = c2t.get(max(int(ob[1]) - 1, 0), 0) + 1

            rows.append({
                "Text": text, "Sentence": text,
                "Span1 Start": s1_start, "Span1 End": s1_end,
                "Span2 Start": s2_start, "Span2 End": s2_end,
                "Label": str(label),
                "Source Type": f"REDFM_{lang}",
                "Doc ID": doc_id, "Sent1 ID": 0, "Sent2 ID": 0,
            })

    if rows:
        prefix = DATASET_REGISTRY[(lang, "relation")]
        _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_relation_{split}.csv",
                          f"Relations {lang}/REDFM {split}")
    else:
        logging.warning(f"Relations {lang}/{split}: no rows produced from REDFM.")


# ============================================================
# Coreference from CorefUD (local zip)
# ============================================================

COREFUD_ENTITY_RE = re.compile(r"\(?(e\d+)")


def _extract_corefud_file(lang: str, split: str) -> Path | None:
    """Extract a CorefUD conllu file from the nested zip archive."""
    cfg = COREFUD_CONFIGS.get(lang)
    if not cfg:
        return None

    outer_zip = DATA_DIR / COREFUD_ZIP
    if not outer_zip.exists():
        logging.warning(f"CorefUD zip not found at {outer_zip}")
        return None

    treebank = cfg["treebank"]
    prefix = cfg["prefix"]
    filename = f"{prefix}-{split}.conllu"
    inner_path = f"CorefUD-1.4-public/data/{treebank}/{filename}"

    dest = DATA_DIR / "raw_corefud" / lang / filename
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer_zip) as outer:
        inner_data = outer.read(COREFUD_INNER_ZIP)
        with zipfile.ZipFile(io.BytesIO(inner_data)) as inner:
            if inner_path not in inner.namelist():
                logging.warning(f"CorefUD: {inner_path} not found in archive")
                return None
            dest.write_bytes(inner.read(inner_path))
    return dest


def _parse_corefud_entities(misc_field: str) -> list[tuple[str, bool, bool]]:
    """Parse Entity annotations from CorefUD MISC column.

    Returns list of (entity_id, is_opening, is_closing) tuples.
    Format examples: (e3509--1-...) for single-token, (e3509--2-... for open, e3509) for close.
    """
    if not misc_field or misc_field == "_":
        return []

    results = []
    # Split on | to get individual MISC fields
    for field in misc_field.split("|"):
        if not field.startswith("Entity="):
            continue
        val = field[len("Entity="):]
        # Parse entity mentions - can have multiple like (e3510--2-...(e3514--1-...)
        # Opening: (eXXXX  Closing: eXXXX)  Single: (eXXXX--1-...)
        pos = 0
        while pos < len(val):
            if val[pos] == "(":
                # Opening bracket - find entity ID
                m = re.match(r"\(e(\d+)", val[pos:])
                if m:
                    eid = "e" + m.group(1)
                    # Check if it closes on same token (contains matching close)
                    # Look for the span length indicator: --N- where N=1 means single token
                    span_m = re.search(r"--(\d+)-", val[pos:])
                    if span_m and span_m.group(1) == "1":
                        results.append((eid, True, True))  # single-token mention
                    else:
                        results.append((eid, True, False))  # opening
                    pos += m.end()
                else:
                    pos += 1
            elif val[pos] == "e":
                # Could be a closing bracket: eXXXX)
                m = re.match(r"e(\d+)\)", val[pos:])
                if m:
                    eid = "e" + m.group(1)
                    results.append((eid, False, True))  # closing
                    pos += m.end()
                else:
                    pos += 1
            else:
                pos += 1
    return results


def build_coref_corefud(lang: str, split: str):
    """Build coreference mention-pair dataset from CorefUD."""
    path = _extract_corefud_file(lang, split)
    if not path:
        return

    rng = random.Random(RNG_SEED)
    all_rows = []

    # Parse documents
    current_doc_id = None
    doc_sentences = []
    doc_tokens = []
    doc_mentions = []  # list of (entity_id, start_tok_idx, end_tok_idx)
    open_mentions = {}  # entity_id -> start_tok_idx
    global_tok_idx = 0

    def _flush_doc():
        nonlocal doc_sentences, doc_tokens, doc_mentions, open_mentions, global_tok_idx
        if not doc_mentions:
            doc_sentences, doc_tokens, doc_mentions = [], [], []
            open_mentions = {}
            global_tok_idx = 0
            return

        # Group mentions by entity ID (coreference chains)
        chains: dict[str, list[int]] = {}
        for i, (eid, start, end) in enumerate(doc_mentions):
            chains.setdefault(eid, []).append(i)

        # Generate positive pairs (coreferent mentions)
        pos_pairs = []
        for eid, idxs in chains.items():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pos_pairs.append((idxs[a], idxs[b]))

        if not pos_pairs:
            doc_sentences, doc_tokens, doc_mentions = [], [], []
            open_mentions = {}
            global_tok_idx = 0
            return

        # Generate negative pairs
        all_indices = list(range(len(doc_mentions)))
        neg_pairs = set()
        target_negs = len(pos_pairs)
        trials = 0
        while len(neg_pairs) < target_negs and trials < target_negs * 20:
            i, j = rng.sample(all_indices, 2)
            if doc_mentions[i][0] != doc_mentions[j][0]:
                neg_pairs.add((min(i, j), max(i, j)))
            trials += 1

        doc_text = " ".join(doc_tokens)
        for i, j in pos_pairs:
            _, s1, e1 = doc_mentions[i]
            _, s2, e2 = doc_mentions[j]
            all_rows.append({
                "Text": doc_text, "Sentence": doc_text,
                "Span1 Start": s1, "Span1 End": e1,
                "Span2 Start": s2, "Span2 End": e2,
                "Label": 1,
                "Source Type": f"CorefUD_{lang}_{split}",
                "Doc ID": current_doc_id or "",
                "Sent1 ID": 0, "Sent2 ID": 0,
            })
        for i, j in neg_pairs:
            _, s1, e1 = doc_mentions[i]
            _, s2, e2 = doc_mentions[j]
            all_rows.append({
                "Text": doc_text, "Sentence": doc_text,
                "Span1 Start": s1, "Span1 End": e1,
                "Span2 Start": s2, "Span2 End": e2,
                "Label": 0,
                "Source Type": f"CorefUD_{lang}_{split}",
                "Doc ID": current_doc_id or "",
                "Sent1 ID": 0, "Sent2 ID": 0,
            })

        doc_sentences, doc_tokens, doc_mentions = [], [], []
        open_mentions = {}
        global_tok_idx = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# newdoc"):
                _flush_doc()
                parts = line.split("=", 1)
                current_doc_id = parts[1].strip() if len(parts) > 1 else ""
                continue
            if line.startswith("#") or not line.strip():
                continue

            cols = line.split("\t")
            if len(cols) < 10:
                continue

            tok_id = cols[0]
            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "4.1")
            if "-" in tok_id or "." in tok_id:
                # But still check for Entity annotations on empty nodes
                if "." in tok_id:
                    misc = cols[9] if len(cols) > 9 else "_"
                    for eid, is_open, is_close in _parse_corefud_entities(misc):
                        if is_open and is_close:
                            # Single-token mention pointing to the head token
                            # Use the integer part as approximate position
                            approx_idx = global_tok_idx - 1 if global_tok_idx > 0 else 0
                            doc_mentions.append((eid, approx_idx, approx_idx + 1))
                        elif is_open:
                            open_mentions[eid] = global_tok_idx - 1 if global_tok_idx > 0 else 0
                        elif is_close and eid in open_mentions:
                            start = open_mentions.pop(eid)
                            doc_mentions.append((eid, start, global_tok_idx))
                continue

            form = cols[1]
            doc_tokens.append(form)
            misc = cols[9] if len(cols) > 9 else "_"

            for eid, is_open, is_close in _parse_corefud_entities(misc):
                if is_open and is_close:
                    doc_mentions.append((eid, global_tok_idx, global_tok_idx + 1))
                elif is_open:
                    open_mentions[eid] = global_tok_idx
                elif is_close and eid in open_mentions:
                    start = open_mentions.pop(eid)
                    doc_mentions.append((eid, start, global_tok_idx + 1))

            global_tok_idx += 1

    _flush_doc()

    if all_rows:
        prefix = DATASET_REGISTRY[(lang, "coref")]
        _write_csv_capped(pd.DataFrame(all_rows), DATA_DIR / f"{prefix}_coref_{split}.csv",
                          f"Coref {lang}/{split}")
    else:
        logging.warning(f"Coref {lang}/{split}: no rows produced.")


def build_coref_rucoco(split: str):
    """Build Russian coreference from RuCoCo (local zip).

    RuCoCo has no official train/dev/test split, so we deterministically split
    the documents: 70% train, 10% dev, 20% test.
    """
    rucoco_zip = DATA_DIR / RUCOCO_ZIP
    if not rucoco_zip.exists():
        logging.warning(f"RuCoCo zip not found at {rucoco_zip}")
        return

    with zipfile.ZipFile(rucoco_zip) as z:
        json_files = sorted(n for n in z.namelist() if n.endswith(".json"))

    # Deterministic split
    rng = random.Random(RNG_SEED)
    shuffled = list(json_files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.7)
    dev_end = train_end + int(n * 0.1)

    split_files = {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }

    docs = split_files.get(split, [])
    if not docs:
        return

    all_rows = []
    rng_pairs = random.Random(RNG_SEED + hash(split))

    with zipfile.ZipFile(rucoco_zip) as z:
        for doc_name in docs:
            data = json.loads(z.read(doc_name).decode("utf-8"))
            text = data.get("text", "")
            chains = data.get("entities", [])
            if not text or not chains:
                continue

            # Tokenize by whitespace
            tokens = text.split()
            # Build char->token mapping
            char_to_tok = {}
            pos = 0
            for i, tok in enumerate(tokens):
                idx = text.find(tok, pos)
                for c in range(idx, idx + len(tok)):
                    char_to_tok[c] = i
                pos = idx + len(tok)

            # Convert char spans to token spans
            mentions = []  # (chain_id, tok_start, tok_end)
            for chain_id, chain in enumerate(chains):
                for span in chain:
                    char_start, char_end = span
                    tok_start = char_to_tok.get(char_start)
                    tok_end = char_to_tok.get(char_end - 1)
                    if tok_start is not None and tok_end is not None:
                        mentions.append((chain_id, tok_start, tok_end + 1))

            if len(mentions) < 2:
                continue

            # Positive pairs
            chain_groups: dict[int, list[int]] = {}
            for i, (cid, _, _) in enumerate(mentions):
                chain_groups.setdefault(cid, []).append(i)

            pos_pairs = []
            for cid, idxs in chain_groups.items():
                if len(idxs) < 2:
                    continue
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        pos_pairs.append((idxs[a], idxs[b]))

            if not pos_pairs:
                continue

            # Negative pairs
            all_indices = list(range(len(mentions)))
            neg_pairs = set()
            target_negs = len(pos_pairs)
            trials = 0
            while len(neg_pairs) < target_negs and trials < target_negs * 20:
                i, j = rng_pairs.sample(all_indices, 2)
                if mentions[i][0] != mentions[j][0]:
                    neg_pairs.add((min(i, j), max(i, j)))
                trials += 1

            sentence = " ".join(tokens)
            doc_id = doc_name.replace(".json", "")
            for i, j in pos_pairs:
                _, s1, e1 = mentions[i]
                _, s2, e2 = mentions[j]
                all_rows.append({
                    "Text": sentence, "Sentence": sentence,
                    "Span1 Start": s1, "Span1 End": e1,
                    "Span2 Start": s2, "Span2 End": e2,
                    "Label": 1,
                    "Source Type": f"RuCoCo_{split}",
                    "Doc ID": doc_id, "Sent1 ID": 0, "Sent2 ID": 0,
                })
            for i, j in neg_pairs:
                _, s1, e1 = mentions[i]
                _, s2, e2 = mentions[j]
                all_rows.append({
                    "Text": sentence, "Sentence": sentence,
                    "Span1 Start": s1, "Span1 End": e1,
                    "Span2 Start": s2, "Span2 End": e2,
                    "Label": 0,
                    "Source Type": f"RuCoCo_{split}",
                    "Doc ID": doc_id, "Sent1 ID": 0, "Sent2 ID": 0,
                })

    if all_rows:
        prefix = DATASET_REGISTRY[("ru", "coref")]
        _write_csv_capped(pd.DataFrame(all_rows), DATA_DIR / f"{prefix}_coref_{split}.csv",
                          f"Coref ru/RuCoCo {split}")
    else:
        logging.warning(f"RuCoCo {split}: no rows produced.")


# ============================================================
# DuIE 2.0 Chinese relation extraction (local zip)
# ============================================================

def _duie_records_to_rows(records: list, split: str) -> list[dict]:
    """Convert DuIE records ({'text', 'spo_list'}) to per-character span rows.
    Whitespace is stripped from the text first so that character index ==
    whitespace-token index of the space-joined character sentence."""
    rows = []
    for rec in records:
        text = re.sub(r"\s+", "", rec.get("text", "") or "")
        spo_list = rec.get("spo_list", [])
        if not text or spo_list is None or len(spo_list) == 0:
            continue
        for spo in spo_list:
            subject = re.sub(r"\s+", "", spo.get("subject", "") or "")
            predicate = spo.get("predicate", "")
            obj_val = spo.get("object", {})
            obj = obj_val.get("@value", "") if isinstance(obj_val, dict) else str(obj_val)
            obj = re.sub(r"\s+", "", obj or "")
            if not subject or not predicate or not obj:
                continue
            subj_start = text.find(subject)
            obj_start = text.find(obj)
            if subj_start == -1 or obj_start == -1:
                continue
            sentence = " ".join(list(text))
            rows.append({
                "Text": sentence, "Sentence": sentence,
                "Span1 Start": subj_start, "Span1 End": subj_start + len(subject),
                "Span2 Start": obj_start, "Span2 End": obj_start + len(obj),
                "Label": predicate, "Source Type": "DuIE_zh",
                "Doc ID": "", "Sent1 ID": 0, "Sent2 ID": 0,
            })
    return rows


def _build_duie_from_hf_mirror(split: str):
    """DuIE 2.0 via the xusenlin/duie parquet mirror (no zip needed)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    split_file = {"train": "data/train-00000-of-00001-84ec7a0a2e99d99c.parquet",
                  "dev": "data/validation-00000-of-00001-c745e3595863248c.parquet"}.get(split)
    if not split_file:
        return
    path = hf_hub_download(repo_id="xusenlin/duie", filename=split_file, repo_type="dataset")
    records = pq.read_table(path).to_pylist()
    rows = _duie_records_to_rows(records, split)
    if rows:
        prefix = DATASET_REGISTRY[("zh", "relation")]
        _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_relation_{split}.csv",
                          f"Relations zh/DuIE {split}")
    else:
        logging.warning(f"DuIE {split}: no rows produced from HF mirror.")


def build_relations_duie(split: str):
    """Build Chinese relation extraction dataset from DuIE 2.0."""
    duie_zip = DATA_DIR / DUIE_ZIP
    if not duie_zip.exists():
        logging.info(f"DuIE zip not found at {duie_zip}; using HF parquet mirror.")
        return _build_duie_from_hf_mirror(split)

    split_file_map = {"train": "DUIE/duie_train.json", "dev": "DUIE/duie_dev.json",
                      "test": "DUIE/duie_test2.json"}
    jsonl_path = split_file_map.get(split)
    if not jsonl_path:
        return

    with zipfile.ZipFile(duie_zip) as z:
        if jsonl_path not in z.namelist():
            logging.warning(f"DuIE: {jsonl_path} not found in archive")
            return
        raw = z.read(jsonl_path).decode("utf-8")

    records = []
    for line in raw.strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    rows = _duie_records_to_rows(records, split)

    if rows:
        prefix = DATASET_REGISTRY[("zh", "relation")]
        _write_csv_capped(pd.DataFrame(rows), DATA_DIR / f"{prefix}_relation_{split}.csv",
                          f"Relations zh/DuIE {split}")
    else:
        logging.warning(f"DuIE {split}: no rows produced.")


# ============================================================
# Main entry point
# ============================================================

# Maps (lang, task) -> (dataset_prefix, csv_basename_prefix)
# dataset_prefix: used in SLURM DATASET_MAP for output directory naming
# csv_basename_prefix: prefix of the CSV files in data/
DATASET_REGISTRY = {
    # UD-based (POS, DEP, Constituents use the same UD treebank)
    ("zh", "pos"): "ud_chinese_gsdsimp",
    ("zh", "dep"): "ud_chinese_gsdsimp",
    ("zh", "constituents"): "ud_chinese_gsdsimp",
    ("tr", "pos"): "ud_turkish_boun",
    ("tr", "dep"): "ud_turkish_boun",
    ("tr", "constituents"): "ud_turkish_boun",
    ("fr", "pos"): "ud_french_gsd",
    ("fr", "dep"): "ud_french_gsd",
    ("fr", "constituents"): "ud_french_gsd",
    ("ru", "pos"): "ud_russian_syntagrus",
    ("ru", "dep"): "ud_russian_syntagrus",
    ("ru", "constituents"): "ud_russian_syntagrus",
    ("de", "pos"): "ud_german_hdt",
    ("de", "dep"): "ud_german_hdt",
    ("de", "constituents"): "ud_german_hdt",
    # NER
    ("zh", "ner"): "msra_ner",
    ("tr", "ner"): "turkish_wikiner",
    ("fr", "ner"): "wikiner_fr_gold",
    ("ru", "ner"): "nerel",
    ("de", "ner"): "germeval_14",
    # SRL (Universal Propositions)
    ("zh", "srl"): "up_chinese",
    ("fr", "srl"): "up_french",
    ("de", "srl"): "up_german",
    # Coreference (CorefUD + RuCoCo)
    ("tr", "coref"): "corefud_turkish_itcc",
    ("fr", "coref"): "corefud_french_democrat",
    ("ru", "coref"): "rucoco",
    ("de", "coref"): "corefud_german_potsdamcc",
    # Relation extraction (REDFM + DuIE)
    ("zh", "relation"): "duie",
    ("fr", "relation"): "redfm_fr",
    ("de", "relation"): "redfm_de",
}


def get_csv_path(lang: str, task: str, split: str) -> Path:
    """Get the expected CSV output path for a language/task/split."""
    key = (lang, task)
    if key not in DATASET_REGISTRY:
        return None
    prefix = DATASET_REGISTRY[key]
    return DATA_DIR / f"{prefix}_{task}_{split}.csv"


def get_dataset_name(lang: str, task: str) -> str | None:
    """Get the dataset name for use in SLURM scripts."""
    return DATASET_REGISTRY.get((lang, task))


def print_availability_table():
    """Print a table of which language-task pairs are available."""
    header = f"{'Task':<20}" + "".join(f"{LANG_NAMES[l]:>12}" for l in LANGUAGES)
    print(header)
    print("-" * len(header))
    for task in TASKS:
        row = f"{task:<20}"
        for lang in LANGUAGES:
            name = DATASET_REGISTRY.get((lang, task))
            if name:
                row += f"{name:>24}"
            else:
                row += f"{'---':>24}"
        print(row)


def main():
    global MAX_ROWS, RNG_SEED
    ap = argparse.ArgumentParser(
        description="Create multilingual edge-probing datasets.")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    ap.add_argument("--langs", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    ap.add_argument("--splits", nargs="+", choices=["train", "dev", "test"],
                    default=["train", "dev", "test"])
    ap.add_argument("--max_rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true", help="Print availability table and exit.")
    args = ap.parse_args()

    RNG_SEED = args.seed
    MAX_ROWS = None if (args.max_rows is None or args.max_rows <= 0) else int(args.max_rows)

    if args.list:
        print_availability_table()
        return

    for lang in args.langs:
        logging.info(f"=== Processing {LANG_NAMES[lang]} ({lang}) ===")

        for split in args.splits:
            # UD-based tasks
            if any(t in args.tasks for t in ("pos", "dep", "constituents")):
                try:
                    build_ud_tasks(lang, split)
                except Exception as e:
                    logging.error(f"UD tasks {lang}/{split} failed: {e}")

            # NER
            if "ner" in args.tasks:
                if lang == "ru":
                    try:
                        build_ner_nerel(split)
                    except Exception as e:
                        logging.error(f"NEREL NER ru/{split} failed: {e}")
                else:
                    try:
                        build_ner_hf(lang, split)
                    except Exception as e:
                        logging.error(f"NER {lang}/{split} failed: {e}")

            # SRL (Universal Propositions)
            if "srl" in args.tasks and lang in SRL_CONFIGS:
                try:
                    build_srl_up(lang, split)
                except Exception as e:
                    logging.error(f"SRL {lang}/{split} failed: {e}")

            # Coreference (CorefUD for tr/fr/de, RuCoCo for ru)
            if "coref" in args.tasks:
                if lang in COREFUD_CONFIGS:
                    try:
                        build_coref_corefud(lang, split)
                    except Exception as e:
                        logging.error(f"Coref {lang}/{split} failed: {e}")
                if lang == "ru":
                    try:
                        build_coref_rucoco(split)
                    except Exception as e:
                        logging.error(f"Coref ru/{split} RuCoCo failed: {e}")

            # Relations (REDFM for fr/de, DuIE for zh)
            if "relation" in args.tasks:
                if lang in REDFM_LANGS:
                    try:
                        build_relations_redfm(lang, split)
                    except Exception as e:
                        logging.error(f"Relations {lang}/{split} REDFM failed: {e}")
                if lang == "zh":
                    try:
                        build_relations_duie(split)
                    except Exception as e:
                        logging.error(f"Relations zh/{split} DuIE failed: {e}")

    logging.info("All requested multilingual datasets have been generated.")


if __name__ == "__main__":
    main()
