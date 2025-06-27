import argparse
import csv
from pathlib import Path
import requests
from conllu import parse_incr
import re

# Map UD repo names to file prefixes and output CSV names
TREEBANK_MAP = {
    "UD_Chinese-GSD": ("zh_gsd", "ud_zh_gsd_dataset.csv"),
    "UD_German-GSD": ("de_gsd", "ud_de_gsd_dataset.csv"),
    "UD_French-GSD": ("fr_gsd", "ud_fr_gsd_dataset.csv"),
    "UD_Russian-SynTagRus": ("ru_syntagrus", "ud_ru_syntagrus_dataset.csv"),
    "UD_Turkish-IMST": ("tr_imst", "ud_tr_imst_dataset.csv"),
    "UD_English-GUM": ("en_gum", "ud_gum_dataset.csv"),
}
BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies"

# data directory one level up from this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

allowed_pattern = re.compile(r"^[A-Za-z']+$")

def get_category_and_label_ud(token):
    upos = token.get("upostag")
    feats = token.get("feats") or {}
    form = token["form"]
    category, label, dimension = None, None, None
    if upos == "VERB":
        category = "Verb"
        dimension = "Tense/Aspect"
        tense = feats.get("Tense")
        person = feats.get("Person")
        number = feats.get("Number")
        verbform = feats.get("VerbForm")
        if tense == "Pres":
            if person == "3" and number == "Sing":
                label = "3rd_pers"
            else:
                label = "base"
        elif tense == "Past":
            label = "past"
        elif verbform == "Part":
            if form.lower().endswith("ing"):
                label = "present_participle"
            else:
                label = "past_participle"
        else:
            label = "base"
    elif upos == "NOUN":
        category = "Noun"
        dimension = "Number"
        num = feats.get("Number")
        if num == "Plur":
            label = "plural"
        else:
            label = "singular"
    elif upos == "ADJ":
        category = "Adjective"
        dimension = "Degree"
        degree = feats.get("Degree")
        if degree == "Cmp":
            label = "comparative"
        elif degree == "Sup":
            label = "superlative"
        else:
            label = "positive"
    return category, label, dimension

def download_file(url: str, dest: Path) -> bool:
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print("Error downloading {}: {}".format(url, e))
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)
    print("Saved {}".format(dest.relative_to(SCRIPT_DIR.parent)))
    return True

def process_conllu_to_csv(conllu_path: Path, out_csv: Path, treebank: str):
    # Only process the train split
    dataset_rows = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sentence_tokens = [token["form"] for token in tokenlist]
            sentence_text = " ".join(sentence_tokens)
            for idx, token in enumerate(tokenlist):
                if not isinstance(token["id"], int):
                    continue
                word_form = token["form"]
                lemma = token.get("lemma")
                # For English, use the same filtering as in the notebook
                if treebank == "UD_English-GUM":
                    if not allowed_pattern.fullmatch(word_form):
                        continue
                    if lemma is None or not allowed_pattern.fullmatch(lemma):
                        continue
                    category, inflection_label, dimension = get_category_and_label_ud(token)
                    if category is None or inflection_label is None:
                        continue
                else:
                    # For other languages, keep all tokens, but fill columns as best as possible
                    category, inflection_label, dimension = None, None, None
                    upos = token.get("upostag")
                    feats = token.get("feats") or {}
                    if upos == "VERB":
                        category = "Verb"
                        dimension = "Tense/Aspect"
                    elif upos == "NOUN":
                        category = "Noun"
                        dimension = "Number"
                    elif upos == "ADJ":
                        category = "Adjective"
                        dimension = "Degree"
                    # Compose inflection label as in original script
                    if feats:
                        parts = []
                        for feat, val in feats.items():
                            if isinstance(val, list):
                                for v in val:
                                    parts.append("{}={}".format(feat, v))
                            else:
                                parts.append("{}={}".format(feat, val))
                        inflection_label = "|".join(parts)
                    else:
                        inflection_label = "_"
                dataset_rows.append({
                    "Sentence": sentence_text,
                    "Target Index": idx,
                    "Lemma": lemma or "_",
                    "Category": category or "_",
                    "Inflection Label": inflection_label or "_",
                    "Word Form": word_form or "_",
                    "Dimension": dimension or "_",
                    "Source Type": treebank
                })
    with open(out_csv, "w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=[
            "Sentence", "Target Index", "Lemma", "Category", "Inflection Label", "Word Form", "Dimension", "Source Type"
        ])
        writer.writeheader()
        for row in dataset_rows:
            writer.writerow(row)
    print("Wrote {}".format(out_csv.relative_to(SCRIPT_DIR.parent)))

def main():
    parser = argparse.ArgumentParser(
        description="Download UD .conllu and convert to CSV in ../data/"
    )
    parser.add_argument(
        "--treebanks", nargs="+", required=True,
        help="UD treebanks to download (keys in TREEBANK_MAP)"
    )
    args = parser.parse_args()

    print("Downloading into {}".format(DATA_DIR))
    for tb in args.treebanks:
        if tb not in TREEBANK_MAP:
            print("Unknown treebank {}, skipping.".format(tb))
            continue
        prefix, out_csv_name = TREEBANK_MAP[tb]
        # Only process the train split
        split = "train"
        url = "{}/{}/master/{}-ud-{}.conllu".format(BASE_URL, tb, prefix, split)
        dest = DATA_DIR / "{}-{}.conllu".format(tb, split)
        print("{} {}: ".format(tb, split), end="")
        if not dest.exists():
            download_file(url, dest)
        else:
            print("Already exists.")
        # Convert to CSV in the same format as dataset.ipynb
        out_csv = DATA_DIR / out_csv_name
        print("Processing {} to {}".format(dest.name, out_csv.name))
        process_conllu_to_csv(dest, out_csv, tb)

    print()
    print("Done. CSV files are in the data folder above.")

if __name__ == "__main__":
    main()