#!/usr/bin/env python3
import os
import requests
import pandas as pd
from conllu import parse_incr

DATASETS = {
    # "UD_English-GUM": "en_gum",
    "UD_Chinese-GSD": "zh_gsd",
    "UD_German-GSD": "de_gsd",
    "UD_French-GSD": "fr_gsd",
    "UD_Russian-SynTagRus": "ru_syntagrus",
    "UD_Turkish-IMST": "tr_imst",
}

def extract_inflection(feats: dict):
    if not feats:
        return None, None

    items = sorted(feats.items())
    label = "|".join(f"{k}={v}" for k, v in items)
    dimension = "|".join(k for k, _ in items)

    return label, dimension

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data2"))
    os.makedirs(data_dir, exist_ok=True)

    for ud_name, slug in DATASETS.items():
        url = (
            f"https://raw.githubusercontent.com/UniversalDependencies/"
            f"{ud_name}/master/{slug}-ud-train.conllu"
        )
        conllu_path = os.path.join(data_dir, f"{slug}-ud-train.conllu")
        csv_path = os.path.join(data_dir, f"ud_{slug}_dataset.csv")

        if not os.path.exists(conllu_path):
            print(f"Downloading {ud_name} training data...")
            resp = requests.get(url)
            resp.raise_for_status()
            with open(conllu_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
        else:
            print(f"Using cached {conllu_path}")

        rows = []

        with open(conllu_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                sentence = " ".join(t["form"] for t in tokenlist)

                for idx, token in enumerate(tokenlist):
                    if not isinstance(token["id"], int):
                        continue

                    feats = token.get("feats") or {}
                    label, dimension = extract_inflection(feats)
                    if label is None:
                        continue

                    lemma = token.get("lemma")
                    form = token.get("form")
                    if lemma is None:
                        continue

                    upos = token.get("upostag")
                    if upos in {"VERB", "NOUN", "ADJ"}:
                        category = {"VERB": "Verb", "NOUN": "Noun", "ADJ": "Adjective"}[upos]
                    else:
                        continue

                    rows.append({
                        "Sentence": sentence,
                        "Target Index": idx,
                        "Lemma": lemma,
                        "Category": category,
                        "Inflection Label": label,
                        "Word Form": form,
                        "Dimension": dimension,
                        "Source Type": ud_name
                    })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")

if __name__ == "__main__":
    main()
