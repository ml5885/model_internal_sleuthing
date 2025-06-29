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

def get_category_and_label_ud(token):
    upos = token.get("upostag")
    feats = token.get("feats") or {}
    form = token["form"]
    category = None
    label = None
    dimension = None

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
        if feats.get("Number") == "Plur":
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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(
        os.path.join(script_dir, "..", "data")
    )
    os.makedirs(data_dir, exist_ok=True)

    for ud_name, slug in DATASETS.items():
        train_file = f"{slug}-ud-train.conllu"
        if ud_name == "UD_Russian-SynTagRus":
            train_file = f"{slug}-ud-train-a.conllu"

        url = (
            f"https://raw.githubusercontent.com/"
            f"UniversalDependencies/{ud_name}/"
            f"master/{train_file}"
        )
        conllu_path = os.path.join(data_dir, train_file)
        csv_path = os.path.join(
            data_dir, f"ud_{slug}_dataset.csv"
        )

        if not os.path.exists(conllu_path):
            print(f"Downloading {ud_name} training data...")
            resp = requests.get(url)
            resp.raise_for_status()
            with open(conllu_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

        rows = []
        with open(conllu_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                sentence = " ".join(
                    t["form"] for t in tokenlist
                )
                for idx, token in enumerate(tokenlist):
                    if not isinstance(token["id"], int):
                        continue

                    category, label, dimension = (
                        get_category_and_label_ud(token)
                    )
                    if category is None or label is None:
                        continue

                    lemma = token.get("lemma")
                    if lemma is None:
                        continue

                    form = token["form"]

                    rows.append({
                        "Sentence": sentence,
                        "Target Index": idx,
                        "Lemma": lemma,
                        "Category": category,
                        "Inflection Label": label,
                        "Word Form": form,
                        "Dimension": dimension,
                        "Source Type": ud_name,
                    })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(rows)} rows to {csv_path}")

if __name__ == "__main__":
    main()
