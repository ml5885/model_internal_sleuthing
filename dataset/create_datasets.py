import argparse
import csv
from pathlib import Path
import requests
from conllu import parse_incr

# Map UD repo names to file prefixes
TREEBANK_MAP = {
    # "UD_English-GUM": "en_gum",
    "UD_Chinese-GSD": "zh_gsd",
    "UD_German-GSD": "de_gsd",
    "UD_French-GSD": "fr_gsd",
    "UD_Russian-SynTagRus": "ru_syntagrus",
    "UD_Turkish-IMST": "tr_imst",
}
DEFAULT_SPLITS = ["train","dev","test"]
BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies"

# data directory one level up from this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

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

def conllu_to_csv(conllu_path: Path):
    csv_path = conllu_path.with_suffix(".csv")
    with open(conllu_path, "r", encoding="utf-8") as rf, \
         open(csv_path, "w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Sentence","Target Index","Lemma","Inflection Label","Word Form"] )
        for tokenlist in parse_incr(rf):
            words = [t["form"] for t in tokenlist]
            sent = " ".join(words)
            for idx, token in enumerate(tokenlist):
                lemma = token.get("lemma") or "_"
                feats = token.get("feats") or {}
                if feats:
                    parts = []
                    for feat, val in feats.items():
                        if isinstance(val, list):
                            for v in val:
                                parts.append("{}={}".format(feat, v))
                        else:
                            parts.append("{}={}".format(feat, val))
                    inf_label = "|".join(parts)
                else:
                    inf_label = "_"
                writer.writerow([sent, idx, lemma, inf_label, token.get("form") or "_"])
    print("Wrote {}".format(csv_path.relative_to(SCRIPT_DIR.parent)))

def main():
    parser = argparse.ArgumentParser(
        description="Download UD .conllu and convert to CSV in ../data/"
    )
    parser.add_argument(
        "--treebanks", nargs="+", required=True,
        help="UD treebanks to download (keys in TREEBANK_MAP)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=DEFAULT_SPLITS,
        help="Which splits to fetch: train, dev, test"
    )
    args = parser.parse_args()

    print("Downloading into {}".format(DATA_DIR))
    for tb in args.treebanks:
        if tb not in TREEBANK_MAP:
            print("Unknown treebank {}, skipping.".format(tb))
            continue
        prefix = TREEBANK_MAP[tb]
        splits = []
        for sp in args.splits:
            if tb == "UD_Russian-SynTagRus" and sp == "train":
                splits.extend(["train-a","train-b","train-c"] )
            else:
                splits.append(sp)
        for split in splits:
            url = "{}/{}/master/{}-ud-{}.conllu".format(BASE_URL, tb, prefix, split)
            dest = DATA_DIR / "{}-{}.conllu".format(tb, split)
            print("{} {}: ".format(tb, split), end="")
            download_file(url, dest)

    print()
    print("Converting .conllu to .csv in {}".format(DATA_DIR))
    for conllu in sorted(DATA_DIR.glob("*.conllu")):
        print(conllu.name)
        conllu_to_csv(conllu)

    print()
    print("Done. CSV files are in the data folder above.")

if __name__ == "__main__":
    main()
