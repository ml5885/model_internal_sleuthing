import argparse
import os
import numpy as np
import pandas as pd
import torch
import sys
import math
from tqdm import tqdm
from src.model_wrapper import ModelWrapper
from src import config, utils


def extract_and_save(data_path, output_dir, model_key, revision=None, max_rows=0, use_attention=False): 
    """
    Extract hidden-state activations for each example in an edge-probing dataset and save them in
    compressed .npz shards.

    The input CSV must contain at least a `Sentence` column. Depending on the task, it may also
    contain one of the following to indicate target spans:

    * `Target Index`: single integer index of the target word (for single-span tasks like POS/NER).
    * `Span1 Start`, `Span1 End`, `Span2 Start`, `Span2 End`: start/end indices (exclusive) of two spans.
    * `Predicate Index`, `Arg Start`, `Arg End`: predicate index and argument span for SRL/SPR tasks.

    The function automatically infers which column set is present and constructs the appropriate
    list of target positions for each example. Multi-span examples produce concatenated span
    representations.
    """
    batch_size = config.MODEL_CONFIGS[model_key]["batch_size"]

    # Load the entire CSV to inspect columns. We require at least the Sentence column.
    df = pd.read_csv(data_path)
    if "Sentence" not in df.columns:
        raise ValueError(f"Input data at {data_path} must contain a 'Sentence' column.")

    # Limit number of rows if requested
    if max_rows > 0 and len(df) > max_rows:
        utils.log_info(f"Sampling {max_rows} rows from {len(df)} total rows.")
        df = df.sample(n=max_rows, random_state=config.SEED).reset_index(drop=True)

    num_rows = len(df)
    total = math.ceil(num_rows / batch_size)

    os.makedirs(output_dir, exist_ok=True)
    model_wrapper = ModelWrapper(model_key, revision=revision) 
    shard_paths = []

    # Save the sampled indices to map activations back to the original dataset
    df.reset_index().to_csv(os.path.join(output_dir, "sampled_indices.csv"), index_label="index")

    # Determine which span specification is present
    has_span_pairs = all(c in df.columns for c in ["Span1 Start", "Span1 End", "Span2 Start", "Span2 End"])
    has_predicate_arg = all(c in df.columns for c in ["Predicate Index", "Arg Start", "Arg End"])
    has_single_index = "Target Index" in df.columns

    if not (has_span_pairs or has_predicate_arg or has_single_index):
        raise ValueError(
            "Input CSV must contain either 'Target Index', or the pair of span columns, "
            "or predicate/argument columns."
        )

    # Build target indices list for all examples upfront to support sharding
    target_list = []
    for _, row in df.iterrows():
        if has_span_pairs:
            # Two separate spans
            start1, end1 = int(row["Span1 Start"]), int(row["Span1 End"])
            start2, end2 = int(row["Span2 Start"]), int(row["Span2 End"])
            positions1 = list(range(start1, end1))
            positions2 = list(range(start2, end2))
            target_list.append([positions1, positions2])
        elif has_predicate_arg:
            # Predicate index and argument span
            pred_idx = int(row["Predicate Index"])
            arg_start, arg_end = int(row["Arg Start"]), int(row["Arg End"])
            arg_positions = list(range(arg_start, arg_end))
            target_list.append([[pred_idx], arg_positions])
        else:
            # Single target index
            target_list.append(int(row["Target Index"]))

    # Process in batches to avoid OOM
    for part_idx, i in enumerate(tqdm(range(0, num_rows, batch_size),
                                      desc="Extracting Batches", total=total,
                                      dynamic_ncols=True, leave=True, file=sys.stdout)):
        chunk = df.iloc[i:i+batch_size]
        sentences = chunk["Sentence"].tolist()
        # Corresponding target indices for this chunk
        targets_chunk = target_list[i:i+batch_size]

        with torch.no_grad():
            activations = model_wrapper.extract_activations(sentences, targets_chunk, use_attention=use_attention)

        batch_array = activations.cpu().numpy()

        fname = f"activations_part_{part_idx:05d}.npz"
        path = os.path.join(output_dir, fname)
        np.savez_compressed(path, activations=batch_array)
        shard_paths.append(path)

    utils.log_info(f"Saved {len(shard_paths)} activation shards to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model activations in shards (and optional combined file).")
    parser.add_argument("--data", "-d", type=str, required=True, help="CSV with probing data including spans/indices.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Directory in which to write activation_part_xxxxx.npz files.")
    parser.add_argument("--model", "-m", type=str, default="gpt2", help="Key into MODEL_CONFIGS (e.g. 'gpt2' or 'gemma2b').")
    parser.add_argument("--revision", type=str, default=None, help="Model revision or checkpoint (e.g., 'step1000', 'main').") 
    parser.add_argument("--max_rows", type=int, default=0, help="Maximum number of rows to process from the data file. 0 means no limit.")
    parser.add_argument("--use_attention", action="store_true", help="Extract per-head attention outputs instead of hidden states.")
    args = parser.parse_args()

    extract_and_save(
        data_path=args.data,
        output_dir=args.output_dir,
        model_key=args.model,
        revision=args.revision,
        max_rows=args.max_rows,
        use_attention=args.use_attention
    )