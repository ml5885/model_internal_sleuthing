import os
import subprocess
import argparse
from src import config, utils

def run_activation_extraction(model_key, dataset, revision=None, activations_dir=None):
    utils.log_info(f"Starting activation extraction for {model_key} (rev={revision}) on dataset {dataset}...")
    dataset_file = os.path.join("data", f"{dataset}.csv")

    # where to write activations
    base = activations_dir if activations_dir else config.OUTPUT_DIR
    save_dir = os.path.join(base, f"{model_key}_{dataset}_reps")
    os.makedirs(save_dir, exist_ok=True)

    combined = save_dir + ".npz"
    if os.path.isfile(combined):
        utils.log_info(f"Found existing activation file: {combined}")
        print(f"Using existing activation file: {combined}")
        return combined
    if os.path.isdir(save_dir) and any(fn.startswith("activations_part") for fn in os.listdir(save_dir)):
        utils.log_info(f"Found existing activation shards in: {save_dir}")
        print(f"Using existing activation shards: {save_dir}")
        return save_dir

    utils.log_info("No existing activations found; extracting new ones...")
    print("No existing activations found; extracting new ones...")
    cmd = [
        "python", "-m", "src.activation_extraction",
        "--data", dataset_file,
        "--output-dir", save_dir,
        "--model", model_key,
    ]
    if revision is not None:
        cmd += ["--revision", revision]
    subprocess.run(cmd, check=True)

    return save_dir

def run_analysis(model_key, dataset, output_dir=None):
    utils.log_info("Running analysis on activations...")
    activations_dir = os.path.join(
        output_dir if output_dir else config.OUTPUT_DIR,
        f"{model_key}_{dataset}_reps"
    )
    subprocess.run([
        "python", "-m", "src.analysis",
        "--activations-dir", activations_dir,
        "--labels", os.path.join("data", f"{dataset}.csv"),
        "--model", model_key,
        "--dataset", dataset,
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline for probing tasks.")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model key (e.g. 'gpt2').")
    parser.add_argument("--revision", type=str, default=None,
                        help="Checkpoint revision to load (e.g. 'step1000-tokens5B').")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset label (e.g. 'gum_dataset').")
    parser.add_argument("--experiment", type=str,
                        choices=["inflection", "lexeme"],
                        help="Specific experiment to run.")
    parser.add_argument("--lambda_reg", type=float, default=1e-3,
                        help="Regularization parameter for probes.")
    parser.add_argument("--probe_type", type=str, default="reg",
                        choices=["reg", "mlp", "nn", "rf"],
                        help="Type of probe to use.")
    parser.add_argument("--pca_dim", type=int, default=0,
                        help="Dimensionality for PCA reduction.")
    parser.add_argument("--no_analysis", action="store_true",
                        help="Skip analysis after running experiments.")
    parser.add_argument("--activations_dir", type=str, default=None,
                        help="Custom output directory for activation shards.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for probe results.")
    args = parser.parse_args()

    model_key = args.model
    revision = args.revision
    dataset = args.dataset
    probe_type = args.probe_type
    pca_dim = args.pca_dim
    pca = pca_dim > 0

    # 1) extract (or re-use) activations, using the chosen revision
    reps = run_activation_extraction(model_key, dataset, revision, args.activations_dir)

    # 2) build per-task probe output dirs
    base_probe_dir = args.output_dir if args.output_dir else config.OUTPUT_DIR
    probe_output_dirs = {
        task: utils.get_probe_output_dir(
            dataset, model_key, task, probe_type,
            pca=pca, pca_dim=pca_dim,
            base_dir=base_probe_dir
        )
        for task in ["inflection", "lexeme"]
    }

    # 3) run probes
    for task in ([args.experiment] if args.experiment else ["inflection", "lexeme"]):
        if task not in probe_output_dirs:
            raise ValueError(f"Unknown experiment name: {task}")
        exp_args = [
            "--activations", reps,
            "--labels", os.path.join("data", f"{dataset}.csv"),
            "--task", task,
            "--lambda_reg", str(args.lambda_reg),
            "--exp_label", f"{model_key}_{task}",
            "--dataset", dataset,
            "--probe_type", probe_type,
            "--pca_dim", str(pca_dim),
            "--output_dir", probe_output_dirs[task],
        ]
        if revision is not None:
            exp_args += ["--revision", revision]
        utils.log_info(f"Running probe for task={task}")
        subprocess.run(["python", "-m", "src.train"] + exp_args, check=True)

    # 4) optional analysis
    if not args.experiment and not args.no_analysis:
        run_analysis(model_key, dataset, args.activations_dir)

    utils.log_info("All experiments and analysis completed.")

if __name__ == "__main__":
    main()
