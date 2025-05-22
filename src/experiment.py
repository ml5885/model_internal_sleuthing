import os
import subprocess
import argparse
from src import config, utils

def run_activation_extraction(model_key, dataset, revision=None, activations_dir_override=None):
    utils.log_info(f"Starting activation extraction for {model_key} (rev={revision}) on dataset {dataset}...")

    dataset_file = os.path.join("data", f"{dataset}.csv")

    base_output_path = activations_dir_override if activations_dir_override else config.OUTPUT_DIR

    revision_component = f"_{revision}" if revision else ""

    output_leaf_name = f"{model_key}{revision_component}_{dataset}_reps"

    activations_output_dir = os.path.join(base_output_path, output_leaf_name)
    os.makedirs(activations_output_dir, exist_ok=True)

    combined_npz_path = os.path.join(base_output_path, output_leaf_name + ".npz")

    if os.path.isfile(combined_npz_path):
        utils.log_info(f"Found existing activation file: {combined_npz_path}")
        print(f"Using existing activation file: {combined_npz_path}")
        return combined_npz_path
    
    if os.path.isdir(activations_output_dir) and any(fn.startswith("activations_part") for fn in os.listdir(activations_output_dir)):
        utils.log_info(f"Found existing activation shards in: {activations_output_dir}")
        print(f"Using existing activation shards: {activations_output_dir}")
        return activations_output_dir

    utils.log_info(f"No existing activations found; extracting new ones into {activations_output_dir}")
    print(f"No existing activations found; extracting new ones into {activations_output_dir}")
    cmd = [
        "python", "-m", "src.activation_extraction",
        "--data", dataset_file,
        "--output-dir", activations_output_dir,
        "--model", model_key,
    ]
    if revision is not None:
        cmd += ["--revision", revision]
    subprocess.run(cmd, check=True)

    if os.path.isfile(combined_npz_path):
        utils.log_info(f"Created activation file: {combined_npz_path}")
        return combined_npz_path
    utils.log_info(f"Created activation shards in: {activations_output_dir}")
    return activations_output_dir

def run_analysis(activations_input_path, model_key_for_logging, dataset_name):
    utils.log_info(f"Running analysis on activations from: {activations_input_path} for model {model_key_for_logging}, dataset {dataset_name}")
    subprocess.run([
        "python", "-m", "src.analysis",
        "--activations-dir", activations_input_path,
        "--labels", os.path.join("data", f"{dataset_name}.csv"),
        "--model", model_key_for_logging,
        "--dataset", dataset_name,
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline for probing tasks.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model key (e.g. 'gpt2').")
    parser.add_argument("--revision", type=str, default=None, help="Checkpoint revision to load (e.g. 'step1000-tokens5B').")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset label (e.g. 'gum_dataset').")
    parser.add_argument("--experiment", type=str, choices=["inflection", "lexeme"], help="Specific experiment to run.")
    parser.add_argument("--lambda_reg", type=float, default=1e-3, help="Regularization parameter for probes.")
    parser.add_argument("--probe_type", type=str, default="reg", choices=["reg", "mlp", "nn", "rf"], help="Type of probe to use.")
    parser.add_argument("--pca_dim", type=int, default=0, help="Dimensionality for PCA reduction.")
    parser.add_argument("--no_analysis", action="store_true", help="Skip analysis after running experiments.")
    parser.add_argument("--activations_dir", type=str, default=None, help="Custom base output directory for activation files/shards.")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom base output directory for probe results.")
    args = parser.parse_args()

    model_key = args.model
    revision = args.revision
    dataset = args.dataset
    probe_type = args.probe_type
    pca_dim = args.pca_dim
    pca = pca_dim > 0

    revision_component = f"_{revision}" if revision else ""
    effective_model_key_for_paths = f"{model_key}{revision_component}"

    reps_path = run_activation_extraction(model_key, dataset, revision, args.activations_dir)

    base_probe_dir = args.output_dir if args.output_dir else os.path.join(config.OUTPUT_DIR, "probes")
    probe_output_dirs = {
        task: utils.get_probe_output_dir(
            dataset, effective_model_key_for_paths, task, probe_type,
            pca=pca, pca_dim=pca_dim,
            base_dir=base_probe_dir
        )
        for task in ["inflection", "lexeme"]
    }

    for task in ([args.experiment] if args.experiment else ["inflection", "lexeme"]):
        if task not in probe_output_dirs:
            raise ValueError(f"Unknown experiment name: {task}")
        
        exp_label = f"{effective_model_key_for_paths}_{task}"
        
        exp_args = [
            "--activations", reps_path,
            "--labels", os.path.join("data", f"{dataset}.csv"),
            "--task", task,
            "--lambda_reg", str(args.lambda_reg),
            "--exp_label", exp_label,
            "--dataset", dataset,
            "--probe_type", probe_type,
            "--pca_dim", str(pca_dim),
            "--output_dir", probe_output_dirs[task],
        ]
        
        utils.log_info(f"Running probe for task={task}, model_config={effective_model_key_for_paths}, dataset={dataset}")
        subprocess.run(["python", "-m", "src.train"] + exp_args, check=True)

    if not args.experiment and not args.no_analysis:
        run_analysis(reps_path, effective_model_key_for_paths, dataset)

    utils.log_info(f"All experiments and analysis completed for {effective_model_key_for_paths} on {dataset}.")

if __name__ == "__main__":
    main()