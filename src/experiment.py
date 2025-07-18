import os
import subprocess
import argparse
from joblib import Parallel, delayed
from src import config, utils

def run_activation_extraction(model_key, dataset, revision=None, activations_dir_override=None, max_rows=0, use_attention=False):
    utils.log_info(f"Starting activation extraction for {model_key} (rev={revision}) on dataset {dataset}...")

    dataset_file = os.path.join("data", f"{dataset}.csv")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Could not find dataset file for {dataset} at {dataset_file}")

    base_output_path = activations_dir_override if activations_dir_override else config.OUTPUT_DIR

    revision_component = f"_{revision}" if revision else ""
    attention_component = "_attn" if use_attention else ""

    output_leaf_name = f"{model_key}{revision_component}_{dataset}_reps{attention_component}"

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
    if max_rows > 0:
        cmd += ["--max_rows", str(max_rows)]
    if use_attention:
        cmd += ["--use_attention"]
    subprocess.run(cmd, check=True)

    if os.path.isfile(combined_npz_path):
        utils.log_info(f"Created activation file: {combined_npz_path}")
        return combined_npz_path
    utils.log_info(f"Created activation shards in: {activations_output_dir}")
    return activations_output_dir

def run_analysis(activations_input_path, model_key_for_logging, dataset_name):
    utils.log_info(f"Running analysis on activations from: {activations_input_path} for model {model_key_for_logging}, dataset {dataset_name}")
    labels_file = os.path.join("data", f"{dataset_name}.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Could not find dataset file for {dataset_name} at {labels_file}")

    subprocess.run([
        "python", "-m", "src.analysis",
        "--activations-dir", activations_input_path,
        "--labels", labels_file,
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
    parser.add_argument("--no_analysis", action="store_true", default=True, help="Skip analysis after running experiments.")
    parser.add_argument("--activations_dir", type=str, default=None, help="Custom base output directory for activation files/shards.")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom base output directory for steering results.")
    parser.add_argument("--probe_dir", type=str, default=None, help="Custom base output directory for probe results.")
    parser.add_argument("--max_rows", type=int, default=75000, help="Maximum number of rows to sample from dataset for activation extraction.")
    parser.add_argument("--use_attention", action="store_true", help="Run extraction on attention outputs rather than residual stream.")
    parser.add_argument("--steering", action="store_true", help="Run steering experiment after probing.")
    parser.add_argument("--lambda_steer", type=float, nargs='+', default=[1.5], help="One or more lambda for steering vector strength.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for steering.")
    args = parser.parse_args()

    model_key = args.model
    revision = args.revision
    dataset = args.dataset
    probe_type = args.probe_type
    pca_dim = args.pca_dim
    pca = pca_dim > 0

    revision_component = f"_{revision}" if revision else ""
    effective_model_key_for_paths = f"{model_key}{revision_component}"

    reps_path = run_activation_extraction(model_key, dataset, revision, args.activations_dir, args.max_rows, args.use_attention)

    # Use --probe_dir if provided, otherwise default to config.OUTPUT_DIR/probes
    base_probe_dir = args.probe_dir if args.probe_dir else os.path.join(config.OUTPUT_DIR, "probes")
    attention_component = "_attn" if args.use_attention else ""
    probe_output_dirs = {
        task: os.path.join(
            base_probe_dir,
            f"{dataset}_{effective_model_key_for_paths}_{task}_{probe_type}{attention_component}" +
            (f"_pca{pca_dim}" if pca else "")
        )
        for task in ["inflection", "lexeme"]
    }

    # Use a separate steering directory for steering results
    base_steering_dir = args.output_dir if args.output_dir else os.path.join(config.OUTPUT_DIR, "steering")

    if not args.steering:
        for task in ([args.experiment] if args.experiment else ["inflection", "lexeme"]):
            if task not in probe_output_dirs:
                raise ValueError(f"Unknown experiment name: {task}")
            
            exp_label = effective_model_key_for_paths
            
            labels_file = os.path.join("data", f"{dataset}.csv")
            if not os.path.exists(labels_file):
                raise FileNotFoundError(f"Could not find dataset file for {dataset} at {labels_file}")

            exp_args = [
                "--activations", reps_path,
                "--labels", labels_file,
                "--task", task,
                "--lambda_reg", str(args.lambda_reg),
                "--exp_label", exp_label,
                "--dataset", dataset,
                "--probe_type", probe_type,
                "--pca_dim", str(pca_dim),
                "--output_dir", probe_output_dirs[task],  # always probes dir
            ]
            
            utils.log_info(f"Running probe for task={task}, model_config={effective_model_key_for_paths}, dataset={dataset}")
            subprocess.run(["python", "-m", "src.train"] + exp_args, check=True)

    if args.steering:
        utils.log_info("Proceeding to steering experiment...")
        task = "inflection"
        probe_dir_for_steering = probe_output_dirs.get(task)
        if not probe_dir_for_steering or not os.path.exists(probe_dir_for_steering):
            utils.log_info(f"Probe directory for inflection task not found at {probe_dir_for_steering}. Running probe training first.")
            exp_args = [
                "--activations", reps_path,
                "--labels", os.path.join("data", f"{dataset}.csv"),
                "--task", task,
                "--lambda_reg", str(args.lambda_reg),
                "--exp_label", effective_model_key_for_paths,
                "--dataset", dataset,
                "--probe_type", probe_type,
                "--pca_dim", str(pca_dim),
                "--output_dir", probe_dir_for_steering,  # always probes dir
            ]
            subprocess.run(["python", "-m", "src.train"] + exp_args, check=True)

        if not any(f.startswith("probe_layer_") for f in os.listdir(probe_dir_for_steering)):
            utils.log_info(f"No trained probes found in {probe_dir_for_steering} even after training attempt. Cannot run steering.")
            return

        from src.steering import run_steering as run_single_steering

        def run_steering_for_lambda(lambda_val):
            steering_output_dir = os.path.join(
                base_steering_dir,
                f"{dataset}_{effective_model_key_for_paths}_{probe_type}{attention_component}_lambda{lambda_val:.1f}"
            )
            steering_args = argparse.Namespace(
                activations=reps_path,
                labels=os.path.join("data", f"{dataset}.csv"),
                probe_dir=probe_dir_for_steering,
                output_dir=steering_output_dir,  # always steering dir
                probe_type=probe_type,
                lambda_steer=lambda_val,
                num_samples=args.num_samples
            )
            utils.log_info(f"Running steering for model={effective_model_key_for_paths}, dataset={dataset}, lambda_steer={lambda_val}")
            run_single_steering(steering_args)

        # Run first lambda sequentially to ensure everything is set up.
        if args.lambda_steer:
            utils.log_info("Running first lambda value sequentially...")
            run_steering_for_lambda(args.lambda_steer[0])

        # Run the rest sequentially (avoids duplicate 40 GB arrays per process)
        for l in args.lambda_steer[1:]:
            run_steering_for_lambda(l)

    if not args.no_analysis:
        run_analysis(reps_path, effective_model_key_for_paths, dataset)

    utils.log_info(f"All experiments and analysis completed for {effective_model_key_for_paths} on {dataset}.")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
    main()
if __name__ == "__main__":
    main()
