"""
step5_train/automl_hyperdrive.py
Azure AutoML HyperDrive Integration for VulnDetector
Authors: Josiah Chuku & Dr. Jinwei Liu, FAMU 2026

NEW CONTRIBUTION vs capstone:
- Replaces manual hyperparameter tuning with systematic HyperDrive search
- Champion/challenger model promotion loop
- Drift-triggered retraining via Azure ML pipelines
"""

import argparse
import json
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.hyperdrive import (
    HyperDriveConfig,
    PrimaryMetricGoal,
    RandomParameterSampling,
    BanditPolicy,
    choice,
    loguniform,
    uniform,
)


def get_or_create_compute(ws, compute_name="gpu-cluster"):
    """Get or create AML GPU compute cluster."""
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print(f"Using existing compute: {compute_name}")
    except ComputeTargetException:
        print(f"Creating compute cluster: {compute_name}")
        config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_NC6",       # 1x K80 GPU — cost-efficient
            min_nodes=0,
            max_nodes=4,
            idle_seconds_before_scaledown=300,
        )
        compute_target = ComputeTarget.create(ws, compute_name, config)
        compute_target.wait_for_completion(show_output=True)
    return compute_target


def define_search_space():
    """
    HyperDrive search space over VulnDetector's key hyperparameters.
    Replaces the manually-tuned values from the capstone:
      lr=1e-4, batch_size=32, encoder_layers=6, gcn_hidden=256
    """
    return RandomParameterSampling({
        # Learning rate — log-uniform over 2 decades
        "--lr": loguniform(-5, -3),               # 1e-5 to 1e-3

        # Batch size — discrete choices
        "--batch_size": choice(16, 32, 64),

        # Transformer encoder depth
        "--encoder_layers": choice(4, 6, 8),

        # GCN hidden dimension
        "--gcn_hidden": choice(128, 256, 512),

        # Dropout rate
        "--dropout": uniform(0.05, 0.3),

        # Class weight for imbalanced training
        # Higher = more penalty for missing vulnerable functions
        "--pos_weight": choice(2.0, 4.0, 8.0, 16.0),

        # Loss function: 0=BCE, 1=Focal
        "--loss_type": choice("bce", "focal"),

        # Focal loss gamma (only used when loss_type=focal)
        "--focal_gamma": choice(1.0, 2.0, 3.0),
    })


def run_hyperdrive_search(
    ws,
    experiment_name,
    compute_target,
    max_total_runs=20,
    max_concurrent_runs=4,
):
    """
    Launch HyperDrive search and return the best run.

    Primary metric: val_f1 (maximise)
    Early termination: Bandit policy — stops runs whose val_f1
    falls more than 10% below the best run at any reporting interval.
    """
    env = Environment.from_pip_requirements(
        name="vulndetector-env",
        file_path="requirements.txt",
    )

    script_config = ScriptRunConfig(
        source_directory=".",
        script="step5_train/train.py",
        compute_target=compute_target,
        environment=env,
        arguments=[
            "--train_path", "data/train_balanced_40k.csv",
            "--val_path",   "data/val_balanced_4k.csv",
            "--cache_path", "data/dfg_cache.pkl",
            "--epochs",     "10",
            "--log_mlflow", "true",
        ],
    )

    early_termination = BanditPolicy(
        evaluation_interval=1,
        slack_factor=0.10,      # Terminate if >10% below best val_f1
        delay_evaluation=2,     # Don't terminate in first 2 epochs
    )

    hd_config = HyperDriveConfig(
        run_config=script_config,
        hyperparameter_sampling=define_search_space(),
        primary_metric_name="val_f1",
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=max_total_runs,
        max_concurrent_runs=max_concurrent_runs,
        policy=early_termination,
    )

    experiment = Experiment(ws, experiment_name)
    hd_run = experiment.submit(hd_config)

    print(f"HyperDrive run submitted: {hd_run.id}")
    print(f"Monitor at: {hd_run.get_portal_url()}")

    hd_run.wait_for_completion(show_output=True)
    return hd_run


def get_best_run(hd_run, output_path):
    """Extract best run, download checkpoint, save hyperparameters."""
    best_run = hd_run.get_best_run_by_primary_metric()

    if best_run is None:
        print("No best run found — all runs may have failed.")
        return None

    best_metrics = best_run.get_metrics()
    best_params  = best_run.get_details()["runDefinition"]["arguments"]

    print("\n=== Best HyperDrive Run ===")
    print(f"Run ID    : {best_run.id}")
    print(f"Val F1    : {best_metrics.get('val_f1', 'N/A'):.4f}")
    print(f"Val AUC   : {best_metrics.get('val_auc', 'N/A'):.4f}")
    print(f"Parameters: {best_params}")

    # Download challenger checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    best_run.download_file(
        name="outputs/best_model_final.pt",
        output_file_path=output_path,
    )

    # Save best hyperparameters for paper reporting
    results = {
        "run_id": best_run.id,
        "val_f1": best_metrics.get("val_f1"),
        "val_auc": best_metrics.get("val_auc"),
        "val_mcc": best_metrics.get("val_mcc"),
        "hyperparameters": best_params,
    }
    results_path = output_path.replace(".pt", "_hparams.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nChallenger model saved to: {output_path}")
    print(f"Hyperparameters saved to : {results_path}")
    return best_run


def main():
    parser = argparse.ArgumentParser(
        description="VulnDetector Azure AutoML HyperDrive Search"
    )
    parser.add_argument("--subscription_id",  required=True)
    parser.add_argument("--resource_group",   required=True)
    parser.add_argument("--workspace_name",   required=True)
    parser.add_argument("--experiment_name",  default="vulndetector-hyperdrive")
    parser.add_argument("--max_runs",         type=int, default=20)
    parser.add_argument("--champion_path",    default="checkpoints/best_model_final.pt")
    parser.add_argument("--output_path",      default="checkpoints/challenger_model.pt")
    args = parser.parse_args()

    # Connect to Azure ML workspace
    ws = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
    )
    print(f"Connected to workspace: {ws.name}")

    compute_target = get_or_create_compute(ws)
    hd_run = run_hyperdrive_search(
        ws,
        args.experiment_name,
        compute_target,
        max_total_runs=args.max_runs,
    )
    get_best_run(hd_run, args.output_path)


if __name__ == "__main__":
    main()
