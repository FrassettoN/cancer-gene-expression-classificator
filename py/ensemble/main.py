import os
import subprocess
import yaml
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from utils.cli import cli
from utils.logger import log_to_file
from utils.plots import roc_plot, accuracy_bar_chart
from evaluate_model import evaluate_models_cv


def softmax(logits):
    # Apply the softmax function row-wise to convert scores to probabilities
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def main():
    configs_path, input_output_folder, seed = cli()

    data_path = os.path.join(input_output_folder, "data")
    processed_path = os.path.join(data_path, "processed")

    with open(configs_path) as f:
        configs = yaml.safe_load(f)

    n_features = configs["n_features"]
    file_paths = configs["datasets"]
    model_configs = configs["models"]

    # List to store the accuracy values for different datasets
    accuracies = {}

    # List to store the standard errors of the mean of accuracies for each dataset
    std_errors = {}

    # Setup for k-fold, models, and output options
    evaluation_type = "kfold"

    models = list(model_configs.keys())
    logits_active = True
    save_selected_features = True
    metrics = True
    chart = True

    # Generate a timestamp to make log filenames unique
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Base directory for all results
    if n_features == 10000:
        n_features_str = "10k"
    else:
        n_features_str = n_features
    results_dir = os.path.join(input_output_folder, "results", 
    f"Ensemble_f{n_features_str}_s{seed}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    abs_path = os.path.abspath(results_dir)

    # Directory to store model accuracy logs
    acc_path = os.path.join(abs_path, f"accuracies.log")

    # Define the path for the metrics log
    metrics_path = os.path.join(abs_path, f"metrics.log")

    paths = {"abs_path": abs_path, "acc_path": acc_path, "metrics_path": metrics_path, "data_path": data_path}

    # Define the path for saving chart results
    charts_dir = os.path.join(abs_path, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Define the path for saving logits outputs (model predictions before softmax)
    logits_dir = os.path.join(results_dir, "logits")

    # If the processed data folder doesn't exist, run the preprocessing script - Commented for practical reasons
    # processed_folder = os.path.join("data", "processed")
    # if not os.path.exists(processed_folder):
    #     subprocess.run(["python", os.path.join("scripts", "datasets_trasp_mod.py")])

    # Set up CPU usage: detect all cores and configure parallel processing limits
    num_cores = multiprocessing.cpu_count()
    os.environ["LOKY_MAX_CPU_COUNT"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device, "\n")

    log_to_file(f"\nMODEL RESULTS", acc_path)

    for file_path, cancer_type in file_paths.items():
        
        # Log dataset name and define dataset path
        log_to_file(f"\n--- Results for: {cancer_type} ---", acc_path)
        dataset_path = os.path.join(processed_path, file_path)

        # Load dataset
        df = pd.read_csv(dataset_path, header=None)

        # Separate labels and features
        labels = df.iloc[:, 0]
        features = df.drop(columns=0)
        features.columns = range(features.shape[1])

        # Count class distribution
        cancer_count = (labels == 1).sum()
        normal_count = (labels == 0).sum()

        # Log class distribution for the dataset
        log_to_file(f"\nCancer: {cancer_count}", acc_path)
        log_to_file(f"Normal: {normal_count}\n", acc_path)

        # Convert to NumPy arrays
        labels = np.array(labels, dtype=np.int32)
        features = np.array(features, dtype=np.float32)

        # Log shapes of features and labels
        log_to_file(f"Shape of features: {features.shape}", acc_path)
        log_to_file(f"Shape of labels: {labels.shape}", acc_path)

        # Get data properties
        num_samples = features.shape[0]  # Number of samples (rows)
        num_features = features.shape[1]  # Number of features (columns)
        num_classes = len(
            torch.unique(torch.tensor(labels))
        )  # Number of unique classes in the labels

        # Evaluate Models with Leave-One-Out Cross-Validation
        avg_loss, avg_accuracy, sem = evaluate_models_cv(
            models,
            configs,
            features,
            labels,
            num_classes,
            cancer_type,
            evaluation_type,
            device,
            logits_active,
            save_selected_features,
            paths,
            seed,
        )

        # Store results
        for model_name in models:
            if model_name not in accuracies:
                accuracies[model_name] = []
            if model_name not in std_errors:
                std_errors[model_name] = []

            accuracies[model_name].append(avg_accuracy[model_name] / 100)
            std_errors[model_name].append(sem[model_name])

        # Log final statistics
        log_to_file("\nFinal Test Results", acc_path)
        for model_name in models:
            if model_name == "SVM":
                log_to_file(
                    f"{model_name} - Average Hinge Loss: {avg_loss[model_name]:.3f}, Accuracy: {avg_accuracy[model_name]:.2f}%",
                    acc_path,
                )
            else:
                log_to_file(
                    f"{model_name} - Average Loss: {avg_loss[model_name]:.3f}, Accuracy: {avg_accuracy[model_name]:.2f}%",
                    acc_path,
                )
        log_to_file("", acc_path)

    print(f"The accuracy results have been saved as 'accuracies.log' in: {abs_path}")

    # ---- Metrics part ----

    if metrics:

        for file_path, cancer_type in file_paths.items():

            log_to_file(f"\n--- Results for: {cancer_type} ---", metrics_path)
            y_test_dir = os.path.join(logits_dir, cancer_type, "ytest")

            fold_files = [
                f
                for f in os.listdir(y_test_dir)
                if f.startswith(f"ytest_{cancer_type}") and f.endswith(".txt")
            ]
            num_folds = len(fold_files)

            for model_name in models:

                all_y_test = []
                all_y_scores = []

                for fold in range(num_folds):

                    # Load real labels
                    y_test_path = os.path.join(
                        y_test_dir, f"ytest_{cancer_type}_{fold + 1}.txt"
                    )
                    y_test = np.loadtxt(y_test_path, dtype=int)

                    # Load logits
                    logits_dataset_model_dir = os.path.join(
                        logits_dir, cancer_type, model_name
                    )
                    logits_path = os.path.join(
                        logits_dataset_model_dir,
                        f"logits_{cancer_type}_{fold + 1}_{model_name}.txt",
                    )
                    if os.path.exists(logits_path):
                        logits = np.loadtxt(logits_path)
                        probs = softmax(logits)
                        y_scores = probs[:, 1]

                        # Save predictions by concatenation
                        all_y_test.append(y_test)
                        all_y_scores.append(y_scores)

                    else:
                        raise ValueError(f"File not found: {logits_path}")

                # Concatenate out-of-fold predictions and calculate overall AUC
                all_y_test_concat = np.concatenate(all_y_test)
                all_y_scores_concat = np.concatenate(all_y_scores)
                mean_auc = roc_auc_score(all_y_test_concat, all_y_scores_concat)

                # Calculate the error under the ROC curve
                error_under_roc = (1 - mean_auc) * 100

                # Log overall AUC (out-of-fold) and corresponding error under ROC
                log_to_file(f"\nModel: {model_name}", metrics_path)
                log_to_file(
                    f"  Overall AUC (out-of-fold): {mean_auc:.3f}", metrics_path
                )
                log_to_file(
                    f"  Error under ROC (%):       {error_under_roc:.3f}\n",
                    metrics_path,
                )

                # Plot ROC curve using out-of-fold predictions (overall ROC)
                roc_plot(
                    all_y_test_concat,
                    all_y_scores_concat,
                    mean_auc,
                    model_name,
                    cancer_type,
                    charts_dir,
                )

        print(f"The metrics have been saved as 'metrics.log' in: {abs_path}")
        print(f"ROC curves have been saved in: {charts_dir}")

    # ---- Chart part ----

    if chart:
        accuracy_bar_chart(file_paths, models, accuracies, std_errors, charts_dir)
        print(f"The chart has been saved as 'bar_chart.png' in: {charts_dir}")


if __name__ == "__main__":
    main()
