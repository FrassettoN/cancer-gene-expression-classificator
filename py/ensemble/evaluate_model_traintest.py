import gc
import os
from datetime import datetime
from types import SimpleNamespace
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from libsvm.svmutil import svm_parameter, svm_predict, svm_problem, svm_train
from pyDeepInsight import ImageTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from evaluate_model import load_model, testing_loop, training_loop
from utils import (
    channel_expansion,
    feature_selection,
    neighbor_informed_gene_expression,
)
from utils.cli import cli
from utils.logger import log_to_file, restore_output, suppress_output
from utils.seed import set_seed


def process_fold(n_features, X_train, X_test, y_train, y_test, device, seed):

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler on the training features and apply the transformation
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same transformation to the test features (without refitting)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection (chi2 first, then optional RfF)
    X_train_scaled_chi2, X_test_scaled_chi2, selected_feature_indices_chi2 = feature_selection(
        X_train_scaled,
        X_test_scaled,
        y_train,
        device,
        method="chi2",
        max_features=10000,
        seed=seed,
    )

    if n_features == 30:
        X_train_scaled, X_test_scaled, selected_feature_indices_rff = feature_selection(
            X_train_scaled_chi2,
            X_test_scaled_chi2,
            y_train,
            device,
            method="RfF",
            max_features=30,
            seed=seed,
        )

        chi2_idx = np.asarray(selected_feature_indices_chi2)
        rff_idx = np.asarray(selected_feature_indices_rff)

        if rff_idx.dtype == bool:
            selected_feature_indices = chi2_idx[rff_idx]
        else:
            selected_feature_indices = chi2_idx[rff_idx.astype(int)]
    elif n_features == 10000:
        X_train_scaled, X_test_scaled = X_train_scaled_chi2, X_test_scaled_chi2
        selected_feature_indices = np.asarray(selected_feature_indices_chi2)
    else:
        raise ValueError("Number of features not supported")

    # Original DeepInsight
    tsne = TSNE(
        n_components=2, perplexity=8, metric="cosine", random_state=1701
    )  # t-SNE for dimensionality reduction
    it = ImageTransformer(
        feature_extractor=tsne, pixels=8
    )  # Initialize Improved DeepInsight with t-SNE as the feature extractor and set image size to 64 pixels
    _ = it.fit(
        X_train_scaled, plot=False
    )  # Fit the ImageTransformer to the training data


    return X_train_scaled, X_test_scaled, y_train, y_test, it, selected_feature_indices


def evaluate_models_traintest(
    models,
    configs,
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes,
    dataset_name,
    device,
    logits_active,
    save_selected_features,
    paths,
    seed,
):
    model_configs = configs["models"]
    n_features = configs["n_features"]

    abs_path = paths["abs_path"]
    acc_path = paths["acc_path"]
    data_path = paths["data_path"]

    fs_seed = seed
    classificator_seed = seed

    loss_eval = {}
    accuracy_eval = {}

    if logits_active:
        logits_dir = os.path.join(abs_path, "logits")
        os.makedirs(logits_dir, exist_ok=True)

    if save_selected_features:
        features_dir = os.path.join(abs_path, "selected_features")
        os.makedirs(features_dir, exist_ok=True)

    log_to_file(f"FS_seed: {fs_seed}", acc_path)

    # Shared preprocessing: fit on train, transform test, select features, build DeepInsight transformer
    # This is the train/test equivalent of process_fold.
    X_train_scaled, X_test_scaled, y_train, y_test, it, selected_feature_indices = process_fold(n_features, X_train, X_test, y_train, y_test, device, seed)

    for model_name in models:
        set_seed(classificator_seed)
        log_to_file(f"\nClassificator_seed: {classificator_seed}", acc_path)
        log_to_file(f"Model: {model_name}", acc_path)

        config_dict = model_configs.get(model_name)
        if config_dict is None:
            raise ValueError(f"Model '{model_name}' is not supported")

        config = SimpleNamespace(**config_dict)
        batch_size = getattr(config, "batch_size", None)
        num_epochs = getattr(config, "num_epochs", None)
        learningRate = getattr(config, "learningRate", None)
        alpha = getattr(config, "alpha", None)
        beta = getattr(config, "beta", None)
        sigma = getattr(config, "sigma", None)
        k = getattr(config, "k", None)

        if save_selected_features and selected_feature_indices is not None:
            features_save_path = os.path.join(
                features_dir, f"selected_features_{dataset_name}.txt"
            )
            np.savetxt(features_save_path, np.array(selected_feature_indices), fmt="%d")


        if model_name in ["DI-CNN+", "IT-KAN"]:

            # Improved DeepInsight with channel wise expansion (apply the channel expansion function to both the training and test sets)
            if model_name in ["DI-CNN+"]:
                X_train_scaled = channel_expansion(it, X_train_scaled)  # img_train
                X_test_scaled = channel_expansion(it, X_test_scaled)  # img_test

            if model_name == "IT-KAN":
                X_train_informed, y_train_tensor = (
                    neighbor_informed_gene_expression(
                        X_train_scaled,
                        it.coords(),
                        y_train,
                        alpha=alpha,
                        beta=beta,
                        k=k,
                        sigma=sigma,
                    )
                )
                X_test_informed, y_test_tensor = neighbor_informed_gene_expression(
                    X_test_scaled,
                    it.coords(),
                    y_test,
                    alpha=alpha,
                    beta=beta,
                    k=k,
                    sigma=sigma,
                )

                # Normalize features to [0, 1]
                minmax_scaler = MinMaxScaler(feature_range=(0, 1))
                X_train_informed_scaled = minmax_scaler.fit_transform(
                    X_train_informed.cpu()
                )
                X_test_informed_scaled = minmax_scaler.transform(
                    X_test_informed.cpu()
                )

                # Convert scaled data back to PyTorch tensors
                X_train_informed_scaled = torch.tensor(
                    X_train_informed_scaled, dtype=torch.float32, device=device
                )
                X_test_informed_scaled = torch.tensor(
                    X_test_informed_scaled, dtype=torch.float32, device=device
                )

                # Build TensorDatasets for training and testing
                train_dataset = TensorDataset(
                    X_train_informed_scaled, y_train_tensor
                )
                test_dataset = TensorDataset(X_test_informed_scaled, y_test_tensor)

        if model_name != "IT-KAN":

            # Converting to PyTorch Tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(
                device
            )
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(
                device
            )
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

            # Creating DataSets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Creating DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if model_name == "SVM":
            prob = svm_problem(y_train, X_train_scaled)
            param = svm_parameter("-t 0 -q -b 1")
            svm_model = svm_train(prob, param)

            suppress_output()
            y_pred_svm, _, logits_svm = svm_predict(y_test, X_test_scaled, svm_model, "-b 1")
            restore_output()

            logits_svm = np.array(logits_svm)
            labels = np.array(svm_model.get_labels())
            logits_fixed = logits_svm[:, [np.where(labels == 0)[0][0], np.where(labels == 1)[0][0]]]

            if logits_active:
                logits_dataset_dir = os.path.join(logits_dir, dataset_name)
                os.makedirs(logits_dataset_dir, exist_ok=True)
                np.savetxt(
                    os.path.join(logits_dataset_dir, f"logits_{dataset_name}_{model_name}.txt"),
                    logits_fixed,
                    fmt="%.6f",
                )

            accuracy = (y_pred_svm == y_test).mean() * 100
            hinge_loss = sum(max(0, 1 - y * y_pred) for y, y_pred in zip(y_test, y_pred_svm)) / len(y_test)

            loss_eval[model_name] = hinge_loss
            accuracy_eval[model_name] = accuracy

        else:
            # Load model
            input_dim = n_features

            if model_name == "DI-CNN+":
                input_dim = X_train_scaled.shape[1]

            model = load_model(model_name, input_dim, num_classes, config).to(device)
            criterion = nn.CrossEntropyLoss()

            optimizer_type = getattr(config, "optimizer", None)
            weight_decay = getattr(config, "weight_decay", 0)

            if optimizer_type == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weight_decay)
            elif optimizer_type == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)
            else:
                raise ValueError(f"Optimizer for '{model_name}' not specified")

            _ = training_loop(num_epochs, optimizer, model, criterion, train_loader, device)
            test_loss, accuracy, logits_nn = testing_loop(model, criterion, test_loader, device)

            loss_eval[model_name] = test_loss
            accuracy_eval[model_name] = accuracy

            if logits_active:
                logits_dataset_dir = os.path.join(logits_dir, dataset_name, model_name)
                os.makedirs(logits_dataset_dir, exist_ok=True)
                np.savetxt(
                    os.path.join(logits_dataset_dir, f"logits_{dataset_name}_{model_name}.txt"),
                    logits_nn,
                    fmt="%.6f",
                )

            del model, train_loader, test_loader, optimizer
            gc.collect()
            torch.cuda.empty_cache()

    return loss_eval, accuracy_eval

# Example config shape:
# configs["train_dataset"] = "train.csv"
# configs["test_dataset"] = "test.csv"

def evaluate_train_test():
    configs_path, input_output_folder, seed = cli()
    data_path = os.path.join(input_output_folder, "data")
    processed_path = os.path.join(data_path, "processed")

    with open(configs_path) as f:
        configs = yaml.safe_load(f)


    logits_active = True
    save_selected_features = True

    model_configs = configs["models"]
    models = list(model_configs.keys())
    n_features = configs["n_features"]
    # Base directory for all results
    if n_features == 10000:
        n_features_str = "10k"
    else:
        n_features_str = n_features

    train_dataset = configs["train_dataset"]
    test_dataset = configs["test_dataset"]

    map_names = {
    "GSE39004_trasp_mod.csv": "GSE39004",
    "GSE86374_trasp_mod.csv": "GSE86374",
    }
    train_name = map_names.get(os.path.basename(train_dataset), os.path.splitext(os.path.basename(train_dataset))[0])
    test_name = map_names.get(os.path.basename(test_dataset), os.path.splitext(os.path.basename(test_dataset))[0])

    results_dir = os.path.join(input_output_folder, "results", f"Train_{train_name}_Test_{test_name}_f{n_features_str}_s{seed}")
    os.makedirs(results_dir, exist_ok=True)
    abs_path = os.path.abspath(results_dir)

    # Directory to store model accuracy logs
    acc_path = os.path.join(abs_path, f"accuracies.log")

    # Define the path for the metrics log
    metrics_path = os.path.join(abs_path, f"metrics.log")

    paths = {"abs_path": abs_path, "acc_path": acc_path, "metrics_path": metrics_path, "data_path": data_path}

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(processed_path, train_dataset)
    test_path = os.path.join(processed_path, test_dataset)

    log_to_file(f"Train dataset: {train_name}", acc_path)
    log_to_file(f"Test dataset: {test_name}", acc_path)


    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    y_train = np.array(train_df.iloc[:, 0], dtype=np.int32)
    X_train = np.array(train_df.drop(columns=0), dtype=np.float32)

    y_test = np.array(test_df.iloc[:, 0], dtype=np.int32)
    X_test = np.array(test_df.drop(columns=0), dtype=np.float32)

    num_classes = len(torch.unique(torch.tensor(np.concatenate([y_train, y_test]))))

    dataset_name = {
        "GSE_39004_trasp_mod.csv": "GSE39004",
        "GSE866574_trasp_mod.csv": "GSE866574",
    }.get(os.path.basename(test_dataset), os.path.splitext(os.path.basename(test_dataset))[0])

    avg_loss, avg_accuracy = evaluate_models_traintest(
        models=models,
        configs=configs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        dataset_name=dataset_name,
        device=device,
        logits_active=logits_active,
        save_selected_features=save_selected_features,
        paths=paths,
        seed=seed,
    )   

    accuracies = {}
    std_errors = {}
    
    # Store results
    for model_name in models:
        if model_name not in accuracies:
            accuracies[model_name] = []

        accuracies[model_name].append(avg_accuracy[model_name] / 100)

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


evaluate_train_test()