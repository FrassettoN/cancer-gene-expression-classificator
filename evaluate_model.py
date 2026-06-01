import gc
import os
import numpy as np
import torch
import torch.nn as nn

from types import SimpleNamespace
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from libsvm.svmutil import svm_problem, svm_parameter, svm_train, svm_predict
from torch.utils.data import DataLoader, TensorDataset

from models import KAN, FourierKAN, MLP, CNN_1D, CNN, VisionTransformer
from utils import (
    channel_expansion,
    neighbor_informed_gene_expression,
)
from utils.logger import log_to_file, suppress_output, restore_output
from process_fold import process_fold


def training_loop(n_epochs, optimizer, model, criterion, train_loader, device):

    losses = []  # List of loss values

    model.train()  # Model to training mode

    for epoch in range(n_epochs):

        loss_train = 0.0

        for features, labels in train_loader:

            # Move the features and labels to the GPU
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Training loss for each minibatch
            loss_train += loss.item()

        losses.append(loss_train / len(train_loader))

    return losses


def testing_loop(model, criterion, test_loader, device):

    model.eval()  # Model to evaluation mode

    logits_list = []

    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0

        test_loss = 0.0

        for features, labels in test_loader:

            # Move the features and labels to the GPU
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Testing loss and logits for each minibatch
            test_loss += loss.item()
            logits_list.append(outputs.cpu().numpy())

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / total

    return test_loss, accuracy, np.vstack(logits_list)


def load_model(
    model_name,
    input_dim,
    num_classes,
):
    if model_name == "KAN":
        model = KAN(
            layers_hidden=[input_dim, 100, num_classes],
            grid_size=9,
            spline_order=3,
        )
    elif model_name == "FourierKAN":
        model = FourierKAN(
            layers=[input_dim, 100, num_classes],
            grid_size=9,
            spline_order=3,
        )
    elif model_name == "MLP":
        model = MLP(input_dim=input_dim, hidden_dim=100, output_dim=num_classes)

    elif model_name == "CNN-1D":
        model = CNN_1D(input_dim=input_dim, num_classes=num_classes)

    elif model_name == "DI-CNN+":
        model = CNN(
            in_channels=input_dim, num_classes=num_classes
        )  # DeepInsight-CNN (multi-channel) [DI-CNN+]

    elif model_name == "ViT":
        model = VisionTransformer(
            patch_size=8,  # Size of each patch (patch_size x patch_size) that the image is split into
            image_size=64,  # Size of the input images (image_size x image_size pixels)
            C=input_dim,  # Number of input channels
            num_layers=4,  # Number of transformer layers in the model
            embedding_dim=2096,  # Dimensionality of the embedding space for each patch
            num_heads=8,  # Number of attention heads in the multi-head attention mechanism
            hidden_dim=2096,  # Dimension of the hidden layer
            dropout_prob=0.1,  # Probability of dropout to prevent overfitting
            num_classes=num_classes,
        )  # Number of classes

    elif model_name == "IT-KAN":
        model = KAN(
            layers_hidden=[input_dim, 100, num_classes],
            grid_size=9,
            spline_order=3,
        )

    else:
        raise ValueError(f"Model '{model_name}' is not supported")

    return model


def init_dicts(models, dicts):
    # For each model, create an empty entry in each dictionary
    for model_name in models:
        for d in dicts:
            d[model_name] = []


def evaluate_models_cv(
    models,
    model_configs,
    features,
    labels,
    num_classes,
    dataset_name,
    evaluation_type,
    device,
    logits_active,
    save_selected_features,
    paths,
):

    # Extract paths
    abs_path = paths["abs_path"]
    acc_path = paths["acc_path"]

    # Initialize dictionaries to store per-model metrics
    loss_eval = {}
    accuracy_eval = {}
    avg_loss = {}
    avg_accuracy = {}
    sem = {}

    splits = []

    # Initialize all metric dictionaries for the given models
    dicts = [loss_eval, accuracy_eval, avg_loss, avg_accuracy, sem]
    init_dicts(models, dicts)

    # Generate data splits based on evaluation type
    if evaluation_type == "loo":

        loo = LeaveOneOut()

        splits_dir = os.path.join("data", "splits", "loo")

        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

        splits_filename = os.path.join(splits_dir, f"splits_loo_{dataset_name}.npy")

        # Create and save splits if they do not exist
        if not os.path.exists(splits_filename):
            for train_index, test_index in loo.split(features):
                splits.append((train_index, test_index))

            np.save(splits_filename, np.array(splits, dtype=object))

        splits_array = np.load(splits_filename, allow_pickle=True)

    elif evaluation_type == "kfold":

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        splits_dir = os.path.join("data", "splits", "kfold")

        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

        splits_filename = os.path.join(splits_dir, f"splits_kfold_{dataset_name}.npy")

        # Create and save splits if they do not exist
        if not os.path.exists(splits_filename):

            for train_index, test_index in kfold.split(features, labels):

                splits.append((train_index, test_index))

            np.save(splits_filename, np.array(splits, dtype=object))

        splits_array = np.load(splits_filename, allow_pickle=True)

    else:
        # Raise error if evaluation type is not supported
        raise ValueError(f"Type {evaluation_type} is not supported")

    # Create directories to store logits, including a timestamped subfolder
    if logits_active:
        logits_dir = os.path.join(abs_path, "logits")
        os.makedirs(logits_dir, exist_ok=True)

    # Create directory to save selected features if feature selection is used
    if save_selected_features:
        features_dir = os.path.join(abs_path, "selected_features")
        os.makedirs(features_dir, exist_ok=True)

    # Process each fold in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_fold)(train_idx, test_idx, features, labels, device)
        for train_idx, test_idx in splits_array
    )

    for model_name in models:

        log_to_file(f"\nModel: {model_name}", acc_path)

        # Parameters
        config_dict = model_configs.get(model_name)
        if config_dict is None:
            raise ValueError(f"Model '{model_name}' is not supported")

        # convenient attribute access (safe)
        config = SimpleNamespace(**config_dict)

        batch_size = getattr(config, "batch_size", None)
        num_epochs = getattr(config, "num_epochs", None)
        learningRate = getattr(config, "learningRate", None)
        alpha = getattr(config, "alpha", None)
        beta = getattr(config, "beta", None)
        sigma = getattr(config, "sigma", None)
        k = getattr(config, "k", None)

        for fold in range(len(splits_array)):

            # Define save paths for logits and test labels
            logits_save_path = os.path.join(
                logits_dir, f"logits_{dataset_name}_{fold + 1}_{model_name}.txt"
            )
            ytest_save_path = os.path.join(
                logits_dir, f"ytest_{dataset_name}_{fold + 1}.txt"
            )
            features_save_path = os.path.join(
                features_dir, f"selected_features_{dataset_name}_{fold + 1}.txt"
            )

            # Unpack current fold data
            (
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                it,
                selected_feature_indices,
            ) = results[fold]
            num_features = X_train_scaled.shape[1]

            # Save test labels if logits logging is enabled
            if logits_active:
                np.savetxt(ytest_save_path, np.array(y_test), fmt="%d")

            # Save selected feature indices if available
            if selected_feature_indices is not None and save_selected_features:
                np.savetxt(
                    features_save_path, np.array(selected_feature_indices), fmt="%d"
                )

            if model_name in ["DI-CNN+", "ViT", "IT-KAN"]:

                # Improved DeepInsight with channel wise expansion (apply the channel expansion function to both the training and test sets)
                if model_name in ["DI-CNN+", "ViT"]:
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

                if model_name == "ViT":

                    # Rearrange the dimensions of the tensors: (batch_size, height, width, channels) to (batch_size, channels, height, width)
                    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
                    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)

                # Creating DataSets
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            # Creating DataLoaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            if model_name == "SVM":

                # Train SVM model with probability estimates
                prob = svm_problem(y_train, X_train_scaled)
                param = svm_parameter("-t 0 -q -b 1")
                svm_model = svm_train(prob, param)

                # Suppress output during prediction
                suppress_output()

                # Predict with SVM and get probabilities (logits)
                y_pred_svm, _, logits_svm = svm_predict(
                    y_test, X_test_scaled, svm_model, "-b 1"
                )

                # Restore normal output
                restore_output()

                # Ensure logits are ordered as [0,1]
                logits_svm = np.array(logits_svm)
                labels = np.array(svm_model.get_labels())
                logits_fixed = logits_svm[
                    :, [np.where(labels == 0)[0][0], np.where(labels == 1)[0][0]]
                ]

                # Save logits if logging is enabled
                if logits_active:
                    np.savetxt(logits_save_path, logits_fixed, fmt="%.6f")

                # Compute accuracy
                correct = (y_pred_svm == y_test).sum()
                accuracy = (correct / len(y_test)) * 100
                accuracy_eval[model_name].append(accuracy)

                # Compute hinge loss
                hinge_loss = sum(
                    max(0, 1 - y * y_pred) for y, y_pred in zip(y_test, y_pred_svm)
                ) / len(y_test)
                loss_eval[model_name].append(hinge_loss)

                # Log test accuracy for current fold
                log_to_file(
                    f"Test accuracy {evaluation_type} ({fold + 1}/{len(splits_array)}): {accuracy:.2f}%",
                    acc_path,
                )

            else:

                # Load model
                input_dim = num_features
                if model_name == "ViT":
                    input_dim = X_train_tensor.shape[1]
                elif model_name == "DI-CNN+":
                    input_dim = X_train_scaled.shape[1]

                model = load_model(
                    model_name,
                    input_dim,
                    num_classes,
                )
                # Move the model to the GPU if available
                model.to(device)

                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()

                optimizer_type = getattr(config, "optimizer", None)
                weight_decay = getattr(config, "weight_decay", 0.01)
                if optimizer_type == "Adam":
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=learningRate, weight_decay=weight_decay
                    )
                elif optimizer_type == "AdamW":
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=learningRate, weight_decay=weight_decay
                    )
                else:
                    raise ValueError(f"Optimizer for '{model_name}' not specified")

                # Training model
                _ = training_loop(
                    num_epochs, optimizer, model, criterion, train_loader, device
                )

                # Testing model
                test_loss, accuracy, logits_nn = testing_loop(
                    model, criterion, test_loader, device
                )
                loss_eval[model_name].append(test_loss)
                accuracy_eval[model_name].append(accuracy)

                # Save logits if logging is enabled
                if logits_active:
                    np.savetxt(logits_save_path, np.array(logits_nn), fmt="%.6f")

                # Log test accuracy for the current fold
                log_to_file(
                    f"Test accuracy {evaluation_type} ({fold + 1}/{len(splits_array)}): {accuracy:.2f}%",
                    acc_path,
                )

                # Delete the model, training data loader and optimizer to free up memory
                del model
                del train_loader
                del optimizer

                # Perform garbage collection to clean up any unused memory
                gc.collect()

                # Clear the GPU cache to free up memory on the GPU
                torch.cuda.empty_cache()

        # Compute average accuracy for the model
        avg_accuracy[model_name] = sum(accuracy_eval[model_name]) / len(
            accuracy_eval[model_name]
        )

        # Compute standard error of the mean (SEM) for accuracy
        std_dev = np.std(accuracy_eval[model_name], ddof=1) / 100
        sem[model_name] = std_dev / np.sqrt(len(accuracy_eval[model_name]))

        # Compute average loss for the model
        avg_loss[model_name] = sum(loss_eval[model_name]) / len(loss_eval[model_name])

    return avg_loss, avg_accuracy, sem
