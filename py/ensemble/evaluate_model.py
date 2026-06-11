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
from utils.seed import set_seed

from models import (
    BaseKAN,
    EfficientKAN,
    FourierKAN,
    MLP,
    CNN_1D,
    CNN,
    VisionTransformer,
)
from utils import (
    channel_expansion,
    neighbor_informed_gene_expression,
)
from utils.logger import log_to_file, suppress_output, restore_output
from utils.process_fold import process_fold


def load_model(model_name, input_dim, num_classes, config):
    if model_name == "MLP":
        hidden_dim = getattr(config, "hidden_dim", 100)
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    elif model_name == "BaseKAN":
        hidden_dim = getattr(config, "hidden_dim", 32)
        model = BaseKAN(layers=[input_dim, hidden_dim, num_classes])
    elif model_name == "EfficientKAN":
        hidden_dim = getattr(config, "hidden_dim", 32)
        model = EfficientKAN(layers=[input_dim, hidden_dim, num_classes])
    elif model_name == "FourierKAN":
        hidden_dim = getattr(config, "hidden_dim", 32)
        model = FourierKAN(layers=[input_dim, hidden_dim, num_classes])
    elif model_name == "CNN-1D":
        model = CNN_1D(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "DI-CNN+":
        model = CNN(
            in_channels=input_dim, num_classes=num_classes
        )  # DeepInsight-CNN (multi-channel) [DI-CNN+]
    elif model_name == "IT-KAN":
        hidden_dim = getattr(config, "hidden_dim", 100)
        grid_size = getattr(config, "grid_size", 9)
        spline_order = getattr(config, "spline_order", 3)
        model = BaseKAN(
            layers=[input_dim, hidden_dim, num_classes],
            grid_size=grid_size,
            spline_order=spline_order,
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported")

    return model


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
            if model == "EfficientKAN":
                loss = loss + 1e-4 * model.regularization_loss()

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


def init_dicts(models, dicts):
    # For each model, create an empty entry in each dictionary
    for model_name in models:
        for d in dicts:
            d[model_name] = []


def evaluate_models_cv(
    models,
    configs,
    features,
    labels,
    num_classes,
    dataset_name,
    evaluation_type,
    device,
    logits_active,
    save_selected_features,
    paths,
    seed,
):
    n_features = configs["n_features"]
    model_configs = configs["models"]
    fs_seed = seed
    classificator_seed = seed

    # Extract paths
    abs_path = paths["abs_path"]
    acc_path = paths["acc_path"]
    data_path = paths["data_path"]

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
        splits_dir = os.path.join(data_path, "splits", "loo")
        splits_filename = os.path.join(splits_dir, f"splits_loo_{dataset_name}.npy")
        if not os.path.exists(splits_filename):
            print("Loo splits don't exist - Saving them")

            loo = LeaveOneOut()

            if not os.path.exists(splits_dir):
                os.makedirs(splits_dir)

            # Create and save splits if they do not exist
            for train_index, test_index in loo.split(features):
                splits.append((train_index, test_index))

            np.save(splits_filename, np.array(splits, dtype=object))

        splits_array = np.load(splits_filename, allow_pickle=True)

    elif evaluation_type == "kfold":
        splits_dir = os.path.join(data_path, "splits", "kfold")
        splits_filename = os.path.join(splits_dir, f"splits_kfold_{dataset_name}.npy")
        
        # Create and save splits if they do not exist
        if not os.path.exists(splits_filename):
            print("Kfold splits don't exist - Saving them")

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            if not os.path.exists(splits_dir):
                os.makedirs(splits_dir)


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

    log_to_file(f"FS_seed: {fs_seed}", acc_path)
    # Process each fold in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_fold)(n_features, train_idx, test_idx, features, labels, device, fs_seed)
        for train_idx, test_idx in splits_array
    )

    for model_name in models:
        set_seed(classificator_seed)
        log_to_file(f"Classificator_seed: {classificator_seed}", acc_path)
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

        logits_dataset_dir = os.path.join(logits_dir, dataset_name)
        ytest_dataset_dir = os.path.join(logits_dataset_dir, "ytest")
        os.makedirs(ytest_dataset_dir, exist_ok=True)
        logits_dataset_model_dir = os.path.join(logits_dataset_dir, model_name)
        os.makedirs(logits_dataset_model_dir, exist_ok=True)

        for fold in range(len(splits_array)):

            # Define save paths for logits and test labels
            ytest_save_path = os.path.join(
                ytest_dataset_dir, f"ytest_{dataset_name}_{fold + 1}.txt"
            )
            logits_save_path = os.path.join(
                logits_dataset_model_dir,
                f"logits_{dataset_name}_{fold + 1}_{model_name}.txt",
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

                if model_name == "DI-CNN+":
                    input_dim = X_train_scaled.shape[1]

                model = load_model(model_name, input_dim, num_classes, config)
                # Move the model to the GPU if available
                model.to(device)

                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()

                optimizer_type = getattr(config, "optimizer", None)
                weight_decay = getattr(config, "weight_decay", 0)
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
