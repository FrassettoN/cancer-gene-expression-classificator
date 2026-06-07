import os

import numpy as np
import pandas as pd
import torch

# import shap
# import pymrmr

from skrebate import ReliefF
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from models import StackedAutoencoder as ae
from models.autoencoder import filter_features


def compute_snr(X, y):
    classes = np.unique(y)
    snr_scores = []

    for i in range(X.shape[1]):
        means = []
        variances = []
        for cls in classes:
            X_cls = X[y == cls, i]
            means.append(np.mean(X_cls))
            variances.append(np.var(X_cls))

        mean_diff = np.abs(means[0] - means[1]) if len(classes) == 2 else np.std(means)
        avg_variance = np.mean(variances)
        snr = mean_diff / (np.sqrt(avg_variance) + 1e-8)
        snr_scores.append(snr)

    return np.array(snr_scores)


def compute_fisher_score(X, y):
    classes = np.unique(y)
    fisher_scores = []

    for i in range(X.shape[1]):
        means = []
        variances = []
        sizes = []

        for cls in classes:
            X_cls = X[y == cls, i]
            means.append(np.mean(X_cls))
            variances.append(np.var(X_cls))
            sizes.append(len(X_cls))

        overall_mean = np.mean(X[:, i])

        numerator = sum(
            [sizes[j] * (means[j] - overall_mean) ** 2 for j in range(len(classes))]
        )
        denominator = sum([sizes[j] * variances[j] for j in range(len(classes))])

        fisher_score = numerator / (denominator + 1e-8)
        fisher_scores.append(fisher_score)

    return np.array(fisher_scores)


def neighbor_graphs(X_scaled, coords, k=8, sigma=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_scaled = (
        torch.tensor(X_scaled, dtype=torch.float, device=device)
        if not torch.is_tensor(X_scaled)
        else X_scaled.to(device)
    )
    coords = (
        torch.tensor(coords, dtype=torch.float, device=device)
        if not torch.is_tensor(coords)
        else coords.to(device)
    )

    n_samples, n_genes = X_scaled.shape
    if k > n_genes - 1:
        raise ValueError(f"k={k} is too large: it must be at most {n_genes-1}")

    # Intra-pixel adjacency
    pixel_coords = coords.round().long().to(device)
    A_intra = torch.zeros((n_genes, n_genes), dtype=torch.float, device=device)
    for p in torch.unique(pixel_coords, dim=0):
        idx = (pixel_coords == p).all(dim=1).nonzero(as_tuple=True)[0]
        for i in idx:
            for j in idx:
                if i != j:
                    A_intra[i, j] = 1.0

    # Inter-pixel adjacency
    dists = torch.cdist(coords, coords)  # [n_genes, n_genes]
    dists.fill_diagonal_(float("inf"))
    knn_dists, knn_idx = torch.topk(-dists, k, dim=1)  # min distances
    knn_dists = -knn_dists
    A_inter = torch.zeros_like(dists)
    for i in range(n_genes):
        for d, j in zip(knn_dists[i], knn_idx[i]):
            if sigma is None:
                w = 1.0 / (d + 1e-8)
            else:
                w = torch.exp(-(d**2) / (2 * sigma**2))
            A_inter[i, j] = w

    return A_intra, A_inter


def neighborhood_informed_feature_selection(
    X_train, y_train, coords, k=8, sigma=1.0, alpha=1.0, beta=1.0, gamma=0.7
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct graphs
    A_intra, A_inter = neighbor_graphs(X_train, coords, k=k, sigma=sigma)
    A_combined = alpha * A_inter + beta * A_intra

    # Score based on the expression
    score_expr = mutual_info_classif(X_train, y_train)

    # Graph-based score
    score_expr_torch = torch.tensor(score_expr, dtype=torch.float, device=device)
    score_graph = (A_combined @ score_expr_torch).cpu().numpy()

    # Combination
    score_final = gamma * score_expr + (1 - gamma) * score_graph

    return score_final


def feature_selection(
    X_train_scaled,
    X_test_scaled,
    y_train,
    device,
    method="ensemble",
    max_features=10000,
    coords="None",
    seed=None
):

    top_indices = min(
        max_features, X_train_scaled.shape[1]
    )  # Number of top features to select
    selected_feature_indices = None

    if method == "chi2":
        chi2_selector = SelectKBest(
            score_func=chi2, k=top_indices
        )  # Chi-squared feature selection

        X_train_selected = chi2_selector.fit_transform(
            X_train_scaled, y_train
        )  # Fit and transform the training data
        X_test_selected = chi2_selector.transform(
            X_test_scaled
        )  # Transform the testing data

        selected_feature_indices = chi2_selector.get_support(
            indices=True
        )  # Store the indices of the selected features

    elif method == "RfF":
        rf = RandomForestClassifier(
            n_estimators=1000, random_state=seed
        )  # Initialize the RandomForestClassifier
        rf.fit(
            X_train_scaled, y_train
        )  # Fit the RandomForest model to the training data

        feature_importance = (
            rf.feature_importances_
        )  # Calculate the importance of features
        important_indices = np.argsort(feature_importance)[
            ::-1
        ]  # Sort the features in descending order by importance

        X_train_selected = X_train_scaled[
            :, important_indices[:top_indices]
        ]  # Select top_indices features in the training data
        X_test_selected = X_test_scaled[
            :, important_indices[:top_indices]
        ]  # Select top_indices features in the testing data

        selected_feature_indices = important_indices[
            :top_indices
        ]  # Store the indices of the selected features

    # elif method == 'mRMR':
    #     df = pd.DataFrame(X_train_scaled, columns=[f"f{i}" for i in range(X_train_scaled.shape[1])])
    #     df['label'] = y_train

    #     discrete_cols = []
    #     valid_col_names = []

    #     for col in df.columns[:-1]:
    #         try:
    #             discretized = pd.qcut(df[col], q=3, labels=False, duplicates='drop')
    #             discrete_cols.append(discretized)
    #             valid_col_names.append(col)
    #         except ValueError:
    #             continue

    #     df_discretized = pd.concat(discrete_cols, axis=1)
    #     df_discretized.columns = valid_col_names

    #     df_discretized.insert(0, 'label', y_train)

    #     df_discretized = df_discretized.dropna(axis=1).astype(int)
    #     df_discretized = df_discretized.copy()

    #     with open(os.devnull, 'w') as devnull:
    #         original_stdout_fd = os.dup(1)
    #         original_stderr_fd = os.dup(2)

    #         os.dup2(devnull.fileno(), 1)
    #         os.dup2(devnull.fileno(), 2)

    #         try:
    #             selected = pymrmr.mRMR(df_discretized, 'MIQ', top_indices)
    #             indices = [int(f[1:]) for f in selected if f.startswith("f") and f[1:].isdigit()]

    #         finally:
    #             os.dup2(original_stdout_fd, 1)
    #             os.dup2(original_stderr_fd, 2)

    #             os.close(original_stdout_fd)
    #             os.close(original_stderr_fd)

    #     X_train_selected = X_train_scaled[:, indices]
    #     X_test_selected = X_test_scaled[:, indices]

    elif method == "IG":
        selector = SelectKBest(
            score_func=mutual_info_classif, k=top_indices,
        )  # Information Gain
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

    elif method == "ReliefF":
        relieff_selector = ReliefF(n_neighbors=10, n_features_to_select=top_indices)
        relieff_selector.fit(X_train_scaled, y_train)

        sorted_indices = relieff_selector.feature_importances_.argsort()[::-1]
        top_indices = sorted_indices[:max_features]

        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]

    elif method == "autoencoder":
        X_train_selected, X_test_selected = filter_features(
            X_train_scaled, X_test_scaled, device, 30, 10, bottleneck=True
        )

    # elif method == "SHAP":
    #     model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    #     model.fit(X_train_scaled, y_train)

    #     explainer = shap.TreeExplainer(model)
    #     shap_values_train = explainer.shap_values(X_train_scaled)

    #     feature_importance = (
    #         np.abs(shap_values_train).mean(axis=0)
    #         if len(shap_values_train.shape) > 2
    #         else np.abs(shap_values_train).mean(axis=0)
    #     )
    #     top_k_indices = np.argsort(feature_importance)[::-1][:top_indices]

    #     X_train_selected = X_train_scaled[:, top_k_indices]
    #     X_test_selected = X_test_scaled[:, top_k_indices]

    elif method == "RFE":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator, n_features_to_select=top_indices, step=1)

        X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
        X_test_selected = rfe.transform(X_test_scaled)

    elif method == "xgboost":
        xgb = XGBClassifier(eval_metric="logloss")
        xgb.fit(X_train_scaled, y_train)

        importances = xgb.feature_importances_
        indices = np.argsort(importances)[::-1][:top_indices]

        X_train_selected = X_train_scaled[:, indices]
        X_test_selected = X_test_scaled[:, indices]

    elif method == "MLP":
        mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train)

        importances = np.abs(mlp.coefs_[0]).mean(axis=1)
        indices = np.argsort(importances)[::-1][:top_indices]

        X_train_selected = X_train_scaled[:, indices]
        X_test_selected = X_test_scaled[:, indices]

    elif method == "SVC":
        svc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000)
        svc.fit(X_train_scaled, y_train)

        importances = np.abs(svc.coef_).mean(axis=0)
        indices = np.argsort(importances)[::-1][:top_indices]

        X_train_selected = X_train_scaled[:, indices]
        X_test_selected = X_test_scaled[:, indices]

    elif method == "ANOVA":

        # Perform ANOVA F-test for Feature Selection
        selector = SelectKBest(f_classif, k="all")  # Select all features initially
        selector.fit(X_train_scaled, y_train)  # Fit the selector to the training set
        p_values = selector.pvalues_  # Extract the p-values for each feature

        selected_indices = np.where(p_values < 0.07)[0]  # Select features with p-values
        X_train_selected = X_train_scaled[
            :, selected_indices
        ]  # Keep only the selected features in the training set
        X_test_selected = X_test_scaled[
            :, selected_indices
        ]  # Keep only the selected features in the testing set

    elif method == "SNR":
        snr_scores = compute_snr(X_train_scaled, y_train)
        top_indices = np.argsort(snr_scores)[-max_features:]

        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]

    elif method == "Fisher":
        fisher_scores = compute_fisher_score(X_train_scaled, y_train)
        top_indices = np.argsort(fisher_scores)[-max_features:]

        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]

    # Proposed feature selection
    elif method == "NIFS":
        score_final = neighborhood_informed_feature_selection(
            X_train_scaled,
            y_train,
            coords,
            k=8,
            sigma=1.0,
            alpha=1.0,
            beta=1.0,
            gamma=0.5,
        )
        top_indices = np.argsort(score_final)[-max_features:]

        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]

    else:
        raise ValueError(f"Method '{method}' not supported.")

    return X_train_selected, X_test_selected, selected_feature_indices
