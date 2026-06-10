# import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
)


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

    else:
        raise ValueError(f"Method '{method}' not supported.")

    return X_train_selected, X_test_selected, selected_feature_indices
