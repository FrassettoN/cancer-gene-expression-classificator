from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from pyDeepInsight import ImageTransformer

from utils import (
    feature_selection,
)


def process_fold(train_index, test_index, features, labels, device):

    # Split the data into training and test sets
    X_train = features[train_index, :]  # Training features
    X_test = features[test_index, :]  # Test features
    y_train = labels[train_index]  # Training labels
    y_test = labels[test_index]  # Test label

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler on the training features and apply the transformation
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same transformation to the test features (without refitting)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    X_train_scaled_chi2, X_test_scaled_chi2, selected_feature_indices_chi2 = (
        feature_selection(
            X_train_scaled,
            X_test_scaled,
            y_train,
            device,
            method="chi2",
            max_features=10000,
        )
    )
    X_train_scaled, X_test_scaled, selected_feature_indices_rff = feature_selection(
        X_train_scaled_chi2,
        X_test_scaled_chi2,
        y_train,
        device,
        method="RfF",
        max_features=30,
    )

    # Combine selected feature indices from both methods
    selected_feature_indices = selected_feature_indices_chi2[
        selected_feature_indices_rff
    ]

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
