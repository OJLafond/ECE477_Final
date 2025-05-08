import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# change back to 30 later
def train_labeler(X_train, y_train, X_val=None, y_val=None, max_depth_range=(1, 2), n_estimators=500):
    """Train a random forest and return the best one based on validation accuracy."""
    best_acc = -np.inf
    best_model = None
    best_depth = None

    for depth in range(max_depth_range[0], max_depth_range[1]):
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_depth = depth

    print(f"Best depth: {best_depth}, validation accuracy: {best_acc:.4f}")
    return best_model


def label_synthetic_data(model, X_syn):
    """Use the trained model to label synthetic data."""
    return model.predict(X_syn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train classifier on real data and label synthetic data.")
    parser.add_argument("--real_data", type=str, required=True, help="Path to real data features (joblib .pkl)")
    parser.add_argument("--real_labels", type=str, required=True, help="Path to real data labels (joblib .pkl)")
    parser.add_argument("--synthetic_data", type=str, required=True, help="Path to synthetic feature data (joblib .pkl)")
    parser.add_argument("--output", type=str, required=True, help="Output path for labeled synthetic data (joblib .pkl)")

    args = parser.parse_args()

    # Load real and synthetic data
    X = joblib.load(args.real_data)
    y = joblib.load(args.real_labels)
    X_syn = joblib.load(args.synthetic_data)

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = train_labeler(X_train, y_train, X_val, y_val)

    # Relabel synthetic data
    y_syn = label_synthetic_data(clf, X_syn)

    # Save
    joblib.dump((X_syn, y_syn), args.output)
    print(f"Labeled synthetic data saved to {args.output}")
