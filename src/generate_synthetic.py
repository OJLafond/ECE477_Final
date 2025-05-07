import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def generate_kde_samples(X_train, X_val, n_samples=200000, bw_list=[0.5, 0.7, 1, 1.5, 2, 3]):
    best_bw = None
    best_score = -np.inf
    for bw in bw_list:
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(X_train)
        score = kde.score(X_val)
        if score > best_score:
            best_score = score
            best_bw = bw

    kde = KernelDensity(kernel='gaussian', bandwidth=best_bw)
    kde.fit(X_train)
    X_syn = kde.sample(n_samples=n_samples, random_state=RANDOM_STATE)
    return X_syn


def generate_gmm_samples(X_train, X_val, n_samples=200000, max_components=5):
    best_n = 1
    best_score = -np.inf

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=RANDOM_STATE)
        gmm.fit(X_train)
        score = gmm.score(X_val)
        if score > best_score:
            best_score = score
            best_n = n

    gmm = GaussianMixture(n_components=best_n, covariance_type='full', random_state=RANDOM_STATE)
    gmm.fit(X_train)
    X_syn, _ = gmm.sample(n_samples)
    return X_syn


def generate_synthetic_data(X, method="KDE", validation_ratio=0.2):
    X_train, X_val = train_test_split(X, test_size=validation_ratio, random_state=RANDOM_STATE)

    if method == "KDE":
        return generate_kde_samples(X_train, X_val)
    elif method == "GMM":
        return generate_gmm_samples(X_train, X_val)
    else:
        raise ValueError("Unsupported method. Choose 'KDE' or 'GMM'.")


if __name__ == "__main__":
    import argparse
    import joblib
    import os

    parser = argparse.ArgumentParser(description="Generate synthetic physiological data.")
    parser.add_argument("--input", type=str, required=True, help="Path to scaled training data (joblib .pkl)")
    parser.add_argument("--method", type=str, choices=["KDE", "GMM"], default="KDE", help="Generation method")
    parser.add_argument("--output", type=str, required=True, help="Output file for synthetic data (joblib .pkl)")
    args = parser.parse_args()

    X_train = joblib.load(args.input)
    X_syn = generate_synthetic_data(X_train, method=args.method)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(X_syn, args.output)
    print(f"Synthetic data saved to {args.output}")
