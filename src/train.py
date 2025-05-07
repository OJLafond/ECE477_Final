import os
import torch
import numpy as np
import random
from model import SDNN
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training_scheme(X, y, scheme="A", checkpoint_dir="results", epochs_per_step=10):
    set_seed(0)

    # Training scheme-specific parameters
    params_dict = {
        'A': {'init_size': 20, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.3, 'loops': 10},
        'B': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9, 'loops': 10},
        'C': {'init_size': 100, 'max_size': 150, 'sparse': True, 'sparse_ratio': 0.9, 'loops': 10},
    }

    assert scheme in params_dict, f"Unsupported scheme '{scheme}'"
    params = params_dict[scheme]

    # Prepare output directory
    save_dir = os.path.join(checkpoint_dir, f"scann_{scheme.lower()}")
    os.makedirs(save_dir, exist_ok=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Binary encoding (Stress = 1, Relax = 0)
    y_train = np.where(y_train == 'Stress', 1, 0)
    y_test = np.where(y_test == 'Stress', 1, 0)

    # Initialize model
    model = SDNN(in_num=X.shape[1], out_num=2,
                 init_size=params['init_size'],
                 max_size=params['max_size'],
                 batch_size=256,
                 scheme=scheme)
    model.m1 = model.m1.requires_grad_(False)
    model.m2 = model.m2.requires_grad_(False)
    model.m3 = model.m3.requires_grad_(False)
    model.m4 = model.m4.requires_grad_(False)

    model.load_data(X_train, y_train, X_test, y_test)
    model.m1 = torch.tensor(model.m1)
    model.m3 = torch.tensor(model.m3)
    model.m1 = torch.tensor(model.m1)

    # Initial structure and training
    model.train(epochs_per_step, save_dir)

    for i in range(params['loops']):
        print(f"\n[SCANN] === Loop {i+1}/{params['loops']} ===")

        model.addConnection(mode='grad',
                            percentile={'m2': 70, 'm1': 70, 'm3': 70, 'm4': 70},
                            full_data=False)
        model.train(epochs_per_step, save_dir)

        if scheme == "A":
            model.cellDivision(mode='acti', full_data=False)
            model.train(epochs_per_step, save_dir)

        elif scheme in ["B", "C"]:
            model.pruneConnections(prune_ratio=0.2)
            model.train(epochs_per_step, save_dir)

    print(f"[SCANN] Final accuracy for Scheme {scheme}:")
    acc = model.display_accuracy()

    final_model_path = os.path.join(save_dir, f"SDNN_model_final.pth.tar")
    model.save_checkpoint(final_model_path, is_best=False)

    return model, acc


# Example run (can be turned into argparse or notebook cell)
if __name__ == "__main__":
    from preprocess import load_preprocessed_data  # must return X, y
    X, y = load_preprocessed_data()

    for scheme in ["A", "B", "C"]:
        print(f"\nRunning training for scheme {scheme}")
        run_training_scheme(X, y, scheme=scheme)
