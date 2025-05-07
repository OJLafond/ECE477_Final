Stress Detection from Wearable Signals Using SDNN + SCANN

This project implements a real-time stress detection pipeline using physiological signals from wearable medical sensors. It integrates signal preprocessing, synthetic data generation (KDE and GMM), dimensionality reduction, and an evolving sparse dynamic neural network (SDNN) optimized via the SCANN framework.

Project Structure

.
├── data/                   # Raw dataset files (downloaded from PhysioNet)
├── src/
│   ├── preprocess.py       # Data preprocessing, normalization, windowing
│   ├── generate_synthetic.py  # KDE & GMM-based data generation
│   ├── label_synthetic.py  # Label synthetic data using Random Forest
│   ├── model.py            # Sparse Dynamic Neural Network with SCANN
│   ├── train.py            # Pretraining + fine-tuning loops for SDNN
│   ├── evaluate.py         # Metrics, ROC, PR, confusion matrix visualizations
│   └── download_dataset.py # Script to download the PhysioNet dataset
├── results/                # Saved model checkpoints, logs
├── figures/                # Generated plots and visualizations
├── requirements.txt        # Python package dependencies
└── README.md               # You're here!

Dataset

This project uses the Non-EEG Physiological Signals for Stress Detection Dataset from PhysioNet.

To download the dataset, run:

python src/download_dataset.py

This downloads all subject files into data/physionet/.

Setup

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Then install the dependencies:

pip install -r requirements.txt

How to Run

Preprocess the data

from src.preprocess import preprocess_all_subjects
X_train, X_test, y_train, y_test = preprocess_all_subjects(n_subjects=20, dim_red_method='rp', n_components=5)

Generate synthetic data

from src.generate_synthetic import sample_generation
X_syn = sample_generation(X_train, X_test, method='KDE')  # or 'GMM'

Label the synthetic data

from src.label_synthetic import label_synthetic
y_syn = label_synthetic(X_train, y_train, X_syn)

Train SDNN with SCANN

from src.train import run_training_pipeline
run_training_pipeline(X_syn, y_syn, X_test, y_test)

Evaluate and visualize results

from src.evaluate import generate_all_figures
generate_all_figures()

Dependencies

The main Python libraries used are:

wfdb

numpy

pandas

scikit-learn

matplotlib

seaborn

torch

Install all at once with:

pip install -r requirements.txt

Outputs

Trained models: results/

Visualizations (ROC, confusion matrices, etc.): figures/

Evaluation metrics: printed to console and optionally saved



