import os
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_DIR = "data/physionet/physionet.org/files/noneeg/1.0.0/"

def parse_header_file(header_path):
    metadata = {}
    with open(header_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                key_value = line[2:].split(": ")
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key] = value
            elif len(line.split()) == 4 and line.split()[1].isdigit():
                parts = line.split()
                metadata["Number of Samples"] = int(parts[3])
                metadata["Sampling Frequency"] = float(parts[2])
    return metadata

def load_signal(subject_id):
    rec1 = os.path.join(DATA_DIR, f"Subject{subject_id}_AccTempEDA")
    rec2 = os.path.join(DATA_DIR, f"Subject{subject_id}_SpO2HR")

    meta1 = parse_header_file(rec1 + ".hea")
    meta2 = parse_header_file(rec2 + ".hea")

    sig1, _ = wfdb.rdsamp(rec1)
    sig2, _ = wfdb.rdsamp(rec2)

    df1 = pd.DataFrame(sig1, columns=['ax', 'ay', 'az', 'temp', 'EDA'])
    df2 = pd.DataFrame(sig2, columns=['SpO2', 'HR'])

    fs1 = int(meta1['Number of Samples'])
    fs2 = int(meta2['Number of Samples'])

    repeat_factor = fs1 // fs2
    df2_exp = df2.loc[df2.index.repeat(repeat_factor)].reset_index(drop=True)
    df1 = df1.iloc[:len(df2_exp)]
    df = pd.concat([df1.reset_index(drop=True), df2_exp], axis=1)

    try:
        annotations = wfdb.rdann(rec1, 'atr')
        task_stages = [''] * len(df)
        for i in range(len(annotations.sample) - 1):
            start_idx = annotations.sample[i]
            end_idx = annotations.sample[i + 1]
            label = annotations.aux_note[i]
            task_stages[start_idx:end_idx] = [label] * (end_idx - start_idx)
        if len(annotations.sample) > 0:
            start_idx = annotations.sample[-1]
            task_stages[start_idx:] = [annotations.aux_note[-1]] * (len(df) - start_idx)
        df['task'] = task_stages
    except:
        df['task'] = "Unknown"

    df['subject_id'] = subject_id
    return df

def label_relaxation_stage(df):
    current_subject = None
    relaxation_counter = 0
    last_task = None

    for index, row in df.iterrows():
        if row['subject_id'] != current_subject:
            current_subject = row['subject_id']
            relaxation_counter = 0
            last_task = None

        if row['task'] == 'Relax':
            if last_task != 'Relax':
                relaxation_counter += 1
            df.at[index, 'task'] = f"Relax{relaxation_counter}"

        last_task = row['task']

    return df

def label_combined_stress(df):
    stress = ['CognitiveStress', 'EmotionalStress', 'PhysicalStress']
    relax = ['Relax1', 'Relax2', 'Relax3', 'Relax4']
    df['task_combined'] = df['task'].apply(
        lambda x: 'Stress' if x in stress else ('Relax' if x in relax else 'Unknown')
    )
    return df

def normalize_and_window(df, window_size=10):
    features = ['ax', 'ay', 'az', 'temp', 'EDA', 'SpO2', 'HR']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    id_cols = ['subject_id', 'task_combined']
    df_sorted = df.sort_values(id_cols)

    def downsample(group):
        n = len(group) // window_size * window_size
        group = group.iloc[:n]
        averaged = group[features].values.reshape(-1, window_size, len(features)).mean(axis=1)
        meta = group[id_cols].iloc[::window_size].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(averaged, columns=features)], axis=1)

    return df_sorted.groupby(id_cols, group_keys=False).apply(downsample).reset_index(drop=True)

def split_and_reduce(df, method='rp', test_size=0.2, n_components=5):
    features = ['ax', 'ay', 'az', 'temp', 'EDA', 'SpO2', 'HR']
    X = df[features]
    y = df['task_combined']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'rp':
        reducer = GaussianRandomProjection(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'rp'.")

    X_train_reduced = reducer.fit_transform(X_train_scaled)
    X_test_reduced = reducer.transform(X_test_scaled)

    return X_train_reduced, X_test_reduced, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def preprocess_all_subjects(n_subjects=20, dim_red_method='rp', n_components=5):
    all_subjects = []
    for i in range(1, n_subjects + 1):
        df = load_signal(i)
        all_subjects.append(df)

    full_df = pd.concat(all_subjects, ignore_index=True)
    full_df = label_relaxation_stage(full_df)
    full_df = label_combined_stress(full_df)
    downsampled_df = normalize_and_window(full_df)
    return split_and_reduce(downsampled_df, method=dim_red_method, n_components=n_components)
