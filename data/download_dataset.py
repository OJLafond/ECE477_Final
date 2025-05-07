import os

def download_physionet_dataset(target_dir="data/physionet"):
    os.makedirs(target_dir, exist_ok=True)
    os.system(f"wget -r -N -c -np https://physionet.org/files/noneeg/1.0.0/ -P {target_dir}")
