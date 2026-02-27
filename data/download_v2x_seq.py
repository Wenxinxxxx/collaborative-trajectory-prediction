"""
V2X-Seq Dataset Download and Setup Guide.

The V2X-Seq dataset is hosted on multiple platforms. This script provides
automated download helpers and manual download instructions.

Dataset paper:
  Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for
  Vehicle-Infrastructure Cooperative Perception and Forecasting," CVPR 2023.

Official repository:
  https://github.com/AIR-THU/DAIR-V2X-Seq

Dataset components needed for trajectory prediction:
  - V2X-Seq-SPD (Sequential Perception Dataset) - trajectory forecasting subset
  - Specifically: cooperative-vehicle-infrastructure/vic-trajectories

Size: ~2-5 GB for the trajectory prediction subset
"""

import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR


def check_git_lfs():
    """Check if git-lfs is installed."""
    try:
        subprocess.run(['git', 'lfs', 'version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_git_lfs():
    """Install git-lfs."""
    print("Installing git-lfs...")
    os.system("sudo apt-get update && sudo apt-get install -y git-lfs")
    os.system("git lfs install")


def download_from_github(target_dir):
    """
    Download V2X-Seq from the official GitHub repository.
    This requires git-lfs for large files.
    """
    repo_url = "https://github.com/AIR-THU/DAIR-V2X-Seq.git"
    print(f"Cloning V2X-Seq repository to {target_dir}...")
    print("Note: This may take a while due to large file sizes.")

    if not check_git_lfs():
        install_git_lfs()

    os.makedirs(target_dir, exist_ok=True)
    cmd = f"git clone --depth 1 {repo_url} {target_dir}/DAIR-V2X-Seq"
    print(f"Running: {cmd}")
    os.system(cmd)


def download_from_opendatalab(target_dir):
    """
    Download from OpenDataLab (alternative source).
    Requires: pip install opendatalab
    """
    try:
        import opendatalab
    except ImportError:
        print("Installing opendatalab...")
        os.system("pip install opendatalab")

    print("Downloading from OpenDataLab...")
    print("You may need to register at https://opendatalab.com/ first.")
    os.makedirs(target_dir, exist_ok=True)
    os.system(f"odl get DAIR-V2X-Seq --target {target_dir}")


def print_manual_instructions():
    """Print manual download instructions."""
    instructions = """
============================================================
V2X-Seq Dataset Manual Download Instructions
============================================================

The V2X-Seq dataset can be downloaded from several sources:

1. OFFICIAL GITHUB REPOSITORY (Recommended):
   URL: https://github.com/AIR-THU/DAIR-V2X-Seq
   
   Steps:
   a) Install git-lfs: sudo apt install git-lfs && git lfs install
   b) Clone: git clone https://github.com/AIR-THU/DAIR-V2X-Seq.git
   c) The trajectory data is in:
      DAIR-V2X-Seq/dataset/v2x-seq-spd/

2. OPENDATALAB:
   URL: https://opendatalab.com/DAIR-V2X-Seq
   
   Steps:
   a) Register at opendatalab.com
   b) pip install opendatalab
   c) odl get DAIR-V2X-Seq

3. BAIDU NETDISK (百度网盘):
   Check the official GitHub README for the latest Baidu Netdisk link.
   This is often the fastest option for users in China.

4. GOOGLE DRIVE:
   Check the official GitHub README for Google Drive links.

============================================================
After downloading, place the data in:
  {data_dir}/v2x_seq/

Expected directory structure:
  {data_dir}/v2x_seq/
  ├── cooperative-vehicle-infrastructure/
  │   ├── infrastructure-side/
  │   │   ├── velodyne/
  │   │   ├── label/
  │   │   └── calib/
  │   ├── vehicle-side/
  │   │   ├── velodyne/
  │   │   ├── label/
  │   │   └── calib/
  │   └── cooperative/
  │       ├── label/
  │       └── data_info.json
  └── vic-trajectories/
      ├── train/
      ├── val/
      └── test/

Then run the preprocessing script:
  python data/preprocess.py --data v2x_seq

============================================================
""".format(data_dir=RAW_DATA_DIR)
    print(instructions)


def verify_dataset(data_dir):
    """Verify that the dataset is properly downloaded."""
    v2x_dir = os.path.join(data_dir, 'v2x_seq')

    if not os.path.exists(v2x_dir):
        print(f"Dataset directory not found: {v2x_dir}")
        return False

    # Check for key files/directories
    expected_paths = [
        'vic-trajectories',
        'cooperative-vehicle-infrastructure',
    ]

    found = 0
    for path in expected_paths:
        full_path = os.path.join(v2x_dir, path)
        if os.path.exists(full_path):
            found += 1
            print(f"  [OK] Found: {path}")
        else:
            print(f"  [MISSING] Not found: {path}")

    if found == len(expected_paths):
        print("\nDataset verification: PASSED")
        return True
    else:
        print(f"\nDataset verification: INCOMPLETE ({found}/{len(expected_paths)})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download V2X-Seq dataset')
    parser.add_argument('--method', type=str, default='instructions',
                        choices=['github', 'opendatalab', 'instructions', 'verify'],
                        help='Download method')
    parser.add_argument('--target', type=str, default=None,
                        help='Target directory (default: data/v2x_seq)')
    args = parser.parse_args()

    target_dir = args.target or os.path.join(RAW_DATA_DIR, 'v2x_seq')

    if args.method == 'instructions':
        print_manual_instructions()
    elif args.method == 'github':
        download_from_github(target_dir)
    elif args.method == 'opendatalab':
        download_from_opendatalab(target_dir)
    elif args.method == 'verify':
        verify_dataset(RAW_DATA_DIR)


if __name__ == '__main__':
    main()
