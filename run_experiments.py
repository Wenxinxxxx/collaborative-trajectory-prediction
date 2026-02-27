"""
One-click script to run the complete experiment pipeline.

This script executes all steps in order:
  1. Preprocess V2X-Seq data
  2. Train all 7 models
  3. Evaluate all models
  4. Generate visualizations

Usage:
    python run_experiments.py              # full pipeline (50 epochs)
    python run_experiments.py --quick      # quick test with 5 epochs
    python run_experiments.py --skip_train # skip training, only evaluate
    python run_experiments.py --model v2x_graph  # train only one model
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import create_dirs, print_config

ALL_MODELS = [
    'lstm_seq2seq',
    'social_lstm',
    'grip_plus',
    'transformer',
    'v2x_graph',
    'co_mtp',
    'enhanced_co_mtp',
]


def run_step(description, command):
    """Run a step and report timing."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    start = time.time()
    ret = os.system(command)
    elapsed = time.time() - start
    status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
    print(f"\n[{status}] {description} ({elapsed:.1f}s)")
    return ret == 0


def main():
    parser = argparse.ArgumentParser(description='Run complete experiment pipeline')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 5 epochs')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training, only evaluate and visualize')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip preprocessing (use existing processed data)')
    parser.add_argument('--model', type=str, default='all',
                        choices=ALL_MODELS + ['all'],
                        help='Which model to train')
    args = parser.parse_args()

    create_dirs()
    print_config()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    total_start = time.time()
    steps_passed = 0
    steps_total = 0

    # Step 1: Preprocess V2X-Seq data
    if not args.skip_preprocess:
        steps_total += 1
        if run_step("Preprocess V2X-Seq Data",
                    f"cd {project_dir} && python data/preprocess.py --data v2x_seq --mode cooperative"):
            steps_passed += 1

    # Step 2: Train models
    if not args.skip_train:
        epochs_flag = "--epochs 5" if args.quick else ""
        models_to_train = ALL_MODELS if args.model == 'all' else [args.model]

        for model_name in models_to_train:
            steps_total += 1
            if run_step(f"Train {model_name}",
                        f"cd {project_dir} && python train.py --model {model_name} {epochs_flag}"):
                steps_passed += 1

    # Step 3: Evaluate models
    steps_total += 1
    if run_step("Evaluate Models",
                f"cd {project_dir} && python evaluate.py --model {args.model}"):
        steps_passed += 1

    # Step 4: Generate visualizations
    steps_total += 1
    if run_step("Generate Visualizations",
                f"cd {project_dir} && python visualize.py"):
        steps_passed += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Steps passed: {steps_passed}/{steps_total}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"\nResults directory: {os.path.join(project_dir, 'results')}")
    print(f"Visualizations:   {os.path.join(project_dir, 'visualizations')}")
    print(f"Checkpoints:      {os.path.join(project_dir, 'checkpoints')}")


if __name__ == '__main__':
    main()
