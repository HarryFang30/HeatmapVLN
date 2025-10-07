#!/usr/bin/env python3
"""
Auto Pipeline: Wait for data generation, then run packing and training
"""
import time
import subprocess
import sys
from pathlib import Path
from glob import glob

def count_clips(split):
    """Count generated clips"""
    pattern = f"./raw_sequences/{split}/Room*/clip_*"
    return len(glob(pattern))

def run_command(cmd, desc):
    """Run command and print status"""
    print(f"\n{'='*60}")
    print(f"🚀 {desc}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, cwd="/home/VLN/Project")

    if result.returncode != 0:
        print(f"❌ Failed: {desc}")
        return False
    else:
        print(f"✅ Success: {desc}")
        return True

def main():
    print("🤖 Auto Pipeline Starting...")
    print("Waiting for data generation to complete...")

    # Wait for training data (30 clips)
    while True:
        train_count = count_clips("train")
        print(f"[{time.strftime('%H:%M:%S')}] Train clips: {train_count}/30", end='\r')

        if train_count >= 30:
            print(f"\n✅ Training data complete: {train_count} clips")
            break

        time.sleep(30)

    # Wait a bit for last clip to finish writing
    time.sleep(5)

    # Start validation data generation
    print("\n🚀 Starting validation data generation...")
    cmd = ("python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomB "
           "--clips 10 --T 8 --W 384 --H 384 --pose_mode face_target --path_mode short_arc "
           "--arc_deg 20 --radius 1.8 --render_mode mesh --mesh_grid 28 --split val")

    subprocess.Popen(cmd, shell=True, cwd="/home/VLN/Project",
                     stdout=open("/tmp/gen_val_auto.log", "w"),
                     stderr=subprocess.STDOUT)

    # Wait for validation data
    time.sleep(10)  # Give it time to start
    while True:
        val_count = count_clips("val")
        print(f"[{time.strftime('%H:%M:%S')}] Val clips: {val_count}/10", end='\r')

        if val_count >= 10:
            print(f"\n✅ Validation data complete: {val_count} clips")
            break

        time.sleep(30)

    time.sleep(5)

    # Pack datasets
    print("\n" + "="*60)
    print("📦 Packing Datasets")
    print("="*60)

    if not run_command("python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train",
                      "Pack training data"):
        return 1

    if not run_command("python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val",
                      "Pack validation data"):
        return 1

    # Quality reports
    print("\n" + "="*60)
    print("📊 Quality Reports")
    print("="*60)

    run_command("python scripts/report_data_quality.py --root ./data/habitat_vln --split train",
               "Training data quality report")

    run_command("python scripts/report_data_quality.py --root ./data/habitat_vln --split val",
               "Validation data quality report")

    print("\n" + "="*60)
    print("🎉 Data Pipeline Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review quality reports in outputs/reports/")
    print("2. Run overfit test: python scripts/overfit_one_batch.py")
    print("3. Run training: CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml")

    return 0

if __name__ == "__main__":
    sys.exit(main())
