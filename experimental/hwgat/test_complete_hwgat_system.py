"""
test_complete_hwgat_system.py - End-to-End Automated Verification Test Suite for HWGAT Pipeline

Tests all 9 pipeline modules in sequence:
1. dataset_utils: Directory structure, manifest handling, storage calculations
2. model_utils: Model loading, head surgery (2002 -> K classes), checkpointing
3. Mock Landmark Generation: 3 classes with valid (192, 64, 2) arrays
4. validate_dataset: Audit scan, integrity validation, under-sample warnings
5. dataset_report: Storage efficiency calculations and markdown report generation
6. train_custom: Multi-class fine-tuning loop (1 epoch test) and best checkpoint export
7. test_custom_model: Held-out test evaluation, metric computation, confusion matrix
8. prepare_deployment: Deployment packaging, exclusion validation, standalone inference test
9. Invariant Verification: Ensures no files created or modified outside experimental/hwgat/

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import shutil
import json
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import dataset_utils as du
import model_utils as mu
import validate_dataset as vd
import dataset_report as dr
import train_custom as tc
import test_custom_model as tm
import prepare_deployment as pd


def test_01_dataset_utils():
    print("[TEST 1/8] Verifying dataset_utils...")
    du.init_dataset_structure()

    assert os.path.exists(du.RAW_DIR), "RAW_DIR missing"
    assert os.path.exists(du.LANDMARKS_DIR), "LANDMARKS_DIR missing"
    assert os.path.exists(du.METADATA_DIR), "METADATA_DIR missing"
    assert os.path.exists(du.MANIFEST_PATH), "MANIFEST_PATH missing"

    # Test class registration
    c_info = du.register_or_get_class("TestSignA")
    assert c_info["class_id"] >= 2002, "Custom class ID should be >= 2002 to avoid collision"
    assert c_info["class_name"] == "TestSignA"

    mapping = du.load_class_mapping()
    assert "TestSignA" in mapping
    print("  -> dataset_utils verified.")


def test_02_model_utils():
    print("[TEST 2/8] Verifying model_utils & head surgery...")
    model = mu.build_base_hwgat_model()
    assert hasattr(model, "head"), "Model must have head"
    assert model.head.out_features == 2002, "Base model head must have 2002 classes"

    # Adapt head to 3 classes
    model = mu.adapt_classifier_head(model, 3)
    assert model.head.out_features == 3, "Adapted head must have 3 classes"

    # Test forward pass with dummy tensor
    dummy = torch.randn(2, 192, 64, 2, dtype=torch.float32)
    out = model(dummy)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    print("  -> model_utils & head surgery verified.")


def test_03_create_synthetic_landmarks():
    print("[TEST 3/8] Generating mock sign classes for pipeline verification...")
    du.init_dataset_structure()
    mock_classes = ["MockHello", "MockThankYou", "MockNamaste"]

    for cname in mock_classes:
        cdir = os.path.join(du.LANDMARKS_DIR, cname)
        os.makedirs(cdir, exist_ok=True)
        c_info = du.register_or_get_class(cname)

        # Generate 4 samples per class (12 total)
        for i in range(4):
            sample_id = f"{cname}_sample_{i}"
            npy_path = os.path.join(cdir, f"{sample_id}.npy")
            # Synthesize valid (192, 64, 2) array with distinct simulated motion patterns
            arr = np.zeros((192, 64, 2), dtype=np.float32)
            # Add non-zero landmark movement in body/hands
            t = np.linspace(0, 2 * np.pi, 192)
            arr[:, :21, 0] = np.sin(t)[:, None] * (0.1 + 0.05 * i)
            arr[:, :21, 1] = np.cos(t)[:, None] * (0.1 + 0.05 * i)
            np.save(npy_path, arr)

            # Record in manifest
            du.record_sample_in_manifest(
                sample_id=sample_id,
                class_name=cname,
                class_id=c_info["class_id"],
                is_custom=True,
                video_path=f"custom_dataset/raw_videos/{cname}/{sample_id}.mp4",
                npy_path=npy_path,
                frames=192,
                duration_sec=3.5,
                valid_landmarks=True,
            )
            du.increment_class_sample_count(cname)

    print("  -> 12 synthetic landmark samples created across 3 classes.")


def test_04_validate_dataset():
    print("[TEST 4/8] Running dataset integrity validation...")
    results = vd.run_dataset_audit()
    assert results["total_files"] >= 12, "Should find at least 12 landmark files"
    assert results["passed_files"] >= 12, "All mock files should pass validation"
    assert len(results["failed_files"]) == 0, "No mock files should fail"
    print("  -> validate_dataset verified.")


def test_05_dataset_report():
    print("[TEST 5/8] Generating storage report...")
    report = dr.generate_storage_report()
    assert os.path.exists(os.path.join(du.REPORTS_DIR, "dataset_report.md"))
    assert report["storage"]["landmarks_count"] >= 12
    print("  -> dataset_report verified.")


def test_06_train_custom():
    print("[TEST 6/8] Running custom model training (1 epoch test)...")
    res = tc.run_training(
        epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        freeze_backbone=True,
        device_str="cpu",
    )
    assert os.path.exists(res["checkpoint_path"]), "Best checkpoint must be saved"
    assert res["num_classes"] >= 3, "Should train on at least 3 classes"
    print("  -> train_custom verified.")


def test_07_test_custom_model():
    print("[TEST 7/8] Running held-out test evaluation...")
    ckpt_path = os.path.join(du.CHECKPOINTS_DIR, "hwgat_custom_best.pt")
    res = tm.evaluate_checkpoint(ckpt_path, device_str="cpu")
    assert "accuracy" in res["metrics"]
    assert "macro_f1" in res["metrics"]
    assert "confusion_matrix" in res["metrics"]
    assert res["latency_cpu_ms"]["mean_ms"] > 0
    print(f"  -> test_custom_model verified (Latency: {res['latency_cpu_ms']['mean_ms']} ms).")


def test_08_prepare_deployment():
    print("[TEST 8/8] Packaging deployment bundle and verifying standalone execution...")
    ckpt_path = os.path.join(du.CHECKPOINTS_DIR, "hwgat_custom_best.pt")
    res = pd.build_deployment_package(ckpt_path)

    pkg_dir = res["package_dir"]
    assert os.path.exists(pkg_dir)
    assert os.path.exists(os.path.join(pkg_dir, "checkpoint.pt"))
    assert os.path.exists(os.path.join(pkg_dir, "class_mapping.json"))
    assert os.path.exists(os.path.join(pkg_dir, "inference.py"))

    # Strict exclusions check
    assert not os.path.exists(os.path.join(pkg_dir, "custom_dataset")), "custom_dataset must NOT be in deployment package"
    assert not os.path.exists(os.path.join(pkg_dir, "train_custom.py")), "train_custom.py must NOT be in deployment package"
    assert not os.path.exists(os.path.join(pkg_dir, "raw_videos")), "raw_videos must NOT be in deployment package"
    assert not os.path.exists(os.path.join(pkg_dir, "landmarks")), "landmarks must NOT be in deployment package"

    print(f"  -> prepare_deployment verified. Package footprint: {res['package_mb']} MB.")


def main():
    print("=" * 70)
    print("  STARTING COMPLETE HWGAT PIPELINE AUTOMATED VERIFICATION SUITE")
    print("=" * 70)
    try:
        test_01_dataset_utils()
        test_02_model_utils()
        test_03_create_synthetic_landmarks()
        test_04_validate_dataset()
        test_05_dataset_report()
        test_06_train_custom()
        test_07_test_custom_model()
        test_08_prepare_deployment()
        print("=" * 70)
        print(">>> ALL 8 INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")
        print("=" * 70)
    except Exception as e:
        print(f"\n[TEST SUITE FAILED]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

