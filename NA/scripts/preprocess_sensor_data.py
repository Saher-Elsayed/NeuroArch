"""
Preprocess raw sensor logs into normalised numpy arrays for SNN training.
Applies: outlier removal (IQR), min-max normalisation, sliding-window segmentation.

Usage:
    python scripts/preprocess_sensor_data.py --building medium_office --window 100
Output:
    data/sensor_logs/<building>/X_train.npy  (N, 14)
    data/sensor_logs/<building>/y_train.npy  (N,)
    data/sensor_logs/<building>/X_val.npy
    data/sensor_logs/<building>/y_val.npy
    data/sensor_logs/<building>/X_test.npy
    data/sensor_logs/<building>/y_test.npy
    data/sensor_logs/<building>/norm_stats.json
"""
import argparse, csv, json, os
import numpy as np
from pathlib import Path

SENSOR_COLS = [
    'T_drybulb_C','RH_pct','CO2_ppm','illuminance_lux','occupancy',
    'window_open','blind_angle_east','blind_angle_west','hvac_valve_pos',
    'air_velocity_ms','globe_T_C','mean_rad_T_C','operative_T_C','T_drybulb_C'
]
BUILDINGS = ['medium_office','residential','mixed_use']


def iqr_filter(X, factor=3.0):
    Q1, Q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all((X >= Q1 - factor*IQR) & (X <= Q3 + factor*IQR), axis=1)
    removed = X.shape[0] - mask.sum()
    if removed > 0:
        print(f"  Removed {removed} outlier rows ({100*removed/X.shape[0]:.1f}%)")
    return X[mask], mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--building', default='medium_office', choices=BUILDINGS+['all'])
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    buildings = BUILDINGS if args.building == 'all' else [args.building]

    for bld in buildings:
        print(f"\nPreprocessing: {bld}")
        data_dir = Path(f"data/sensor_logs/{bld}")

        # Load features CSV
        feat_file = data_dir / "features_train.csv"
        if not feat_file.exists():
            print(f"  [WARN] {feat_file} not found — skipping"); continue

        def load_split(split):
            p = data_dir / f"features_{split}.csv"
            if not p.exists(): return None, None
            rows = list(csv.DictReader(open(p)))
            X = np.array([[float(r[f"feat_{j:02d}"]) for j in range(14)] for r in rows])
            y = np.array([int(r["comfort_class"]) for r in rows])
            return X, y

        X_tr, y_tr = load_split("train")
        X_va, y_va = load_split("val")
        X_te, y_te = load_split("test")
        if X_tr is None: continue

        # IQR filtering on training set
        X_tr, mask = iqr_filter(X_tr)
        y_tr = y_tr[mask]

        # Normalisation stats from training set
        mins = X_tr.min(axis=0)
        maxs = X_tr.max(axis=0)
        stats = {"mins": mins.tolist(), "maxs": maxs.tolist(), "building": bld}

        def norm(X):
            return (X - mins) / (maxs - mins + 1e-8)

        X_tr = norm(X_tr); X_va = norm(X_va); X_te = norm(X_te)

        # Save
        np.save(data_dir / "X_train.npy", X_tr.astype(np.float32))
        np.save(data_dir / "y_train.npy", y_tr.astype(np.int64))
        np.save(data_dir / "X_val.npy",   X_va.astype(np.float32))
        np.save(data_dir / "y_val.npy",   y_va.astype(np.int64))
        np.save(data_dir / "X_test.npy",  X_te.astype(np.float32))
        np.save(data_dir / "y_test.npy",  y_te.astype(np.int64))
        json.dump(stats, open(data_dir / "norm_stats.json", "w"), indent=2)
        print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
        print(f"  Saved to {data_dir}/")

if __name__ == "__main__":
    main()
