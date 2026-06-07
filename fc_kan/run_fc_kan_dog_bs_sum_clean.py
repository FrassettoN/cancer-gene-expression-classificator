#!/usr/bin/env python3
"""
FC-KAN with Derivative of Gaussian and B-Spline Activations (Clean)
-------------------------------------------------------------------
Runs FC-KAN_dog_bs_sum configuration on gene expression datasets.

The model uses:
- Activation functions: Derivative of Gaussian (dog) and B-spline (bs)
- Combination: Sum (additive)
- Architecture: [input_features, 32, 2]

Results are saved per fold with logits and true labels, plus summary.json.

Usage examples:
  python run_fc_kan_dog_bs_sum_clean.py --cancer "Bladder Urothelial Carcinoma" --gse GSE13507 --mode 10k
  python run_fc_kan_dog_bs_sum_clean.py --cancer "Bladder Urothelial Carcinoma" --gse GSE13507 --mode 30
  python run_fc_kan_dog_bs_sum_clean.py --all --mode 10k
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import functions

from fc_kan import FC_KAN
FC_KAN_AVAILABLE = (FC_KAN is not None)

# Cancer types and their GSE codes
CANCER_TO_GSE: Dict[str, str] = {
    "Bladder Urothelial Carcinoma": "GSE13507",
    "Breast cancer": "TCGA-BRCA",
    "Breast invasive carcinoma cancer": "GSE39004",
    "Colon adenocarcinoma": "GSE41657",
    "Esophageal carcinoma": "GSE20347",
    "Head and Neck squamous cell carcinoma": "GSE6631",
    "Kidney Chromophobe": "GSE15641",
    "Kidney renal clear cell carcinoma": "GSE15641",
    "Kidney renal papillary cell carcinoma": "GSE15641",
    "Liver hepatocellular carcinoma": "GSE45267",
    "Lung adenocarcinoma": "GSE10072",
    "Lung squamous cell carcinoma": "GSE33479",
    "Prostate adenocarcinoma": "GSE6919",
    "Rectum adenocarcinoma": "GSE20842",
    "Stomach adenocarcinoma": "GSE2685",
    "Thyroid carcinoma": "GSE33630",
    "Uterine Corpus Endometrial Carcinoma": "GSE17025",
}

# GSE15641: three cohorts mapping
GSE15641_CANCER_TO_SUFFIX: Dict[str, str] = {
    "Kidney Chromophobe": "1",
    "Kidney renal clear cell carcinoma": "2",
    "Kidney renal papillary cell carcinoma": "3",
}

# TCGA-BRCA: selected_features files use "Breast cancer" not "Breast cancer TCGA"
CANCER_TO_SELECTED_FEATURES_NAME: Dict[str, str] = {
    "Breast cancer TCGA": "Breast cancer"
}

FC_KAN_CONFIG = {
    "func_list": "dog,bs",
    "combined_type": "sum",
    "name": "FC-KAN_dog_bs_sum"
}


def load_processed_matrix(processed_dir: str, gse: str, suffix: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed data matrix. If suffix is set (e.g. '1','2','3'), use only that file for GSE15641-style names."""
    base = Path(processed_dir)
    if suffix is not None:
        csv_path = base / f"{gse}_{suffix}_trasp_mod.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find {csv_path}")
    else:
        candidates = [base / f"{gse}_trasp_mod.csv", *(base / f"{gse}_{s}_trasp_mod.csv" for s in ("1", "2", "3"))]
        csv_path = None
        for p in candidates:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(f"Could not find processed CSV for {gse}")
    
    # Files have no header row; first row is also a sample.
    # Use header=None so we don't lose the first example.
    df = pd.read_csv(csv_path, header=None)
    
    # Find label column (binary 0/1)
    label_col = None
    for c in df.columns:
        vals = pd.unique(df[c])
        try:
            vals_set = set(int(v) for v in vals)
            if vals_set <= {0, 1} and len(vals_set) == 2:
                label_col = c
                break
        except Exception:
            continue
    
    if label_col is None:
        label_col = df.columns[0]
    
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    return X, y


def read_kfold_splits(kfold_file: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Read k-fold splits from file"""
    import re
    
    text = Path(kfold_file).read_text(encoding="utf-8")
    blocks = re.findall(r"\[([^\]]*?)\]", text)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    
    for i in range(0, len(blocks), 2):
        if i + 1 >= len(blocks):
            break
        left = blocks[i]
        right = blocks[i + 1]
        train_idx = np.array([int(x) for x in left.replace("\n", " ").split() 
                             if x.strip().lstrip("-+").isdigit()], dtype=int)
        test_idx = np.array([int(x) for x in right.replace("\n", " ").split() 
                            if x.strip().lstrip("-+").isdigit()], dtype=int)
        splits.append((train_idx, test_idx))
    
    return splits


def read_selected_feature_indices(path: str, limit: int) -> List[int]:
    """Read pre-selected feature indices"""
    indices: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(line))
            if len(indices) >= limit:
                break
    return indices


def sanitize_indices(idx: np.ndarray, n: int) -> np.ndarray:
    """Boundary check and deduplication"""
    if idx.size == 0:
        return idx
    idx = idx[(idx >= 0) & (idx < n)]
    return np.unique(idx)


def prepare_fold_data(
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    mode: str,
    cancer: str,
    fold_idx: int,
    splits_dir: Path,
    datasets_dir: Path,
    selected_features_dir: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Prepare one fold: split, feature selection, and scaling without data leakage"""
    n = len(y)
    tr = sanitize_indices(np.array(tr), n)
    te = sanitize_indices(np.array(te), n)
    
    if tr.size == 0 or te.size == 0:
        return None
    
    # Ensure disjointness
    overlap = np.intersect1d(tr, te, assume_unique=False)
    if overlap.size > 0:
        te = np.array([i for i in te if i not in set(overlap)], dtype=int)
    
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    # Feature selection per fold
    if mode == "30":
        sel_cancer = CANCER_TO_SELECTED_FEATURES_NAME.get(cancer, cancer)
        fold_specific = selected_features_dir / f"selected_features_{sel_cancer}_{fold_idx}.txt"
        default_first = selected_features_dir / f"selected_features_{sel_cancer}_1.txt"
        sel_path = fold_specific if fold_specific.exists() else default_first
        
        if not sel_path.exists():
            # Fallback to relative paths
            fallback_fold = Path(f"selected_features/selected_features_{cancer}_{fold_idx}.txt")
            fallback_first = Path(f"selected_features/selected_features_{cancer}_1.txt")
            sel_path = fallback_fold if fallback_fold.exists() else fallback_first
        
        if not sel_path.exists():
            raise FileNotFoundError(f"Selected features file not found for {cancer} fold {fold_idx}")
        
        idxs = read_selected_feature_indices(str(sel_path), 30)
        X_tr = X_tr[:, idxs]
        X_te = X_te[:, idxs]
    else:
        # 10k mode: fit selector on train only, then transform test
        try:
            selector = SelectKBest(score_func=chi2, k=min(10000, X_tr.shape[1]))
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)
        except ValueError:
            selector = SelectKBest(score_func=f_classif, k=min(10000, X_tr.shape[1]))
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)

    # Always fit the scaler on train only, then transform test
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    return X_tr, y_tr, X_te, y_te


def train_and_predict_logits(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train FC-KAN model and return test logits and predictions"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        logits_np = logits.cpu().numpy()
        preds = np.argmax(logits_np, axis=1)
    
    return logits_np, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC-KAN dog+bs sum on gene expression datasets")
    parser.add_argument("--cancer", help="Cancer type name (exact, matches splits file)")
    parser.add_argument("--gse", help="GSE code, e.g. GSE13507")
    parser.add_argument("--mode", choices=["10k", "30"], default="10k", help="Feature mode")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--splits-dir", type=Path, default=Path("../data/splits/kfold"), help="Directory containing split files")
    parser.add_argument("--datasets-dir", type=Path, default=Path("../data/processed"), help="Directory containing dataset CSV files")
    parser.add_argument("--selected-features-dir", type=Path, default=Path("selected_features"), help="Directory containing selected features files")
    parser.add_argument("--output-dir", type=Path, default=Path("FC-KAN_dog_bs_sum_results"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    if not FC_KAN_AVAILABLE:
        print("Error: FC-KAN is not available. Please install FC-KAN first.")
        return

    config = FC_KAN_CONFIG
    config_name = config["name"]
    func_list = config["func_list"].split(",")
    combined_type = config["combined_type"]

    print(f"Running {config_name} (func_list={func_list}, combined_type={combined_type})")
    print(f"Results will be saved to: {args.output_dir}\n")

    # Determine which cancers to process
    if args.all:
        cancers_to_process = list(CANCER_TO_GSE.items())
    elif args.cancer and args.gse:
        cancers_to_process = [(args.cancer, args.gse)]
    else:
        print("Error: Either specify --cancer and --gse, or use --all")
        return

    base_out = args.output_dir
    out_10k = base_out / "10k_features"
    out_30 = base_out / "30_features"
    out_10k.mkdir(parents=True, exist_ok=True)
    out_30.mkdir(parents=True, exist_ok=True)

    for cancer, gse in cancers_to_process:
        print("=" * 70)
        print(f"Processing: {cancer} ({gse})")

        # Load data
        datasets_path = args.datasets_dir
        if not datasets_path.exists():
            # Try fallback paths
            for candidate in [Path("../datasets").resolve(), Path("processed")]:
                if candidate.exists():
                    if (candidate / f"{gse}_trasp_mod.csv").exists() or \
                       any((candidate / f"{gse}_{s}_trasp_mod.csv").exists() for s in ("1", "2", "3")):
                        datasets_path = candidate
                        break
        
        gse_suffix = None
        if gse == "GSE15641" and cancer in GSE15641_CANCER_TO_SUFFIX:
            gse_suffix = GSE15641_CANCER_TO_SUFFIX[cancer]
        
        X_df, y_sr = load_processed_matrix(str(datasets_path), gse, suffix=gse_suffix)
        X_all = X_df.to_numpy(copy=False)
        y_all = y_sr.to_numpy(copy=False)
        print(f"Samples={len(y_all)}")

        # Read splits
        kfold_file = args.splits_dir / f"splits_kfold_{cancer}.txt"
        if not kfold_file.exists():
            print(f"  Skipping: splits file not found: {kfold_file}")
            continue
        
        splits = read_kfold_splits(str(kfold_file))
        
        # Convert 1-based to 0-based if needed
        max_idx = max(
            max(t.max(), e.max()) for t, e in splits
            if t.size > 0 and e.size > 0
        ) if splits else 0
        if max_idx == len(y_all) and len(y_all) > 0:
            splits = [
                (np.clip(np.asarray(t) - 1, 0, len(y_all) - 1), 
                 np.clip(np.asarray(e) - 1, 0, len(y_all) - 1))
                for t, e in splits
            ]
            print(f"  Converted splits from 1-based to 0-based")

        print(f"Folds={len(splits)}")

        # Process specified mode
        mode = args.mode
            
        print(f"- Running {mode} features...")
        folder_slug = cancer.lower().replace(" ", "_")
        if mode == "10k" and gse:
            folder_slug = f"{folder_slug}_{gse.lower()}"
        
        target_dir = (out_10k if mode == "10k" else out_30) / folder_slug / config_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        fold_records = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            prep = prepare_fold_data(
                X_all, y_all, train_idx, test_idx, mode, cancer, fold_idx + 1,
                args.splits_dir, args.datasets_dir, args.selected_features_dir
            )
            if prep is None:
                continue
            
            X_train, y_train, X_test, y_test = prep

            try:
                model = FC_KAN(
                    layer_list=[X_train.shape[1], 32, 2],
                    func_list=func_list,
                    grid_size=5,
                    spline_order=3,
                    combined_type=combined_type,
                )
                
                logits, preds = train_and_predict_logits(
                    model, X_train, y_train, X_test,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device
                )
                
                acc = accuracy_score(y_test, preds)
                
                # Write fold file
                fold_file = target_dir / f"fold_{fold_idx + 1}.txt"
                with open(fold_file, "w") as f:
                    for i in range(len(y_test)):
                        f.write(f"{logits[i][0]:.6f} {logits[i][1]:.6f} {int(y_test[i])}\n")
                
                fold_records.append({
                    "fold_file": f"fold_{fold_idx + 1}.txt",
                    "samples": len(y_test),
                    "accuracy": float(acc),
                })
                
                print(f"  Fold {fold_idx + 1}: Accuracy={acc:.4f}, Samples={len(y_test)}")
                
            except Exception as e:
                print(f"  Fold {fold_idx + 1}: Error: {e}")
                continue
        
        # Write summary.json
        total_samples = int(sum(fr["samples"] for fr in fold_records))
        if total_samples > 0:
            weighted_acc = float(sum(fr["accuracy"] * fr["samples"] for fr in fold_records) / total_samples)
        else:
            weighted_acc = 0.0

        summary = {
            "dataset": folder_slug,
            "config": config_name,
            "samples": total_samples,
            "accuracy": weighted_acc,
            "folds": [
                {"fold_file": fr["fold_file"], "samples": fr["samples"], "accuracy": fr["accuracy"]}
                for fr in fold_records
            ],
        }
        with open(target_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Completed: {cancer}\n")
    
    print("=" * 70)
    print("All datasets completed!")
    print(f"Results saved to: {base_out}")


if __name__ == "__main__":
    main()
