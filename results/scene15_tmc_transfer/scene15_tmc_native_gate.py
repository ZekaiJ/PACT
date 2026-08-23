#!/usr/bin/env python3
"""Prospective native-reproduction gate for TMC on Scene15.

This runner leaves the pinned TMC and RCML repositories unchanged. It uses the
TMC implementation at the locked commit and the Scene15 feature payload stored
by the locked RCML repository because the public TMC repository distributes
only HandWritten/Mfeat.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset


CLASSES = 15
DIMS = [[20], [59], [40]]
VIEWS = 3
LEARNING_RATES = [3e-3, 1e-3, 3e-4, 1e-4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SplitDataset(Dataset):
    def __init__(self, views: list[np.ndarray], labels: np.ndarray, record_ids: np.ndarray):
        self.views = [np.asarray(view, dtype=np.float32) for view in views]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.record_ids = np.asarray(record_ids, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int):
        return (
            {view: self.views[view][index] for view in range(len(self.views))},
            int(self.labels[index]),
            int(self.record_ids[index]),
        )


def load_scene15(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    payload = sio.loadmat(path)
    views = [np.asarray(payload["X"][0, view]).T for view in range(VIEWS)]
    labels = np.asarray(payload["gt"]).reshape(-1).astype(np.int64)
    if labels.min() == 1:
        labels -= 1
    assert [list(view.shape) for view in views] == [[4485, 20], [4485, 59], [4485, 40]]
    assert len(labels) == 4485 and len(np.unique(labels)) == CLASSES
    return views, labels


def make_outer_split(
    views: list[np.ndarray], labels: np.ndarray, seed: int
) -> tuple[SplitDataset, SplitDataset]:
    all_indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    # Match the released TMC loader: fit MinMaxScaler independently on the
    # complete training and test partitions before optimization/evaluation.
    train_views = [MinMaxScaler((0, 1)).fit_transform(view[train_indices]) for view in views]
    test_views = [MinMaxScaler((0, 1)).fit_transform(view[test_indices]) for view in views]
    return (
        SplitDataset(train_views, labels[train_indices], train_indices),
        SplitDataset(test_views, labels[test_indices], test_indices),
    )


def model_seed(run_seed: int, lr_index: int, fold: int) -> int:
    return int(run_seed * 1000 + lr_index * 10 + fold)


def train_model(TMC, dataset: Dataset, indices: np.ndarray, lr: float, epochs: int,
                batch_size: int, seed: int, device: torch.device):
    set_seed(seed)
    model = TMC(CLASSES, VIEWS, DIMS, 50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # The feature matrices are small. Keeping a split on the GPU removes
    # DataLoader and host-transfer overhead without changing batch size,
    # optimizer steps, epochs, loss, or model architecture.
    features = {
        view: torch.as_tensor(dataset.views[view][indices], dtype=torch.float32, device=device)
        for view in range(VIEWS)
    }
    targets = torch.as_tensor(dataset.labels[indices], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(indices), generator=generator, device=device)
        for start in range(0, len(indices), batch_size):
            batch = order[start : start + batch_size]
            data = {view: features[view][batch] for view in range(VIEWS)}
            target = targets[batch]
            _, _, loss = model(data, target, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def infer(model, dataset: Dataset, indices: np.ndarray, batch_size: int,
          device: torch.device, export_path: Path | None = None) -> float:
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    export_handle = gzip.open(export_path, "wt", encoding="utf-8") if export_path else None
    try:
        model.eval()
        with torch.no_grad():
            for data, target, record_id in loader:
                for view in range(VIEWS):
                    data[view] = data[view].to(device)
                evidence = model.infer(data)
                alpha = {view: evidence[view] + 1.0 for view in range(VIEWS)}
                fused_alpha = model.DS_Combin(alpha)
                prediction = torch.argmax(fused_alpha, dim=1).cpu()
                correct += int((prediction == target).sum().item())
                total += int(len(target))
                if export_handle is not None:
                    evidence_cpu = [evidence[view].cpu().numpy() for view in range(VIEWS)]
                    fused_cpu = fused_alpha.cpu().numpy()
                    for row in range(len(target)):
                        export_handle.write(
                            json.dumps(
                                {
                                    "record_id": int(record_id[row]),
                                    "y": int(target[row]),
                                    "prediction": int(prediction[row]),
                                    "evidences": [values[row].astype(float).tolist() for values in evidence_cpu],
                                    "fused_alpha": fused_cpu[row].astype(float).tolist(),
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
    finally:
        if export_handle is not None:
            export_handle.close()
    return correct / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmc-dir", type=Path, required=True)
    parser.add_argument("--scene15-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--published-mean", type=float, default=0.6774)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--selected-lr", type=float, choices=LEARNING_RATES)
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--cv-task-lr", type=float, choices=LEARNING_RATES)
    parser.add_argument("--cv-task-fold", type=int, choices=range(5))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "exports").mkdir(exist_ok=True)
    sys.path.insert(0, str(args.tmc_dir.resolve()))
    from model import TMC  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the source-faithful TMC loss implementation")
    device = torch.device("cuda:0")
    views, labels = load_scene15(args.scene15_mat)
    rows = []
    for seed in args.seeds:
        train_dataset, test_dataset = make_outer_split(views, labels, seed)
        lr_rows = []
        if args.selected_lr is None:
            folds = list(
                StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(
                    np.arange(len(train_dataset)), train_dataset.labels
                )
            )
            if args.cv_task_lr is not None or args.cv_task_fold is not None:
                if args.cv_task_lr is None or args.cv_task_fold is None or len(args.seeds) != 1:
                    raise ValueError("A CV task requires one seed, --cv-task-lr, and --cv-task-fold")
                lr = float(args.cv_task_lr)
                lr_index = LEARNING_RATES.index(lr)
                fit_indices, validation_indices = folds[int(args.cv_task_fold)]
                model = train_model(
                    TMC,
                    train_dataset,
                    fit_indices,
                    lr,
                    args.epochs,
                    args.batch_size,
                    model_seed(seed, lr_index, int(args.cv_task_fold)),
                    device,
                )
                task_payload = {
                    "seed": seed,
                    "learning_rate": lr,
                    "fold": int(args.cv_task_fold),
                    "validation_accuracy": infer(
                        model, train_dataset, validation_indices, args.batch_size, device
                    ),
                }
                (args.output / "CV_TASK.json").write_text(
                    json.dumps(task_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                print(json.dumps(task_payload, indent=2))
                return 0
            for lr_index, lr in enumerate(LEARNING_RATES):
                fold_accuracies = []
                for fold, (fit_indices, validation_indices) in enumerate(folds):
                    model = train_model(
                        TMC,
                        train_dataset,
                        fit_indices,
                        lr,
                        args.epochs,
                        args.batch_size,
                        model_seed(seed, lr_index, fold),
                        device,
                    )
                    fold_accuracies.append(
                        infer(model, train_dataset, validation_indices, args.batch_size, device)
                    )
                lr_rows.append(
                    {
                        "learning_rate": lr,
                        "fold_accuracies": fold_accuracies,
                        "mean_validation_accuracy": float(np.mean(fold_accuracies)),
                    }
                )
            # Published candidate order is used as the deterministic tie-break.
            selected = max(lr_rows, key=lambda row: row["mean_validation_accuracy"])
            selected_lr = float(selected["learning_rate"])
            cv_payload = {
                "seed": seed,
                "selected_learning_rate": selected_lr,
                "lr_cv": lr_rows,
            }
            (args.output / "CV_SELECTION.json").write_text(
                json.dumps(cv_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if args.cv_only:
                print(json.dumps(cv_payload, indent=2))
                return 0
        else:
            selected_lr = float(args.selected_lr)
        selected_index = LEARNING_RATES.index(selected_lr)
        final_model = train_model(
            TMC,
            train_dataset,
            np.arange(len(train_dataset)),
            selected_lr,
            args.epochs,
            args.batch_size,
            model_seed(seed, selected_index, 9),
            device,
        )
        export_path = args.output / "exports" / f"seed_{seed}.jsonl.gz"
        test_accuracy = infer(
            final_model,
            test_dataset,
            np.arange(len(test_dataset)),
            args.batch_size,
            device,
            export_path,
        )
        row = {
            "seed": seed,
            "selected_learning_rate": selected_lr,
            "test_accuracy": test_accuracy,
            "train_instances": len(train_dataset),
            "test_instances": len(test_dataset),
            "lr_cv": lr_rows,
            "export": str(export_path),
            "export_sha256": sha256(export_path),
        }
        rows.append(row)
        (args.output / f"seed_{seed}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    accuracies = [float(row["test_accuracy"]) for row in rows]
    mean_accuracy = float(np.mean(accuracies))
    gate_pass = (
        len(rows) == len(args.seeds)
        and abs(mean_accuracy - args.published_mean) <= args.tolerance
    )
    summary = {
        "schema_version": "scene15-tmc-native-gate-1.0",
        "verdict": "PASS" if gate_pass else "MISMATCH",
        "dataset": "Scene15",
        "emitter": "TMC",
        "published_mean_accuracy": args.published_mean,
        "published_std_accuracy": 0.0036,
        "acceptance_tolerance_absolute": args.tolerance,
        "observed_mean_accuracy": mean_accuracy,
        "observed_population_std_accuracy": float(np.std(accuracies)),
        "successful_runs": len(rows),
        "required_runs": len(args.seeds),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rates": LEARNING_RATES,
        "dataset_sha256": sha256(args.scene15_mat),
        "rows": rows,
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
