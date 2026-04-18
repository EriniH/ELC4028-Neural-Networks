"""Pipeline 1: K-Means bootstrapping with human-in-the-loop SVM refinement.

Single-file submission script for Assignment 1 Part 3.

Usage:
- Run pipeline: python pipeline1_human_in_loop.py [run options]
- Practical set prepare: python pipeline1_human_in_loop.py practical-prepare [...]
- Practical set score: python pipeline1_human_in_loop.py practical-score [...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.fftpack import dct
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "assignmnet_materials" / "Indian_Digits_Train"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pipeline1_results"
IMAGE_SUFFIX = ".bmp"
EXPECTED_IMAGE_COUNT = 10_000
PRACTICAL_GATE_LABEL_COUNT = 500
PRACTICAL_GATE_MIN_ACCURACY = 0.99


def resolve_input_path(path: Path) -> Path:
    """Resolve user-provided input paths from common invocation locations."""
    raw = Path(path)
    if raw.is_absolute():
        return raw

    candidates = [raw, SCRIPT_DIR / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return raw


def find_latest_run_predictions_csv(output_dir: Path) -> Path:
    """Return the newest run_*/predictions_final.csv under output_dir."""
    resolved_output_dir = resolve_input_path(output_dir)
    candidates: list[Path] = []

    if resolved_output_dir.exists() and resolved_output_dir.is_dir():
        for child in resolved_output_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("run_"):
                continue
            pred_csv = child / "predictions_final.csv"
            if pred_csv.exists():
                candidates.append(pred_csv)

    if not candidates:
        for pred_csv in PROJECT_ROOT.rglob("predictions_final.csv"):
            if pred_csv.parent.name.startswith("run_"):
                candidates.append(pred_csv)

    if not candidates:
        raise FileNotFoundError(
            "No predictions_final.csv found in run_* folders. "
            "Run the pipeline once first, or pass --predictions-csv explicitly. "
            f"Checked output-dir={resolved_output_dir} and project root={PROJECT_ROOT}."
        )

    return max(candidates, key=lambda p: (p.parent.name, p.stat().st_mtime))


def resolve_practical_predictions_csv(
    predictions_csv: Path | None,
    output_dir: Path,
) -> tuple[Path, bool]:
    """Resolve predictions CSV path; auto-select latest run when omitted."""
    if predictions_csv is None:
        return find_latest_run_predictions_csv(output_dir), True

    resolved = resolve_input_path(predictions_csv)
    if resolved.exists() and resolved.is_dir():
        return resolved / "predictions_final.csv", False
    return resolved, False


@dataclass
class ClusterDecision:
    cluster_id: int
    cluster_size: int
    sampled_image_ids: list[int]
    chosen_label: int | None
    assigned_count: int


@dataclass
class BoundaryAnnotation:
    iteration: int
    image_id: int
    assigned_label: int
    predicted_label_before: int
    margin_before: float


@dataclass
class IterationMetric:
    iteration: int
    train_size: int
    boundary_labeled_this_iter: int
    total_boundary_labeled: int
    total_manual_images: int
    total_manual_time_s: float
    measured_accuracy: float | None
    accuracy_scope: str
    improvement: float | None


@dataclass
class PredictionRow:
    image_id: int
    predicted_label: int


def with_progress(iterable, desc: str, total: int | None = None):
    """Wrap an iterable with tqdm when available."""
    try:
        import importlib

        tqdm = importlib.import_module("tqdm.auto").tqdm
    except ImportError:
        if total is None:
            return iterable

        checkpoints = sorted(set(max(1, int(round(total * frac / 10.0))) for frac in range(1, 11)))

        def fallback_iter():
            print(f"{desc}: started ({total} items)")
            checkpoint_idx = 0
            for idx, item in enumerate(iterable, start=1):
                while checkpoint_idx < len(checkpoints) and idx >= checkpoints[checkpoint_idx]:
                    pct = 100.0 * checkpoints[checkpoint_idx] / total
                    print(f"{desc}: {checkpoints[checkpoint_idx]}/{total} ({pct:.0f}%)")
                    checkpoint_idx += 1
                yield item
            if checkpoint_idx < len(checkpoints):
                print(f"{desc}: {total}/{total} (100%)")
            print(f"{desc}: completed")

        return fallback_iter()

    return tqdm(iterable, desc=desc, total=total, leave=False)


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline 1 with human-in-the-loop labeling prompts"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder that contains 1.bmp ... 10000.bmp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where logs, previews, and predictions are saved",
    )
    parser.add_argument(
        "--feature",
        choices=["raw", "dct", "hog", "pca"],
        default="hog",
        help="Feature representation for clustering and SVM",
    )
    parser.add_argument(
        "--k-clusters",
        type=int,
        choices=[50],
        default=50,
        help="K for K-means is fixed to 50 for this assignment run",
    )
    parser.add_argument(
        "--sample-per-cluster",
        type=int,
        default=8,
        help="How many images to show human per cluster",
    )
    parser.add_argument(
        "--boundary-batch",
        type=int,
        choices=[30],
        default=30,
        help="Boundary labeling batch size is fixed to 30 (assignment setting)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=8,
        help="Maximum refinement iterations",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.99,
        help="Stop when measured accuracy reaches this threshold",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help=(
            "Stop when improvement falls below this fraction "
            "(0.001 means 0.1%%)."
        ),
    )
    parser.add_argument(
        "--cluster-weight",
        type=float,
        default=1.0,
        help="Sample weight for cluster-propagated labels",
    )
    parser.add_argument(
        "--trusted-weight",
        type=float,
        default=100.0,
        help="Sample weight for human boundary labels",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Expected image height and width",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        help="Optional CSV with columns image_id,label for auto accuracy",
    )
    parser.add_argument(
        "--practical-annotated-csv",
        type=Path,
        help=(
            "Optional practical CSV with columns image_id,human_label. "
            "When provided, measured accuracy is computed from this subset "
            "if oracle ground truth is unavailable."
        ),
    )
    parser.add_argument(
        "--use-practical-for-training",
        action="store_true",
        help=(
            "Allow practical subset labels to be injected as training corrections. "
            "By default, practical labels are evaluation-only."
        ),
    )
    return parser.parse_args()


def parse_image_id(value: str) -> int:
    text = value.strip().lower()
    if text.endswith(IMAGE_SUFFIX):
        text = text[: -len(IMAGE_SUFFIX)]
    return int(text)


def normalize_fieldname(name: str | None) -> str:
    if name is None:
        return ""
    text = str(name).strip().lstrip("\ufeff").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.lower()


def normalize_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if fieldnames is None:
        return []
    return [normalize_fieldname(name) for name in fieldnames]


def parse_optional_digit(text: str) -> int | None:
    value = text.strip()
    if value == "":
        return None
    digit = int(value)
    if digit < 0 or digit > 9:
        raise ValueError(f"Label must be in [0,9], got {digit}")
    return digit


def read_predictions(path: Path) -> list[PredictionRow]:
    resolved_path = resolve_input_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")

    rows: list[PredictionRow] = []
    with resolved_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = normalize_fieldnames(reader.fieldnames)
        required = {"image_id", "predicted_label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("predictions CSV must contain columns: image_id,predicted_label")

        for row in reader:
            rows.append(
                PredictionRow(
                    image_id=parse_image_id(str(row["image_id"])),
                    predicted_label=int(str(row["predicted_label"]).strip()),
                )
            )

    if not rows:
        raise ValueError("Predictions CSV is empty")
    return rows


def read_dataset_image_ids(data_dir: Path) -> list[int]:
    resolved_dir = resolve_input_path(data_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    image_ids: list[int] = []
    for path in resolved_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != IMAGE_SUFFIX:
            continue
        try:
            image_ids.append(parse_image_id(path.name))
        except ValueError:
            continue

    if not image_ids:
        raise ValueError(f"No {IMAGE_SUFFIX} files found in: {data_dir}")

    return sorted(set(image_ids))


def uniform_sample_ids(image_ids: list[int], sample_size: int, seed: int) -> list[int]:
    if sample_size > len(image_ids):
        raise ValueError(f"sample_size={sample_size} exceeds available rows={len(image_ids)}")
    rng = random.Random(seed)
    picked = rng.sample(image_ids, sample_size)
    picked.sort()
    return picked


def write_practical_annotation_sheet(
    path: Path,
    sampled_ids: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "predicted_label", "human_label", "notes"])
        for image_id in sampled_ids:
            writer.writerow([image_id, "", "", ""])


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def enforce_practical_accuracy_gate(
    accuracy: float,
    labeled_rows: int,
    context: str,
) -> None:
    """Require practical accuracy >99% when evaluated on exactly 500 labeled rows."""
    if labeled_rows != PRACTICAL_GATE_LABEL_COUNT:
        return

    if accuracy <= PRACTICAL_GATE_MIN_ACCURACY:
        raise SystemExit(
            f"{context} failed: accuracy must exceed "
            f"{100.0 * PRACTICAL_GATE_MIN_ACCURACY:.2f}% on "
            f"{PRACTICAL_GATE_LABEL_COUNT} labeled images, "
            f"but got {100.0 * accuracy:.2f}%."
        )


def score_practical_annotated_sheet(
    path: Path,
    strict_size: int,
    predictions_by_id: dict[int, int],
) -> dict:
    resolved_path = resolve_input_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Annotated CSV not found: {path}")

    labeled = 0
    correct = 0
    unlabeled = 0

    with resolved_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = normalize_fieldnames(reader.fieldnames)
        required = {"image_id", "human_label"}

        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("annotated CSV must contain columns: image_id,human_label")

        for line_no, row in enumerate(reader, start=2):
            try:
                image_id = parse_image_id(str(row.get("image_id", "")).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid image_id at CSV line {line_no}") from exc

            human = parse_optional_digit(str(row.get("human_label", "")))
            if human is None:
                unlabeled += 1
                continue

            if image_id not in predictions_by_id:
                raise ValueError(
                    f"Image ID {image_id} (CSV line {line_no}) was not found in predictions CSV"
                )
            pred = predictions_by_id[image_id]

            labeled += 1
            if pred == human:
                correct += 1

    if strict_size > 0 and labeled != strict_size:
        raise ValueError(
            f"Expected exactly {strict_size} labeled rows, but found {labeled}. "
            "Fill the 'human_label' column first (0-9) for all sampled rows."
        )

    if labeled == 0:
        raise ValueError(
            "No human-labeled rows found; cannot compute practical accuracy. "
            "Fill the 'human_label' column (0-9) first."
        )

    accuracy = correct / labeled
    ci_lo, ci_hi = wilson_interval(correct, labeled)

    return {
        "labeled_rows": labeled,
        "unlabeled_rows": unlabeled,
        "correct": correct,
        "accuracy": accuracy,
        "wilson95_ci": [ci_lo, ci_hi],
        "prediction_source": "predictions_csv",
    }


def parse_practical_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Practical 5% hold-out helpers (single-file mode)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("practical-prepare", help="Create practical_500_to_label.csv")
    prep.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder with 1.bmp...10000.bmp",
    )
    prep.add_argument("--output-csv", type=Path, required=True, help="Output annotation CSV path")
    prep.add_argument("--sample-size", type=int, default=500, help="Number of images to sample")
    prep.add_argument("--seed", type=int, default=42, help="Random seed")

    score = sub.add_parser("practical-score", help="Score practical accuracy")
    score.add_argument("--annotated-csv", type=Path, required=True, help="CSV with image_id,human_label")
    score.add_argument(
        "--predictions-csv",
        type=Path,
        help=(
            "predictions_final.csv for scoring by image_id. "
            "If omitted, the newest run_*/predictions_final.csv under --output-dir is used."
        ),
    )
    score.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder used for auto-selecting latest predictions CSV",
    )
    score.add_argument("--strict-size", type=int, default=500, help="Expected labeled count (0 disables check)")
    score.add_argument("--save-json", type=Path, help="Optional output JSON summary path")

    return parser.parse_args(argv)


def run_practical_prepare(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")

    image_ids = read_dataset_image_ids(args.data_dir)
    sampled_ids = uniform_sample_ids(image_ids, args.sample_size, args.seed)
    write_practical_annotation_sheet(args.output_csv, sampled_ids)

    print("Prepared practical annotation sheet")
    print(f"- Source      : dataset folder: {args.data_dir}")
    print(f"- Output CSV  : {args.output_csv}")
    print(f"- Sample size : {len(sampled_ids)}")
    print("- Sampling    : uniform")
    print("Next step: fill human_label (0-9) for each sampled row.")


def run_practical_score(args: argparse.Namespace) -> None:
    predictions_csv, auto_selected = resolve_practical_predictions_csv(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
    )

    predictions = read_predictions(predictions_csv)
    predictions_by_id = {row.image_id: row.predicted_label for row in predictions}

    summary = score_practical_annotated_sheet(args.annotated_csv, args.strict_size, predictions_by_id)

    print("Practical ground-truth results")
    if auto_selected:
        print(f"- Auto-selected predictions CSV: {predictions_csv}")
    print(f"- Prediction source: {summary['prediction_source']}")
    print(f"- Predictions CSV : {predictions_csv}")
    print(f"- Labeled rows   : {summary['labeled_rows']}")
    print(f"- Unlabeled rows : {summary['unlabeled_rows']}")
    print(f"- Correct        : {summary['correct']}/{summary['labeled_rows']}")
    print(f"- Accuracy       : {summary['accuracy']:.6f} ({100.0 * summary['accuracy']:.2f}%)")
    print(f"- Wilson 95% CI  : [{summary['wilson95_ci'][0]:.6f}, {summary['wilson95_ci'][1]:.6f}]")

    enforce_practical_accuracy_gate(
        accuracy=float(summary["accuracy"]),
        labeled_rows=int(summary["labeled_rows"]),
        context="Practical score quality gate",
    )
    if int(summary["labeled_rows"]) == PRACTICAL_GATE_LABEL_COUNT:
        print(
            f"- Quality gate   : PASS (>"
            f"{100.0 * PRACTICAL_GATE_MIN_ACCURACY:.2f}% on "
            f"{PRACTICAL_GATE_LABEL_COUNT} labeled images)"
        )

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"- Saved summary  : {args.save_json}")


def load_images(data_dir: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    resolved_dir = resolve_input_path(data_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    files = sorted(
        [path for path in resolved_dir.iterdir() if path.is_file() and path.suffix.lower() == IMAGE_SUFFIX],
        key=lambda path: int(path.stem),
    )

    if not files:
        raise FileNotFoundError(f"No .bmp files found in: {data_dir}")

    if len(files) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_IMAGE_COUNT} .bmp images in {data_dir}, "
            f"found {len(files)}."
        )

    images: list[np.ndarray] = []
    image_ids: list[int] = []

    for path in with_progress(files, "Loading images", len(files)):
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        if arr.shape != (image_size, image_size):
            raise ValueError(
                f"Expected shape {(image_size, image_size)} for {path.name}, found {arr.shape}."
            )
        images.append(arr / 255.0)
        image_ids.append(int(path.stem))

    return np.asarray(images), np.asarray(image_ids, dtype=np.int32), files


def dct2(image: np.ndarray) -> np.ndarray:
    return dct(dct(image.T, norm="ortho").T, norm="ortho")


def extract_features(images: np.ndarray, feature_kind: str, random_seed: int) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()

    if feature_kind == "raw":
        features = images.reshape(len(images), -1)
        meta = {"feature": "raw", "dims": int(features.shape[1])}
    elif feature_kind == "dct":
        rows = []
        for image in with_progress(images, "DCT features", len(images)):
            coeffs = dct2(image)
            rows.append(coeffs[:15, :15].reshape(-1))
        features = np.asarray(rows)
        meta = {"feature": "dct", "dims": int(features.shape[1]), "block": "15x15"}
    elif feature_kind == "hog":
        rows = []
        for image in with_progress(images, "HOG features", len(images)):
            rows.append(
                hog(
                    image,
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    feature_vector=True,
                )
            )
        features = np.asarray(rows)
        meta = {
            "feature": "hog",
            "dims": int(features.shape[1]),
            "orientations": 9,
            "pixels_per_cell": [4, 4],
            "cells_per_block": [2, 2],
        }
    elif feature_kind == "pca":
        flat = images.reshape(len(images), -1)
        pca = PCA(n_components=0.95, svd_solver="full", random_state=random_seed)
        features = pca.fit_transform(flat)
        meta = {
            "feature": "pca",
            "dims": int(features.shape[1]),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        }
    else:
        raise ValueError(f"Unsupported feature type: {feature_kind}")

    elapsed = time.perf_counter() - start
    meta["feature_time_s"] = elapsed
    return features, meta


def create_contact_sheet(
    images: np.ndarray,
    image_ids: np.ndarray,
    selected_indices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    tile_scale = 4
    rows = int(math.ceil(len(selected_indices) / 4))
    cols = min(4, len(selected_indices))

    tile_w = images.shape[2] * tile_scale
    tile_h = images.shape[1] * tile_scale
    pad = 14
    title_h = 28

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad + title_h
    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=0)

    for pos, idx in enumerate(selected_indices):
        row = pos // 4
        col = pos % 4
        x = pad + col * (tile_w + pad)
        y = title_h + pad + row * (tile_h + pad)

        img = Image.fromarray((images[idx] * 255.0).astype(np.uint8), mode="L")
        img = img.resize((tile_w, tile_h), Image.Resampling.NEAREST)
        canvas.paste(img, (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=0, width=1)
        draw.text((x + 2, y + 2), str(int(image_ids[idx])), fill=255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def prompt_digit_or_mixed(prompt_text: str) -> int | None:
    while True:
        answer = input(prompt_text).strip().lower()
        if answer in {"q", "quit", "exit"}:
            raise KeyboardInterrupt("Stopped by user")
        if answer in {"m", "mixed"}:
            return None
        if answer.isdigit() and 0 <= int(answer) <= 9:
            return int(answer)
        print(
            "Invalid input. Use 0-9 to label the whole cluster, "
            "m for mixed cluster (leave unlabeled), or q to quit the run."
        )


def prompt_digit(prompt_text: str) -> int | None:
    while True:
        answer = input(prompt_text).strip().lower()
        if answer in {"q", "quit", "exit"}:
            raise KeyboardInterrupt("Stopped by user")
        if answer in {"s", "skip", ""}:
            return None
        if answer.isdigit() and 0 <= int(answer) <= 9:
            return int(answer)
        print(
            "Invalid input. Use 0-9 to assign human label, "
            "s to skip and keep current label/weight, or q to quit the run."
        )


def bootstrap_cluster_labels(
    images: np.ndarray,
    image_ids: np.ndarray,
    cluster_ids: np.ndarray,
    sample_per_cluster: int,
    cluster_weight: float,
    preview_dir: Path,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ClusterDecision], int, float]:
    n = len(images)
    labels = np.full(n, -1, dtype=np.int32)
    sources = np.array(["unlabeled"] * n, dtype=object)
    weights = np.zeros(n, dtype=np.float32)

    manual_images_viewed = 0
    manual_time_s = 0.0
    decisions: list[ClusterDecision] = []

    unique_clusters = sorted(int(c) for c in np.unique(cluster_ids))

    print("\nStep 2: Human cluster-majority labeling")
    print(
        f"For each cluster: inspect {sample_per_cluster} sampled examples, then choose one option."
    )
    print(
        "Input guide: 0-9 = assign this label to the whole cluster | "
        "m = mixed cluster (do not assign cluster label) | "
        "q = quit run"
    )
    print(f"Cluster previews will be saved to: {preview_dir}")

    for cluster_id in unique_clusters:
        member_idx = np.where(cluster_ids == cluster_id)[0]
        if len(member_idx) == 0:
            continue

        local_rng = np.random.default_rng(random_seed + cluster_id)
        take = min(sample_per_cluster, len(member_idx))
        sampled_idx = np.sort(local_rng.choice(member_idx, size=take, replace=False))

        preview_path = preview_dir / f"cluster_{cluster_id:03d}.png"
        title = f"Cluster {cluster_id} | size={len(member_idx)}"
        create_contact_sheet(images, image_ids, sampled_idx, preview_path, title)

        sampled_ids = [int(image_ids[idx]) for idx in sampled_idx]
        print(f"\nCluster {cluster_id:03d} size={len(member_idx)}")
        print(f"Sample IDs: {sampled_ids}")
        print(f"Preview   : {preview_path}")

        chosen = prompt_digit_or_mixed(
            "Assign cluster label [0-9=label, m=mixed, q=quit]: "
        )

        manual_images_viewed += take
        manual_time_s += 20.0

        assigned_count = 0
        if chosen is not None:
            labels[member_idx] = chosen
            sources[member_idx] = "cluster"
            weights[member_idx] = cluster_weight
            assigned_count = int(len(member_idx))

        decisions.append(
            ClusterDecision(
                cluster_id=cluster_id,
                cluster_size=int(len(member_idx)),
                sampled_image_ids=sampled_ids,
                chosen_label=chosen,
                assigned_count=assigned_count,
            )
        )

    labeled_count = int(np.sum(labels >= 0))
    print(f"\nCluster bootstrapping complete: labeled {labeled_count}/{n} images.")
    return labels, sources, weights, decisions, manual_images_viewed, manual_time_s


def train_weighted_svm(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    mask = labels >= 0
    train_x = features[mask]
    train_y = labels[mask]
    train_w = weights[mask]

    if len(train_x) == 0:
        raise ValueError("No labeled samples available for SVM training.")

    unique_labels = np.unique(train_y)
    if len(unique_labels) < 2:
        raise ValueError(
            "SVM needs at least two classes. Cluster labels are too sparse/mixed; relabel clusters."
        )

    model = make_pipeline(
        StandardScaler(),
        # OVR decision scores align with top1-top2 margin uncertainty ranking.
        SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr"),
    )
    model.fit(train_x, train_y, svc__sample_weight=train_w)
    return model, mask


def predict_labels_and_margins(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(features)
    scores = model.decision_function(features)

    if scores.ndim == 1:
        margins = np.abs(scores)
    else:
        top1 = np.max(scores, axis=1)
        top2 = np.partition(scores, -2, axis=1)[:, -2]
        margins = top1 - top2

    return predictions.astype(np.int32), margins.astype(np.float32)


def pick_low_margin_indices(
    margins: np.ndarray,
    already_trusted_mask: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    candidate_idx = np.where(~already_trusted_mask)[0]
    if len(candidate_idx) == 0:
        return np.array([], dtype=np.int32)

    ordered = candidate_idx[np.argsort(margins[candidate_idx])]
    return ordered[: min(batch_size, len(ordered))]


def human_label_boundary_images(
    iteration: int,
    images: np.ndarray,
    image_ids: np.ndarray,
    file_paths: list[Path],
    selected_indices: np.ndarray,
    predicted_labels: np.ndarray,
    margins: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
    trusted_weight: float,
    preview_dir: Path,
) -> tuple[int, list[BoundaryAnnotation]]:
    added = 0
    annotations: list[BoundaryAnnotation] = []

    print(f"\nStep 4 (iteration {iteration}): Human boundary-image labeling")
    print(
        "Input guide: 0-9 = assign human label and trust it strongly | "
        "s = skip this image and keep current label/weight | "
        "q = quit run"
    )

    for rank, idx in enumerate(selected_indices, start=1):
        image_id = int(image_ids[idx])
        pred = int(predicted_labels[idx])
        margin = float(margins[idx])

        preview_path = preview_dir / f"iter_{iteration:02d}_rank_{rank:02d}_id_{image_id:05d}.png"
        create_contact_sheet(
            images,
            image_ids,
            np.asarray([idx], dtype=np.int32),
            preview_path,
            title=f"Iter {iteration} | id={image_id} | pred={pred} | margin={margin:.6f}",
        )

        print(f"\n[{rank}/{len(selected_indices)}] Image {image_id}.bmp")
        print(f"Prediction before human label: {pred}")
        print(f"Margin (smaller = more uncertain): {margin:.6f}")
        print(f"Original file: {file_paths[idx]}")
        print(f"Preview file : {preview_path}")

        assigned = prompt_digit(
            "Human label [0-9=assign, s=skip, q=quit]: "
        )
        if assigned is None:
            continue

        labels[idx] = assigned
        sources[idx] = "boundary"
        weights[idx] = trusted_weight
        added += 1

        annotations.append(
            BoundaryAnnotation(
                iteration=iteration,
                image_id=image_id,
                assigned_label=int(assigned),
                predicted_label_before=pred,
                margin_before=margin,
            )
        )

    print(f"Boundary labels added this iteration: {added}")
    return added, annotations


def auto_label_practical_mismatches(
    iteration: int,
    image_ids: np.ndarray,
    predicted_labels: np.ndarray,
    margins: np.ndarray,
    practical_map: dict[int, int],
    labels: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
    trusted_weight: float,
    limit: int,
) -> tuple[int, list[BoundaryAnnotation]]:
    """Use practical human labels as trusted corrections for current mismatches."""
    if not practical_map:
        return 0, []

    idx_by_id = {int(image_id): idx for idx, image_id in enumerate(image_ids)}
    mismatch_idx: list[int] = []

    for image_id, true_label in practical_map.items():
        idx = idx_by_id.get(int(image_id))
        if idx is None:
            continue
        if int(predicted_labels[idx]) != int(true_label):
            mismatch_idx.append(idx)

    if not mismatch_idx:
        return 0, []

    mismatch_idx.sort(key=lambda idx: float(margins[idx]))
    if limit > 0:
        selected = mismatch_idx[:limit]
    else:
        selected = mismatch_idx

    annotations: list[BoundaryAnnotation] = []
    for idx in selected:
        image_id = int(image_ids[idx])
        assigned_label = int(practical_map[image_id])
        pred_before = int(predicted_labels[idx])
        margin_before = float(margins[idx])

        labels[idx] = assigned_label
        sources[idx] = "practical"
        weights[idx] = trusted_weight

        annotations.append(
            BoundaryAnnotation(
                iteration=iteration,
                image_id=image_id,
                assigned_label=assigned_label,
                predicted_label_before=pred_before,
                margin_before=margin_before,
            )
        )

    return len(selected), annotations


def load_ground_truth_csv(path: Path) -> dict[int, int]:
    resolved_path = resolve_input_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    truth: dict[int, int] = {}

    with resolved_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        first = next(reader, None)
        if first is None:
            return truth

        rows = []
        if len(first) >= 2:
            header_like = first[0].strip().lower() in {"image_id", "id", "filename"}
            if not header_like:
                rows.append(first)
        for row in reader:
            rows.append(row)

    for row in rows:
        if len(row) < 2:
            continue
        image_id = parse_image_id(row[0])
        label = int(row[1])
        if 0 <= label <= 9:
            truth[image_id] = label

    return truth


def load_practical_annotated_csv(path: Path) -> dict[int, int]:
    resolved_path = resolve_input_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Practical annotated CSV not found: {path}")

    practical_map: dict[int, int] = {}

    with resolved_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        normalized = {normalize_fieldname(name): name for name in fieldnames}

        if "image_id" not in normalized or "human_label" not in normalized:
            raise ValueError(
                "Practical annotated CSV must contain columns: image_id,human_label"
            )

        image_col = normalized["image_id"]
        human_col = normalized["human_label"]

        for line_no, row in enumerate(reader, start=2):
            raw_image = "" if row.get(image_col) is None else str(row.get(image_col)).strip()
            raw_human = "" if row.get(human_col) is None else str(row.get(human_col)).strip()

            if raw_human == "":
                continue
            if raw_image == "":
                raise ValueError(f"Missing image_id at CSV line {line_no}")

            try:
                image_id = parse_image_id(raw_image)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid image_id '{raw_image}' at CSV line {line_no}"
                ) from exc

            try:
                human_label = int(raw_human)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid human_label '{raw_human}' at CSV line {line_no}"
                ) from exc

            if not 0 <= human_label <= 9:
                raise ValueError(
                    f"human_label must be in [0,9], got {human_label} at CSV line {line_no}"
                )

            practical_map[image_id] = human_label

    if not practical_map:
        raise ValueError(
            "No non-empty human_label rows found in practical annotated CSV."
        )

    return practical_map


def evaluate_with_reference(
    predictions: np.ndarray,
    image_ids: np.ndarray,
    reference_map: dict[int, int],
    source_name: str,
) -> tuple[float | None, str]:
    if not reference_map:
        return None, "none"

    matched_pred = []
    matched_true = []
    for idx, image_id in enumerate(image_ids):
        key = int(image_id)
        if key in reference_map:
            matched_pred.append(int(predictions[idx]))
            matched_true.append(int(reference_map[key]))

    if not matched_true:
        return None, "none"

    y_pred = np.asarray(matched_pred)
    y_true = np.asarray(matched_true)
    acc = float(np.mean(y_pred == y_true))
    return acc, f"{source_name} ({len(y_true)} samples)"


def evaluate_with_truth(
    predictions: np.ndarray,
    image_ids: np.ndarray,
    truth_map: dict[int, int],
) -> tuple[float | None, str]:
    return evaluate_with_reference(
        predictions=predictions,
        image_ids=image_ids,
        reference_map=truth_map,
        source_name="ground_truth_csv",
    )


def evaluate_with_practical(
    predictions: np.ndarray,
    image_ids: np.ndarray,
    practical_map: dict[int, int],
) -> tuple[float | None, str]:
    return evaluate_with_reference(
        predictions=predictions,
        image_ids=image_ids,
        reference_map=practical_map,
        source_name="practical_annotated_csv",
    )


def evaluate_with_available_sources(
    predictions: np.ndarray,
    image_ids: np.ndarray,
    truth_map: dict[int, int],
    practical_map: dict[int, int],
) -> tuple[float | None, str]:
    measured_acc, scope = evaluate_with_truth(predictions, image_ids, truth_map)
    if measured_acc is None:
        measured_acc, scope = evaluate_with_practical(predictions, image_ids, practical_map)
    return measured_acc, scope


def dispatch_practical_command() -> bool:
    if len(sys.argv) < 2:
        return False

    command = sys.argv[1].strip().lower()
    if command not in {"practical-prepare", "practical-score"}:
        return False

    args = parse_practical_args(sys.argv[1:])
    if args.command == "practical-prepare":
        run_practical_prepare(args)
    elif args.command == "practical-score":
        run_practical_score(args)
    else:
        raise ValueError(f"Unsupported practical command: {args.command}")

    return True


def write_cluster_decisions(path: Path, decisions: list[ClusterDecision]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["cluster_id", "cluster_size", "sampled_image_ids", "chosen_label", "assigned_count"])
        for item in decisions:
            writer.writerow(
                [
                    item.cluster_id,
                    item.cluster_size,
                    " ".join(str(x) for x in item.sampled_image_ids),
                    "mixed" if item.chosen_label is None else item.chosen_label,
                    item.assigned_count,
                ]
            )


def write_boundary_annotations(path: Path, rows: list[BoundaryAnnotation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "iteration",
                "image_id",
                "assigned_label",
                "predicted_label_before",
                "margin_before",
            ]
        )
        for item in rows:
            writer.writerow(
                [
                    item.iteration,
                    item.image_id,
                    item.assigned_label,
                    item.predicted_label_before,
                    f"{item.margin_before:.8f}",
                ]
            )


def write_iteration_log(path: Path, rows: list[IterationMetric]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "iteration",
                "train_size",
                "boundary_labeled_this_iter",
                "total_boundary_labeled",
                "total_manual_images",
                "total_manual_time_s",
                "measured_accuracy",
                "accuracy_scope",
                "improvement",
            ]
        )

        for item in rows:
            writer.writerow(
                [
                    item.iteration,
                    item.train_size,
                    item.boundary_labeled_this_iter,
                    item.total_boundary_labeled,
                    item.total_manual_images,
                    f"{item.total_manual_time_s:.2f}",
                    "" if item.measured_accuracy is None else f"{item.measured_accuracy:.8f}",
                    item.accuracy_scope,
                    "" if item.improvement is None else f"{item.improvement:.8f}",
                ]
            )


def write_predictions(
    path: Path,
    image_ids: np.ndarray,
    predicted_labels: np.ndarray,
    training_labels: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "predicted_label",
                "training_label",
                "label_source",
                "sample_weight",
            ]
        )
        for idx, image_id in enumerate(image_ids):
            train_label = "" if training_labels[idx] < 0 else int(training_labels[idx])
            writer.writerow(
                [
                    int(image_id),
                    int(predicted_labels[idx]),
                    train_label,
                    str(sources[idx]),
                    f"{float(weights[idx]):.1f}",
                ]
            )


def save_summary(
    path: Path,
    args: argparse.Namespace,
    feature_meta: dict,
    total_images: int,
    cluster_decisions: list[ClusterDecision],
    manual_images_clusters: int,
    manual_time_clusters_s: float,
    total_boundary_labeled: int,
    iteration_metrics: list[IterationMetric],
    final_accuracy: float | None,
    final_scope: str,
) -> None:
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "data_dir": str(args.data_dir),
            "feature": args.feature,
            "k_clusters": args.k_clusters,
            "boundary_batch": args.boundary_batch,
            "max_iters": args.max_iters,
            "target_accuracy": args.target_accuracy,
            "min_improvement": args.min_improvement,
            "cluster_weight": args.cluster_weight,
            "trusted_weight": args.trusted_weight,
            "random_seed": args.random_seed,
            "ground_truth_csv": None if args.ground_truth_csv is None else str(args.ground_truth_csv),
            "practical_annotated_csv": (
                None
                if args.practical_annotated_csv is None
                else str(args.practical_annotated_csv)
            ),
            "use_practical_for_training": bool(args.use_practical_for_training),
        },
        "feature_meta": feature_meta,
        "dataset": {"images": total_images},
        "cluster_step": {
            "clusters_reviewed": len(cluster_decisions),
            "manual_images_viewed": manual_images_clusters,
            "manual_time_s": manual_time_clusters_s,
            "mixed_clusters": int(sum(1 for x in cluster_decisions if x.chosen_label is None)),
        },
        "boundary_step": {
            "total_boundary_labeled": total_boundary_labeled,
            "manual_time_s": float(total_boundary_labeled * 10.0),
        },
        "totals": {
            "manual_images": int(manual_images_clusters + total_boundary_labeled),
            "manual_time_s": float(manual_time_clusters_s + total_boundary_labeled * 10.0),
            "manual_time_h": float((manual_time_clusters_s + total_boundary_labeled * 10.0) / 3600.0),
            "iterations_completed": len(iteration_metrics),
        },
        "final_accuracy": {
            "value": final_accuracy,
            "scope": final_scope,
        },
        "iteration_metrics": [asdict(item) for item in iteration_metrics],
    }

    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    if dispatch_practical_command():
        return

    args = parse_run_args()

    run_dir = args.output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    preview_cluster_dir = run_dir / "cluster_previews"
    preview_boundary_dir = run_dir / "boundary_previews"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Pipeline 1 (human-in-the-loop) ===")
    print(f"Data folder  : {args.data_dir}")
    print(f"Output folder: {run_dir}")
    print("Loading images from disk (this can take some time)...")

    images, image_ids, file_paths = load_images(args.data_dir, args.image_size)
    print(f"Loaded {len(images)} images.")

    features, feature_meta = extract_features(images, args.feature, args.random_seed)
    print(
        f"Feature={feature_meta['feature']} dims={feature_meta['dims']} "
        f"time={feature_meta['feature_time_s']:.3f}s"
    )

    print("\nStep 1: K-means clustering")
    kmeans = KMeans(n_clusters=args.k_clusters, random_state=args.random_seed, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    (
        labels,
        sources,
        weights,
        cluster_decisions,
        manual_images_clusters,
        manual_time_clusters_s,
    ) = bootstrap_cluster_labels(
        images=images,
        image_ids=image_ids,
        cluster_ids=cluster_ids,
        sample_per_cluster=args.sample_per_cluster,
        cluster_weight=args.cluster_weight,
        preview_dir=preview_cluster_dir,
        random_seed=args.random_seed,
    )

    write_cluster_decisions(run_dir / "cluster_bootstrap_labels.csv", cluster_decisions)

    truth_map: dict[int, int] = {}
    if args.ground_truth_csv:
        truth_map = load_ground_truth_csv(args.ground_truth_csv)
        print(f"Loaded ground-truth entries: {len(truth_map)}")

    practical_map: dict[int, int] = {}
    if args.practical_annotated_csv:
        practical_map = load_practical_annotated_csv(args.practical_annotated_csv)
        print(f"Loaded practical human-labeled entries: {len(practical_map)}")
        if args.use_practical_for_training:
            print("Practical labels mode: training + evaluation.")
        else:
            print(
                "Practical labels mode: evaluation-only "
                "(set --use-practical-for-training to enable corrections)."
            )

    if not truth_map and not practical_map:
        print(
            "Warning: no accuracy source provided. "
            "Use --ground-truth-csv or --practical-annotated-csv "
            "to record measured accuracy in summary/iteration logs."
        )

    boundary_annotations: list[BoundaryAnnotation] = []
    metrics: list[IterationMetric] = []

    total_boundary_labeled = 0
    previous_accuracy: float | None = None
    final_accuracy: float | None = None
    final_scope = "none"
    practical_target_mode = bool(practical_map)
    practical_stalled_iters = 0
    practical_stall_limit = 20

    try:
        final_predictions = np.zeros(len(images), dtype=np.int32)
        previous_predictions: np.ndarray | None = None

        iteration = 0
        while True:
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")
            print("Step 3/5: Train SVM and estimate uncertainty")

            model, train_mask = train_weighted_svm(features, labels, weights)
            preds_before, margins = predict_labels_and_margins(model, features)

            trusted_sources = ["boundary"]
            if args.use_practical_for_training:
                trusted_sources.append("practical")
            trusted_mask = np.isin(sources, trusted_sources)
            uncertain_idx = pick_low_margin_indices(margins, trusted_mask, args.boundary_batch)

            if len(uncertain_idx) == 0:
                print("No candidates left for boundary labeling this iteration.")
                added = 0
                ann: list[BoundaryAnnotation] = []
            else:
                added, ann = human_label_boundary_images(
                    iteration=iteration,
                    images=images,
                    image_ids=image_ids,
                    file_paths=file_paths,
                    selected_indices=uncertain_idx,
                    predicted_labels=preds_before,
                    margins=margins,
                    labels=labels,
                    sources=sources,
                    weights=weights,
                    trusted_weight=args.trusted_weight,
                    preview_dir=preview_boundary_dir,
                )

            # Do NOT auto-apply practical corrections when no manual boundary labels were added.
            # The user requested that when the run stops iterating, nothing in the training data
            # should be changed automatically. Practical labels remain evaluation-only unless
            # explicitly injected by a user action outside this automatic loop.
            # (No-op)

            boundary_annotations.extend(ann)
            total_boundary_labeled += added

            if added > 0:
                print("Step 5: Retrain with updated human labels")
                model, train_mask = train_weighted_svm(features, labels, weights)
                final_predictions, _ = predict_labels_and_margins(model, features)
            else:
                final_predictions = preds_before

            prediction_change = None
            if previous_predictions is not None:
                prediction_change = float(np.mean(final_predictions != previous_predictions))
            previous_predictions = final_predictions.copy()

            measured_acc, scope = evaluate_with_available_sources(
                final_predictions,
                image_ids,
                truth_map,
                practical_map,
            )

            improvement = None
            if measured_acc is not None and previous_accuracy is not None:
                improvement = measured_acc - previous_accuracy

            total_manual_images = manual_images_clusters + total_boundary_labeled
            total_manual_time_s = manual_time_clusters_s + total_boundary_labeled * 10.0

            metric = IterationMetric(
                iteration=iteration,
                train_size=int(np.sum(train_mask)),
                boundary_labeled_this_iter=added,
                total_boundary_labeled=total_boundary_labeled,
                total_manual_images=total_manual_images,
                total_manual_time_s=total_manual_time_s,
                measured_accuracy=measured_acc,
                accuracy_scope=scope,
                improvement=improvement,
            )
            metrics.append(metric)

            if measured_acc is not None:
                final_accuracy = measured_acc
                final_scope = scope
                print(
                    f"Measured accuracy={100.0 * measured_acc:.3f}% "
                    f"({scope}); improvement={'' if improvement is None else f'{100.0 * improvement:.3f}%'}"
                )
                previous_accuracy = measured_acc
            else:
                if prediction_change is None:
                    print("Measured accuracy unavailable this iteration.")
                else:
                    print(
                        "Measured accuracy unavailable this iteration; "
                        f"prediction-change={100.0 * prediction_change:.3f}%"
                    )

            target_reached = False
            if measured_acc is not None:
                target_reached = measured_acc >= args.target_accuracy
                if practical_map and len(practical_map) == PRACTICAL_GATE_LABEL_COUNT:
                    # Match the strict practical gate: accuracy must be strictly above threshold.
                    target_reached = measured_acc > max(
                        args.target_accuracy,
                        PRACTICAL_GATE_MIN_ACCURACY,
                    )

            if target_reached:
                print("Target accuracy reached. Stopping.")
                break

            stop_value = improvement if improvement is not None else prediction_change
            stop_label = (
                "accuracy improvement" if improvement is not None else "prediction-change"
            )
            if stop_value is not None and stop_value < args.min_improvement:
                print(
                    "Improvement threshold reached. Stopping because "
                    f"{stop_label} ({100.0 * stop_value:.3f}%) is below "
                    f"--min-improvement ({100.0 * args.min_improvement:.3f}%)."
                )
                break

            if practical_target_mode:
                if measured_acc is None:
                    raise SystemExit(
                        "Practical target mode is enabled, but measured accuracy is unavailable. "
                        "Check --practical-annotated-csv image_id overlap with dataset."
                    )

                if (improvement is None or improvement <= 0.0) and added == 0:
                    practical_stalled_iters += 1
                else:
                    practical_stalled_iters = 0

                if practical_stalled_iters >= practical_stall_limit:
                    raise SystemExit(
                        "Unable to reach target practical accuracy after repeated no-progress "
                        "iterations. Try providing manual boundary labels for hard samples."
                    )

                if iteration >= args.max_iters:
                    print(
                        f"Target not reached yet ({100.0 * measured_acc:.3f}%). "
                        f"Continuing beyond --max-iters={args.max_iters} until target is met."
                    )
                continue

            if iteration >= args.max_iters:
                print(f"Reached --max-iters={args.max_iters}. Stopping.")
                break

        if final_accuracy is None:
            measured_acc, scope = evaluate_with_available_sources(
                final_predictions,
                image_ids,
                truth_map,
                practical_map,
            )
            if measured_acc is not None:
                final_accuracy = measured_acc
                final_scope = scope
                print(
                    f"Final measured accuracy={100.0 * measured_acc:.3f}% "
                    f"({scope})"
                )

        write_boundary_annotations(run_dir / "boundary_labels.csv", boundary_annotations)
        write_iteration_log(run_dir / "iteration_log.csv", metrics)
        write_predictions(
            run_dir / "predictions_final.csv",
            image_ids=image_ids,
            predicted_labels=final_predictions,
            training_labels=labels,
            sources=sources,
            weights=weights,
        )
        save_summary(
            run_dir / "summary.json",
            args=args,
            feature_meta=feature_meta,
            total_images=len(images),
            cluster_decisions=cluster_decisions,
            manual_images_clusters=manual_images_clusters,
            manual_time_clusters_s=manual_time_clusters_s,
            total_boundary_labeled=total_boundary_labeled,
            iteration_metrics=metrics,
            final_accuracy=final_accuracy,
            final_scope=final_scope,
        )

        if practical_map:
            practical_acc, _ = evaluate_with_practical(
                final_predictions,
                image_ids,
                practical_map,
            )
            if practical_acc is not None:
                enforce_practical_accuracy_gate(
                    accuracy=practical_acc,
                    labeled_rows=len(practical_map),
                    context="Pipeline practical quality gate",
                )
                if len(practical_map) == PRACTICAL_GATE_LABEL_COUNT:
                    print(
                        f"Practical quality gate: PASS (>"
                        f"{100.0 * PRACTICAL_GATE_MIN_ACCURACY:.2f}% on "
                        f"{PRACTICAL_GATE_LABEL_COUNT} labeled images)"
                    )

        print("\nSaved outputs:")
        print(f"- {run_dir / 'cluster_bootstrap_labels.csv'}")
        print(f"- {run_dir / 'boundary_labels.csv'}")
        print(f"- {run_dir / 'iteration_log.csv'}")
        print(f"- {run_dir / 'predictions_final.csv'}")
        print(f"- {run_dir / 'summary.json'}")

        total_manual_images = manual_images_clusters + total_boundary_labeled
        total_manual_time_s = manual_time_clusters_s + total_boundary_labeled * 10.0
        print("\nManual effort summary:")
        print(f"- Manual images handled: {total_manual_images}")
        print(f"- Manual time (seconds): {total_manual_time_s:.1f}")
        print(f"- Manual time (hours)  : {total_manual_time_s / 3600.0:.3f}")

    except KeyboardInterrupt:
        print("\nRun stopped by user.")
        print(f"Partial outputs are in: {run_dir}")


if __name__ == "__main__":
    main()
