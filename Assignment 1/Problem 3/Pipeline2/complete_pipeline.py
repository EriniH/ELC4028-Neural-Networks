#!/usr/bin/env python3
"""
Pipeline 2: Manual Seed Labelling with Augmentation and Self-Training Refinement
Complete end-to-end implementation following the task specification.

This pipeline:
1. Loads/validates a 300-image seed set with manual labels
2. Augments the seed set (4 copies per image)
3. Trains an SVM classifier
4. Iteratively refines by:
   - Selecting 20 most uncertain images for manual annotation
   - Pseudo-labeling 50 high-confidence images per class
   - Retraining and evaluating
5. Stops when accuracy >= 99% or improvement < 0.1%
"""

import os
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from collections import defaultdict

# =========================================================
# CONFIGURATION
# =========================================================
SEED_PATH = "TrainingSeed300/*/*.bmp"
FULL_DATASET_PATH = "Indian_Digits_Train/*.bmp"
GROUND_TRUTH_PATH = "PracticalGroundTruth500/*/*.bmp"
SEED_ROOT = "TrainingSeed300"

MANUAL_LABEL_ROOT = "uncertain_for_annotation"
HISTORY_FILE = os.path.join(MANUAL_LABEL_ROOT, "iteration_metrics.tsv")

IMAGE_SIZE = 28
SEED_WEIGHT = 100
AUG_WEIGHT = 1
PSEUDO_WEIGHT = 1

TARGET_ACCURACY = 0.99
MIN_IMPROVEMENT = 0.001
N_UNCERTAIN_PER_ITER = 20
N_PSEUDO_PER_CLASS = 50
PSEUDO_MARGIN_PERCENTILE = 75

SEED_POOL_SIZE = 350
SEED_TARGET_PER_CLASS = 30
SEED_RANDOM_STATE = 42

# Deterministic augmentation settings (Step 2 in task specification)
ROTATION_ANGLES = (-5, 5)
SHIFT_PIXELS = 2
NOISE_STD = 0.03

# =========================================================
# UTILITIES
# =========================================================
def load_image_vector(path):
    """Load, resize, and normalize a grayscale image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return (img.astype(np.float32) / 255.0).flatten()


def extract_label(path):
    """Extract digit label from folder structure."""
    return int(os.path.basename(os.path.dirname(path)))


def extract_index(path):
    """Extract image index from filename."""
    return int(os.path.basename(path).split(".")[0])


def validate_seed_dataset(X, y):
    """Validate the expected 300-image seed set and report its class balance."""
    if len(X) != 300:
        raise ValueError(f"Expected 300 seed images, but found {len(X)}")

    classes, counts = np.unique(y, return_counts=True)
    distribution = {int(cls): int(count) for cls, count in zip(classes, counts)}
    print(f"Seed class distribution: {distribution}")
    if len(classes) != 10:
        print("Warning: seed set does not contain all 10 digit classes.")
    expected = {digit: SEED_TARGET_PER_CLASS for digit in range(10)}
    if distribution != expected:
        raise ValueError(
            "Seed set is not balanced at 30 images per class. "
            f"Expected {expected}, found {distribution}"
        )


# =========================================================
# AUGMENTATION
# =========================================================
def rotate_image(img, angle_deg):
    """Rotate image around center while keeping 28x28 output size."""
    h, w = img.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shift_image(img, dx, dy):
    """Translate image by integer pixel offsets (dx, dy)."""
    h, w = img.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def add_gaussian_noise(img, std=NOISE_STD):
    """Add low-level Gaussian noise in normalized space [0, 1]."""
    noisy = np.clip(img + np.random.normal(0.0, std, img.shape).astype(np.float32), 0.0, 1.0)
    return noisy


def augment_dataset(X, y):
    """
    Augment dataset by creating multiple transformed copies.
    
    Args:
        X: array of flattened images
        y: array of labels
    Returns:
        X_aug, y_aug: augmented features and labels
    """
    X_aug, y_aug = [], []
    for i in range(len(X)):
        img = X[i].reshape(IMAGE_SIZE, IMAGE_SIZE)

        # Two rotation copies: -5 and +5 degrees.
        for angle in ROTATION_ANGLES:
            aug = rotate_image(img, angle)
            X_aug.append(aug.flatten())
            y_aug.append(y[i])

        # One Gaussian-noise copy.
        aug_noise = add_gaussian_noise(img)
        X_aug.append(aug_noise.flatten())
        y_aug.append(y[i])

        # Four directional shifts: up, down, left, right.
        shifts = [(0, -SHIFT_PIXELS), (0, SHIFT_PIXELS), (-SHIFT_PIXELS, 0), (SHIFT_PIXELS, 0)]
        for dx, dy in shifts:
            aug_shift = shift_image(img, dx, dy)
            X_aug.append(aug_shift.flatten())
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


# =========================================================
# DATA LOADING
# =========================================================
def load_seed(seed_path=SEED_PATH):
    """Load all seed images."""
    paths = sorted(glob.glob(seed_path), key=extract_index)
    X, y, indices = [], [], set()

    for p in tqdm(paths, desc="Loading seed"):
        idx = extract_index(p)
        indices.add(idx)
        X.append(load_image_vector(p))
        y.append(extract_label(p))

    return np.array(X), np.array(y), indices


def load_full_paths(full_path=FULL_DATASET_PATH):
    """Load and sort full dataset paths without decoding image pixels."""
    return sorted(glob.glob(full_path), key=extract_index)


def load_full_dataset(full_path=FULL_DATASET_PATH, paths=None):
    """Load all 10,000 images."""
    if paths is None:
        paths = sorted(glob.glob(full_path), key=extract_index)
    X = np.array([load_image_vector(p) for p in tqdm(paths, desc="Loading full dataset")])
    return paths, X


def initialize_seed_pool(paths_full, seed_root=SEED_ROOT, pool_size=SEED_POOL_SIZE):
    """Create an oversized random seed pool for manual labeling under TrainingSeed300."""
    if os.path.exists(seed_root):
        shutil.rmtree(seed_root)
    os.makedirs(seed_root, exist_ok=True)
    for digit in range(10):
        os.makedirs(os.path.join(seed_root, str(digit)), exist_ok=True)

    if pool_size <= 300:
        raise ValueError(f"SEED_POOL_SIZE must be > 300, got {pool_size}")
    if pool_size > len(paths_full):
        raise ValueError(f"SEED_POOL_SIZE={pool_size} exceeds dataset size {len(paths_full)}")

    rng = np.random.default_rng(SEED_RANDOM_STATE)
    chosen_indices = rng.choice(len(paths_full), size=pool_size, replace=False)
    chosen_paths = [paths_full[i] for i in chosen_indices]

    entries = []
    for src in chosen_paths:
        filename = os.path.basename(src)
        idx = extract_index(src)
        dst = os.path.join(seed_root, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        entries.append({'index': idx, 'filename': filename, 'source_path': src})

    return entries


def get_seed_label_progress(seed_entries, seed_root=SEED_ROOT):
    """Count labeled images from the sampled pool by class."""
    pool_filenames = {item['filename'] for item in seed_entries}
    per_class = {digit: [] for digit in range(10)}

    for digit in range(10):
        digit_dir = os.path.join(seed_root, str(digit))
        if not os.path.exists(digit_dir):
            continue
        for p in glob.glob(os.path.join(digit_dir, "*.bmp")):
            filename = os.path.basename(p)
            if filename in pool_filenames:
                per_class[digit].append(p)

    counts = {digit: len(per_class[digit]) for digit in range(10)}
    return per_class, counts


def wait_for_seed_labeling(seed_entries, seed_root=SEED_ROOT):
    """Wait until each class has at least 30 manually labeled seed candidates."""
    while True:
        per_class, counts = get_seed_label_progress(seed_entries, seed_root)
        summary = ", ".join([f"{d}:{counts[d]}" for d in range(10)])
        print(f"Seed labeling progress by class -> {summary}")

        if all(counts[d] >= SEED_TARGET_PER_CLASS for d in range(10)):
            return per_class

        user_choice = input(
            "Need at least 30 labeled images per class in TrainingSeed300/0..9. "
            "Copy or move sampled images from TrainingSeed300 root into those class folders, "
            "then press Enter (or type 'q' to quit): "
        ).strip().lower()
        if user_choice == "q":
            raise KeyboardInterrupt("Seed labeling aborted by user.")


def build_balanced_seed_from_labels(per_class_paths):
    """Randomly truncate labeled class folders to exactly 30 images per class."""
    rng = np.random.default_rng(SEED_RANDOM_STATE)

    for digit in range(10):
        candidates = per_class_paths[digit]
        if len(candidates) < SEED_TARGET_PER_CLASS:
            raise ValueError(
                f"Class {digit} has only {len(candidates)} labels; need at least {SEED_TARGET_PER_CLASS}"
            )
        selected_idx = rng.choice(len(candidates), size=SEED_TARGET_PER_CLASS, replace=False)
        keep = {os.path.basename(candidates[i]) for i in selected_idx}

        digit_dir = os.path.join(SEED_ROOT, str(digit))
        for p in glob.glob(os.path.join(digit_dir, "*.bmp")):
            if os.path.basename(p) not in keep:
                os.remove(p)


def ensure_balanced_seed_from_random_pool(paths_full):
    """Ensure TrainingSeed300 is created from a random oversized manually labeled pool."""
    print("\nPreparing random oversized seed pool for manual labeling...")
    seed_entries = initialize_seed_pool(paths_full)
    print(
        f"Sampled {len(seed_entries)} images into TrainingSeed300 root. "
        "Label by copying or moving them into TrainingSeed300/0..9."
    )
    print(
        f"Once each class has at least {SEED_TARGET_PER_CLASS} images, "
        "the script will randomly truncate to 30 per class."
    )

    per_class_paths = wait_for_seed_labeling(seed_entries)
    build_balanced_seed_from_labels(per_class_paths)
    print("Built TrainingSeed300 from random oversized pool with exact 30 images/class.")


def load_ground_truth(gt_path=GROUND_TRUTH_PATH):
    """Load practical ground truth (500 labeled images)."""
    paths = sorted(glob.glob(gt_path), key=extract_index)
    idx, labels = [], []

    for p in paths:
        idx.append(extract_index(p))
        labels.append(extract_label(p))

    return np.array(idx), np.array(labels)


# =========================================================
# MODEL TRAINING
# =========================================================
def train_svm(X, y, weights):
    """Train SVM classifier with sample weights."""
    model = SVC(kernel="rbf", decision_function_shape="ovo", probability=True, random_state=42)
    model.fit(X, y, sample_weight=weights)
    return model


# =========================================================
# EVALUATION
# =========================================================
def evaluate_on_gt(pred_map, gt_idx, gt_y):
    """Evaluate accuracy on ground truth set."""
    correct = 0
    missing = 0
    for idx, true_label in zip(gt_idx, gt_y):
        pred_label = pred_map.get(idx)
        if pred_label is None:
            missing += 1
            continue
        if pred_label == true_label:
            correct += 1
    
    evaluated_count = len(gt_y) - missing
    acc = correct / evaluated_count if evaluated_count > 0 else 0.0
    return acc, correct, evaluated_count, missing


def evaluate_with_oracle(predicted_labels):
    """Evaluate accuracy using the hidden oracle."""
    try:
        from check_accuracy import check_accuracy
        accuracy, n_correct, n_total = check_accuracy(predicted_labels)
        return accuracy, n_correct, n_total
    except ImportError:
        print("Warning: check_accuracy not available. Oracle evaluation skipped.")
        return None, None, None


# =========================================================
# ACTIVE LEARNING: SELECT UNCERTAIN IMAGES
# =========================================================
def compute_margins(probs):
    """Compute prediction margins (confidence scores)."""
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def select_uncertain(paths, probs, k=20, incorporated=None):
    """
    Select k most uncertain (lowest margin) images for manual annotation.
    
    Args:
        paths: list of image paths
        probs: prediction probabilities from model
        k: number of images to select
        incorporated: set of already-incorporated image indices
    
    Returns:
        list of (path, margin, predicted_label) tuples for k most uncertain images
    """
    if incorporated is None:
        incorporated = set()
    
    margins = compute_margins(probs)
    preds = np.argmax(probs, axis=1)
    
    # Filter out already incorporated images
    candidates = []
    for i, (path, margin, pred) in enumerate(zip(paths, margins, preds)):
        idx = extract_index(path)
        if idx not in incorporated:
            candidates.append((path, margin, pred, idx))
    
    # Sort by confidence (smallest margin first) and take top k
    candidates.sort(key=lambda x: x[1])
    return candidates[:k]


# =========================================================
# PSEUDO-LABELING: SELECT HIGH-CONFIDENCE PREDICTIONS
# =========================================================
def select_pseudo(paths, probs, preds, incorporated):
    """
    Select high-confidence pseudo-labeled images.
    
    Args:
        paths: list of image paths
        probs: prediction probabilities
        preds: predicted labels
        incorporated: set of already-incorporated indices
    
    Returns:
        selected_data: list of selected pseudo-labeled samples
        global_threshold: margin threshold used for selection
        rejected_by_threshold: count rejected within top-50/class window
        candidate_count: total candidates considered before thresholding
        topk_considered_count: total top-50 windows considered across classes
    """
    margins = compute_margins(probs)
    
    # Find all candidates not yet incorporated
    candidates = []
    for i, (path, margin, pred) in enumerate(zip(paths, margins, preds)):
        idx = extract_index(path)
        if idx not in incorporated:
            candidates.append({
                'path': path,
                'index': idx,
                'predicted': int(pred),
                'margin': margin
            })
    
    if not candidates:
        return [], None, 0, 0, 0

    # Group by predicted class and select top-N per class above the global threshold.
    by_class = defaultdict(list)
    for candidate in candidates:
        by_class[candidate['predicted']].append(candidate)
    
    selected_data = []
    global_threshold = float(np.percentile([candidate['margin'] for candidate in candidates], PSEUDO_MARGIN_PERCENTILE))
    rejected_by_threshold = 0
    topk_considered_count = 0
    for digit in range(10):
        class_images = by_class[digit]

        # Strict "most confident" ordering, then apply threshold in top-N window.
        class_images.sort(key=lambda x: x['margin'], reverse=True)
        top_window = class_images[:N_PSEUDO_PER_CLASS]
        topk_considered_count += len(top_window)
        eligible = [img for img in top_window if img['margin'] >= global_threshold]
        rejected_by_threshold += (len(top_window) - len(eligible))
        selected_data.extend(eligible)

    return selected_data, global_threshold, rejected_by_threshold, len(candidates), topk_considered_count


# =========================================================
# MANUAL LABEL LOADING
# =========================================================
def load_manual_labels(incorporated, manual_label_root=MANUAL_LABEL_ROOT):
    """Load all manually labeled images from all iteration folders."""
    X, y = [], []
    manual_count = 0

    iter_dirs = sorted(glob.glob(os.path.join(manual_label_root, "iter_*")))

    for d in iter_dirs:
        for digit in range(10):
            folder = os.path.join(d, str(digit))
            if not os.path.exists(folder):
                continue

            for p in glob.glob(os.path.join(folder, "*.bmp")):
                idx = extract_index(p)
                if idx in incorporated:
                    continue
                incorporated.add(idx)
                X.append(load_image_vector(p))
                y.append(digit)
                manual_count += 1

    return np.array(X) if X else np.array([]).reshape(0, IMAGE_SIZE * IMAGE_SIZE), \
           np.array(y) if y else np.array([], dtype=int), \
           manual_count


def load_iteration_manual_labels(iteration_num, incorporated, manual_label_root=MANUAL_LABEL_ROOT):
    """Load manually labeled images for a specific iteration only."""
    X, y = [], []
    manual_count = 0

    iter_dir = os.path.join(manual_label_root, f"iter_{iteration_num:03d}")
    for digit in range(10):
        folder = os.path.join(iter_dir, str(digit))
        if not os.path.exists(folder):
            continue

        for p in glob.glob(os.path.join(folder, "*.bmp")):
            idx = extract_index(p)
            if idx in incorporated:
                continue
            incorporated.add(idx)
            X.append(load_image_vector(p))
            y.append(digit)
            manual_count += 1

    return np.array(X) if X else np.array([]).reshape(0, IMAGE_SIZE * IMAGE_SIZE), \
           np.array(y) if y else np.array([], dtype=int), \
           manual_count


# =========================================================
# ITERATION MANAGEMENT
# =========================================================
def get_next_iteration_number(manual_label_root=MANUAL_LABEL_ROOT):
    """Get the next iteration number."""
    existing_iters = []
    for path in glob.glob(os.path.join(manual_label_root, "iter_*")):
        name = os.path.basename(path)
        try:
            existing_iters.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return (max(existing_iters) + 1) if existing_iters else 1


def create_annotation_folder(uncertain_samples, iteration_num, manual_label_root=MANUAL_LABEL_ROOT):
    """Create folder structure for manual annotation of uncertain samples."""
    iter_dir = os.path.join(manual_label_root, f"iter_{iteration_num:03d}")
    os.makedirs(iter_dir, exist_ok=True)

    # Create digit folders
    for digit in range(10):
        os.makedirs(os.path.join(iter_dir, str(digit)), exist_ok=True)

    # Copy uncertain samples and create annotation list
    annotation_list = []
    for path, margin, pred, idx in uncertain_samples:
        src = path
        dst = os.path.join(iter_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        annotation_list.append({
            'index': idx,
            'filename': os.path.basename(src),
            'predicted_label': pred,
            'margin': float(margin)
        })

    # Save annotation list
    with open(os.path.join(iter_dir, "annotation_list.txt"), "w") as f:
        f.write("index\tfilename\tpredicted_label\tmargin\n")
        for item in annotation_list:
            f.write(f"{item['index']}\t{item['filename']}\t{item['predicted_label']}\t{item['margin']:.6f}\n")

    return iter_dir, annotation_list


def save_pseudo_labels(pseudo_data, iteration_num, manual_label_root=MANUAL_LABEL_ROOT):
    """Persist pseudo-labeled samples so they are reused in later iterations."""
    iter_dir = os.path.join(manual_label_root, f"iter_{iteration_num:03d}")
    pseudo_dir = os.path.join(iter_dir, "pseudo_labels")
    os.makedirs(pseudo_dir, exist_ok=True)

    out_file = os.path.join(pseudo_dir, "pseudo_label_list.txt")
    with open(out_file, "w") as f:
        f.write("index\tfilename\tpseudo_label\tmargin\n")
        for item in pseudo_data:
            f.write(
                f"{item['index']}\t{os.path.basename(item['path'])}\t{item['predicted']}\t{item['margin']:.6f}\n"
            )


def load_pseudo_labels(incorporated, paths_full, manual_label_root=MANUAL_LABEL_ROOT):
    """Load pseudo-labeled images saved from previous iterations."""
    X, y = [], []
    pseudo_count = 0
    index_to_path = {extract_index(p): p for p in paths_full}

    iter_dirs = sorted(glob.glob(os.path.join(manual_label_root, "iter_*")))
    for d in iter_dirs:
        pseudo_file = os.path.join(d, "pseudo_labels", "pseudo_label_list.txt")
        if not os.path.exists(pseudo_file):
            continue

        with open(pseudo_file, "r") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    idx = int(parts[0])
                    label = int(parts[2])
                except ValueError:
                    continue

                if idx in incorporated:
                    continue

                path = index_to_path.get(idx)
                if path is None:
                    continue

                incorporated.add(idx)
                X.append(load_image_vector(path))
                y.append(label)
                pseudo_count += 1

    return np.array(X) if X else np.array([]).reshape(0, IMAGE_SIZE * IMAGE_SIZE), \
           np.array(y) if y else np.array([], dtype=int), \
           pseudo_count


def get_annotation_progress(iteration_num, manual_label_root=MANUAL_LABEL_ROOT):
    """Return expected vs completed manual labels for a given iteration folder."""
    iter_dir = os.path.join(manual_label_root, f"iter_{iteration_num:03d}")
    annotation_file = os.path.join(iter_dir, "annotation_list.txt")
    if not os.path.exists(annotation_file):
        return 0, 0

    expected = 0
    with open(annotation_file, "r") as f:
        next(f, None)
        for line in f:
            if line.strip():
                expected += 1

    labeled_filenames = set()
    for digit in range(10):
        digit_dir = os.path.join(iter_dir, str(digit))
        if not os.path.exists(digit_dir):
            continue
        for p in glob.glob(os.path.join(digit_dir, "*.bmp")):
            labeled_filenames.add(os.path.basename(p))

    return expected, len(labeled_filenames)


def find_pending_annotation_iteration(manual_label_root=MANUAL_LABEL_ROOT):
    """Return the earliest iteration with unfinished manual labels, else None."""
    iter_dirs = sorted(glob.glob(os.path.join(manual_label_root, "iter_*")))
    for d in iter_dirs:
        name = os.path.basename(d)
        try:
            iter_num = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue

        expected, labeled = get_annotation_progress(iter_num, manual_label_root)
        if expected > 0 and labeled < expected:
            return iter_num, expected, labeled

    return None


# =========================================================
# HISTORY TRACKING
# =========================================================
def load_history(history_file=HISTORY_FILE):
    """Load iteration history."""
    history = {}
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    iter_id = int(parts[0])
                    practical_acc = float(parts[1])
                    oracle_acc = float(parts[2]) if len(parts) > 2 else None
                    history[iter_id] = {
                        'practical': practical_acc,
                        'oracle': oracle_acc,
                        'manual_new': int(parts[3]) if len(parts) > 3 else 0,
                        'pseudo_added': int(parts[4]) if len(parts) > 4 else 0,
                        'pseudo_rejected': int(parts[5]) if len(parts) > 5 else 0,
                        'pseudo_candidates': int(parts[6]) if len(parts) > 6 else 0,
                        'pseudo_threshold': float(parts[7]) if len(parts) > 7 else np.nan,
                        'pseudo_topk_considered': int(parts[8]) if len(parts) > 8 else 0,
                    }
                except ValueError:
                    continue
    return history


def save_history(history, history_file=HISTORY_FILE):
    """Save iteration history."""
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w") as f:
        f.write(
            "iteration\tpractical_accuracy\toracle_accuracy\tmanual_new"
            "\tpseudo_added\tpseudo_rejected\tpseudo_candidates\tpseudo_threshold\tpseudo_topk_considered\n"
        )
        for iter_id in sorted(history.keys()):
            practical_acc = history[iter_id].get('practical', 0.0)
            oracle_acc = history[iter_id].get('oracle', 0.0)
            manual_new = history[iter_id].get('manual_new', 0)
            pseudo_added = history[iter_id].get('pseudo_added', 0)
            pseudo_rejected = history[iter_id].get('pseudo_rejected', 0)
            pseudo_candidates = history[iter_id].get('pseudo_candidates', 0)
            pseudo_threshold = history[iter_id].get('pseudo_threshold', np.nan)
            pseudo_topk_considered = history[iter_id].get('pseudo_topk_considered', 0)
            threshold_str = f"{pseudo_threshold:.6f}" if np.isfinite(pseudo_threshold) else "nan"
            f.write(
                f"{iter_id}\t{practical_acc:.6f}\t{oracle_acc:.6f}\t{manual_new}"
                f"\t{pseudo_added}\t{pseudo_rejected}\t{pseudo_candidates}\t{threshold_str}\t{pseudo_topk_considered}\n"
            )


def print_block_diagram():
    """Print a compact block diagram with exact configured values."""
    print("\nBLOCK DIAGRAM")
    print("[Seed 300, w=100]")
    print(
        f"  -> [Augment x7/image: rotate({ROTATION_ANGLES[0]},{ROTATION_ANGLES[1]}), "
        f"noise(std={NOISE_STD}), shift({SHIFT_PIXELS}px U/D/L/R), w={AUG_WEIGHT}]"
    )
    print("  -> [Train SVM-1: RBF, OVO]")
    print(f"  -> [Uncertain select: {N_UNCERTAIN_PER_ITER} smallest margins -> manual label, w={SEED_WEIGHT}]")
    print(
        f"  -> [Pseudo select: top {N_PSEUDO_PER_CLASS}/class with margin >= p{PSEUDO_MARGIN_PERCENTILE}, w={PSEUDO_WEIGHT}]"
    )
    print("  -> [Retrain SVM-k]")
    print(
        f"  -> [Evaluate practical/oracle, stop if acc >= {TARGET_ACCURACY:.2f} or improvement < {MIN_IMPROVEMENT:.3f}]"
    )


def print_compact_iteration_table(history):
    """Print a compact per-iteration table required by the task."""
    if not history:
        return

    print("\nITERATION TABLE")
    print("iter | practical% | oracle% | manual+ | pseudo+ | reject(top50) | threshold")
    print("-----+------------+---------+---------+---------+---------------+----------")
    for iter_id in sorted(history.keys()):
        row = history[iter_id]
        threshold = row.get('pseudo_threshold', np.nan)
        threshold_str = f"{threshold:.4f}" if np.isfinite(threshold) else "n/a"
        print(
            f"{iter_id:>4} | {100.0 * row.get('practical', 0.0):>10.2f} | "
            f"{100.0 * row.get('oracle', 0.0):>7.2f} | {row.get('manual_new', 0):>7} | "
            f"{row.get('pseudo_added', 0):>7} | {row.get('pseudo_rejected', 0):>13} | {threshold_str:>8}"
        )


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_iteration(iteration_num, seed_features, seed_labels, seed_incorporated,
                  paths_full, X_full, gt_indices, gt_labels, history, manual_label_root=MANUAL_LABEL_ROOT):
    """
    Run a single iteration of the active learning pipeline.
    
    Returns:
        practical_acc, oracle_acc, should_continue, aborted_by_user, stop_reason
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*70}")

    # =========================================================
    # Load all data incorporated so far
    # =========================================================
    incorporated = seed_incorporated.copy()
    
    X_train = seed_features.copy()
    y_train = seed_labels.copy()
    weights_train = np.full(len(seed_features), SEED_WEIGHT)

    # Add deterministic augmented seed data
    X_aug, y_aug = augment_dataset(seed_features, seed_labels)
    X_train = np.vstack([X_train, X_aug])
    y_train = np.concatenate([y_train, y_aug])
    weights_train = np.concatenate([weights_train, np.full(len(X_aug), AUG_WEIGHT)])

    # Load manually labeled images from previous iterations
    X_manual, y_manual, manual_count = load_manual_labels(incorporated, manual_label_root)
    if len(X_manual) > 0:
        X_train = np.vstack([X_train, X_manual])
        y_train = np.concatenate([y_train, y_manual])
        weights_train = np.concatenate([weights_train, np.full(len(X_manual), SEED_WEIGHT)])
        print(f"Loaded {manual_count} manually labeled images")

    # Pseudo-labels are refreshed every iteration to avoid locking early mistakes.
    pseudo_prev_count = 0

    print(
        f"Training set size: {len(X_train)} "
        f"({len(seed_features)} seed + {len(X_aug)} augmented + {manual_count} manual + {pseudo_prev_count} pseudo-prev)"
    )

    # =========================================================
    # Train SVM (stage 1)
    # =========================================================
    print("Training SVM (stage 1)...")
    model = train_svm(X_train, y_train, weights_train)
    print("SVM stage-1 training completed.")

    # =========================================================
    # Predict on full dataset (stage 1)
    # =========================================================
    print("Making stage-1 predictions on full dataset...")
    probs_full = model.predict_proba(X_full)
    preds_full = np.argmax(probs_full, axis=1)

    # Export smallest-margin uncertain samples for manual annotation first
    print(f"Selecting {N_UNCERTAIN_PER_ITER} most uncertain samples for manual annotation...")
    uncertain_samples = select_uncertain(paths_full, probs_full, k=N_UNCERTAIN_PER_ITER, incorporated=incorporated)
    reserved_uncertain_indices = set()
    manual_new_count = 0
    if uncertain_samples:
        # Reserve uncertain IDs only for pseudo-label selection in this iteration.
        for _, _, _, idx in uncertain_samples:
            reserved_uncertain_indices.add(idx)

        iter_dir, annotation_list = create_annotation_folder(uncertain_samples, iteration_num, manual_label_root)
        print(f"Created annotation folder: {iter_dir}")
        print(f"Action required: Manually label {len(uncertain_samples)} images in the digit subfolders")
        print(f"Expected time: {len(uncertain_samples)} images × 10 seconds = {len(uncertain_samples) * 10} seconds (~{len(uncertain_samples) * 10 / 60:.1f} minutes)")

        # Wait for current iteration manual labels so stage-2 retraining uses the updated set.
        while True:
            expected, labeled = get_annotation_progress(iteration_num, manual_label_root)
            if expected > 0 and labeled >= expected:
                break
            user_choice = input(
                f"Labeling progress for iter_{iteration_num:03d}: {labeled}/{expected}. "
                "Finish labeling, then press Enter to continue (or type 'q' to quit): "
            ).strip().lower()
            if user_choice == "q":
                print("Stopping by user request before stage-2 retraining.")
                return None, None, False, True, "user quit during boundary manual labeling"

        # Add newly labeled boundary samples before pseudo-label selection/retraining.
        X_manual_new, y_manual_new, manual_new_count = load_iteration_manual_labels(
            iteration_num, incorporated, manual_label_root
        )
        if len(X_manual_new) > 0:
            X_train = np.vstack([X_train, X_manual_new])
            y_train = np.concatenate([y_train, y_manual_new])
            weights_train = np.concatenate([weights_train, np.full(len(X_manual_new), SEED_WEIGHT)])
            print(f"Loaded {manual_new_count} newly labeled uncertain images")
    else:
        print("Warning: No uncertain samples available to export.")

    # Select pseudo-labeled samples from the same stage-1 model and add them now.
    # Do not persist pseudo-labels across iterations; they are re-estimated next round.
    print("Selecting high-confidence pseudo-labeled samples...")
    incorporated_for_pseudo = incorporated.union(reserved_uncertain_indices)
    pseudo_data_new, global_threshold, rejected_by_threshold, pseudo_candidate_count, pseudo_topk_considered = select_pseudo(
        paths_full, probs_full, preds_full, incorporated_for_pseudo
    )
    if global_threshold is not None:
        threshold_summary = f"margin >= p{PSEUDO_MARGIN_PERCENTILE} ({global_threshold:.6f})"
    else:
        threshold_summary = "none"
    print(
        f"Selected {len(pseudo_data_new)} new pseudo-labeled images "
        f"(threshold: {threshold_summary}, rejected by threshold: {rejected_by_threshold}/{pseudo_topk_considered} top-50 slots)"
    )

    if pseudo_data_new:
        X_pseudo_new = np.array([load_image_vector(item['path']) for item in pseudo_data_new])
        y_pseudo_new = np.array([item['predicted'] for item in pseudo_data_new], dtype=int)
        X_train = np.vstack([X_train, X_pseudo_new])
        y_train = np.concatenate([y_train, y_pseudo_new])
        weights_train = np.concatenate([weights_train, np.full(len(X_pseudo_new), PSEUDO_WEIGHT)])
        for item in pseudo_data_new:
            incorporated.add(item['index'])

    # Final retrain once per iteration after all newly added labels are merged.
    if manual_new_count > 0 or pseudo_data_new:
        print(
            "Retraining SVM (stage 2) with newly added labels "
            f"(manual={manual_new_count}, pseudo={len(pseudo_data_new)})..."
        )
        model = train_svm(X_train, y_train, weights_train)
        print("SVM stage-2 training completed.")

        probs_full = model.predict_proba(X_full)
        preds_full = np.argmax(probs_full, axis=1)

    pred_map_full = {extract_index(p): int(y) for p, y in zip(paths_full, preds_full)}

    # =========================================================
    # Evaluate on practical ground truth (500)
    # =========================================================
    print("Evaluating on practical ground truth...")
    practical_acc, correct, evaluated_count, missing = evaluate_on_gt(pred_map_full, gt_indices, gt_labels)
    print(f"Practical Accuracy = {practical_acc*100:.2f}% ({correct}/{evaluated_count})")

    # =========================================================
    # Evaluate with oracle (if available)
    # =========================================================
    oracle_acc, oracle_correct, oracle_total = evaluate_with_oracle(preds_full)
    if oracle_acc is not None:
        print(f"Oracle Accuracy = {oracle_acc*100:.2f}% ({oracle_correct}/{oracle_total})")
    else:
        oracle_acc = practical_acc  # fallback

    # =========================================================
    # Save history entry
    # =========================================================
    history[iteration_num] = {
        'practical': practical_acc,
        'oracle': oracle_acc,
        'manual_new': int(manual_new_count),
        'pseudo_added': int(len(pseudo_data_new)),
        'pseudo_rejected': int(rejected_by_threshold),
        'pseudo_candidates': int(pseudo_candidate_count),
        'pseudo_threshold': float(global_threshold) if global_threshold is not None else np.nan,
        'pseudo_topk_considered': int(pseudo_topk_considered),
    }

    # =========================================================
    # Check stopping conditions
    # =========================================================
    use_oracle_for_stop = oracle_acc is not None
    metric_name = 'oracle' if use_oracle_for_stop else 'practical'
    current_metric = oracle_acc if use_oracle_for_stop else practical_acc
    prev_metric = history.get(iteration_num - 1, {}).get(metric_name)
    improvement = None
    if prev_metric is not None:
        improvement = current_metric - prev_metric

    print(f"\n--- Iteration {iteration_num} Summary ---")
    print(f"Practical Accuracy: {practical_acc*100:.2f}%")
    if oracle_acc is not None:
        print(f"Oracle Accuracy: {oracle_acc*100:.2f}%")
    if improvement is not None:
        print(f"Improvement vs Iteration {iteration_num-1} ({metric_name}): {improvement*100:+.2f}%")

    stop_by_target = (current_metric >= TARGET_ACCURACY)
    stop_by_convergence = (improvement is not None) and (improvement < MIN_IMPROVEMENT)
    should_continue = not (stop_by_target or stop_by_convergence)
    stop_reason = "continue"

    if stop_by_target:
        stop_reason = (
            f"target reached on {metric_name}: {current_metric*100:.2f}% >= {TARGET_ACCURACY*100:.2f}%"
        )
        print(
            f"\nSTOPPING: Target accuracy reached "
            f"({metric_name} {current_metric*100:.2f}% >= {TARGET_ACCURACY*100:.2f}%)"
        )
    elif stop_by_convergence:
        stop_reason = (
            f"convergence on {metric_name}: improvement {improvement*100:.2f}% < {MIN_IMPROVEMENT*100:.2f}%"
        )
        print(
            f"\nSTOPPING: Convergence reached on {metric_name} accuracy "
            f"(improvement {improvement*100:.2f}% < {MIN_IMPROVEMENT*100:.2f}%)"
        )
    else:
        print(f"\n→ CONTINUING: Next iteration needed")

    if should_continue:
        print("After labeling, rerun this script to continue to the next iteration.")

    save_history(history)
    return practical_acc, oracle_acc, should_continue, False, stop_reason


def main():
    """Main pipeline execution."""
    print("="*70)
    print("PIPELINE 2: Manual Seed + Active Learning + Pseudo-Labeling")
    print("="*70)
    
    # =========================================================
    # Initialize history
    # =========================================================
    os.makedirs(MANUAL_LABEL_ROOT, exist_ok=True)
    history = load_history()

    # =========================================================
    # STEP 1: Prepare random oversized seed pool and manual labeling
    # =========================================================
    print("\n--- STEP 1: Random Seed Pool + Manual Labeling ---")
    paths_full = load_full_paths()
    print(f"Full dataset paths discovered: {len(paths_full)}")
    ensure_balanced_seed_from_random_pool(paths_full)

    # =========================================================
    # STEP 2: Load balanced 300-seed set and validate
    # =========================================================
    print("\n--- STEP 2: Load Balanced Seed & Validate ---")
    print("Loading balanced seed images...")
    X_seed, y_seed, seed_incorporated = load_seed()
    print(f"Seed set: {len(X_seed)} images")
    validate_seed_dataset(X_seed, y_seed)
    aug_per_image = len(ROTATION_ANGLES) + 1 + 4
    print(
        f"Augmented: {len(X_seed) * aug_per_image} additional images planned "
        f"({aug_per_image} per original: 2 rotation + 1 noise + 4 shifts)"
    )

    # =========================================================
    # STEP 3: Load full dataset features and ground truth
    # =========================================================
    print("\n--- STEP 3: Load Full Dataset Features & Ground Truth ---")
    paths_full, X_full = load_full_dataset(paths=paths_full)
    gt_indices, gt_labels = load_ground_truth()
    print(f"Full dataset: {len(X_full)} images")
    print(f"Ground truth: {len(gt_indices)} labeled samples")

    # =========================================================
    # MAIN ITERATION LOOP (AUTO-CONTINUE, HUMAN-IN-THE-LOOP)
    # =========================================================
    should_continue = True
    final_stop_reason = "not reached yet"
    while should_continue:
        # If any annotation batch is unfinished, wait here instead of requiring script rerun.
        pending = find_pending_annotation_iteration(MANUAL_LABEL_ROOT)
        if pending is not None:
            pending_iter, expected, labeled = pending
            print("\nManual labeling is still pending.")
            print(f"Iteration {pending_iter:03d}: labeled {labeled}/{expected} images.")
            user_choice = input("Finish labeling those images, then press Enter to continue (or type 'q' to quit): ").strip().lower()
            if user_choice == "q":
                print("Stopping by user request.")
                final_stop_reason = "user quit with pending manual labels"
                should_continue = False
                break
            # Re-check pending folders after user confirmation.
            continue

        # Iteration number is based on existing iter_* folders, not history length.
        iteration_num = get_next_iteration_number(MANUAL_LABEL_ROOT)
        practical_acc, oracle_acc, should_continue, aborted_by_user, stop_reason = run_iteration(
            iteration_num, X_seed, y_seed, seed_incorporated,
            paths_full, X_full, gt_indices, gt_labels, history
        )

        if aborted_by_user:
            should_continue = False
            final_stop_reason = stop_reason
            break

        if not should_continue:
            final_stop_reason = stop_reason

    # =========================================================
    # Final Summary
    # =========================================================
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETED")
    print(f"{'='*70}")
    print(f"Completed iterations: {len(history)}")
    
    if history:
        sorted_iters = sorted(history.keys())
        first_acc = history[sorted_iters[0]]['practical']
        final_acc = history[sorted_iters[-1]]['practical']
        print(f"Accuracy progression: {first_acc*100:.2f}% → {final_acc*100:.2f}%")
        print(f"Total improvement: {(final_acc - first_acc)*100:+.2f}%")

        total_boundary_manual = sum(history[i].get('manual_new', 0) for i in sorted_iters)
        total_manual_labels = len(X_seed) + total_boundary_manual
        total_pseudo_added = sum(history[i].get('pseudo_added', 0) for i in sorted_iters)
        total_manual_seconds = total_manual_labels * 10

        print_block_diagram()
        print_compact_iteration_table(history)

        print("\nREPORT TOTALS")
        print(f"Final practical accuracy: {final_acc*100:.2f}%")
        final_oracle = history[sorted_iters[-1]].get('oracle', np.nan)
        if np.isfinite(final_oracle):
            print(f"Final oracle accuracy: {final_oracle*100:.2f}%")
        print(f"Iterations performed: {len(sorted_iters)}")
        print(f"Total manually labelled images: {total_manual_labels} (seed={len(X_seed)}, boundary={total_boundary_manual})")
        print(f"Total pseudo-labelled images incorporated: {total_pseudo_added}")
        print(
            "Pseudo-labelled rejected by threshold per iteration (out of top-50/class window): " +
            ", ".join(
                [
                    f"iter {i}: {history[i].get('pseudo_rejected', 0)}/"
                    f"{history[i].get('pseudo_topk_considered', 0)}"
                    for i in sorted_iters
                ]
            )
        )
        print(
            f"Total estimated manual time: {total_manual_seconds} seconds "
            f"(~{total_manual_seconds / 60:.1f} minutes)"
        )

    if should_continue:
        print("\nPipeline is paused (manual step pending or user quit).")
    else:
        print("\nStopping condition reached. No further manual labeling required.")

    print(f"Stop reason: {final_stop_reason}")

    print(f"History saved to: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
