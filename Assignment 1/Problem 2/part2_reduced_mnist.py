"""Part 2 ReducedMNIST pipeline for Assignment 1."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRAIN_DIR_NAMES = [
    "train",
    "training",
    "reducedmnist_train",
    "reduced training data",
    "reduced trainging data",
]
TEST_DIR_NAMES = [
    "test",
    "testing",
    "reducedmnist_test",
    "reduced testing data",
]
# Base directory for generated datasets and outputs: place generated files
# next to this script (the `part2` folder).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_CANDIDATES = [
    SCRIPT_DIR / "ReducedMNIST",
    SCRIPT_DIR / "Reduced MNIST Data(just an experiment)",
    SCRIPT_DIR / "ReducedMNIST_generated",
]
GENERATED_ROOT = SCRIPT_DIR / "ReducedMNIST_generated"
MNIST_ROOT = SCRIPT_DIR / "mnist_cache"
TRAIN_SAMPLES_PER_CLASS = 1000
TEST_SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42
DIGITS = tuple(range(10))


@dataclass
class FeatureSet:
    name: str
    train_x: np.ndarray
    test_x: np.ndarray
    feature_time_s: float


@dataclass
class ExperimentResult:
    classifier: str
    feature_name: str
    accuracy: float
    total_time_s: float
    confusion: np.ndarray


def with_progress(iterable, desc: str, total: int | None = None):
    """Wrap an iterable with tqdm when available."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, desc=desc, total=total, leave=False)


class PerClassKMeansClassifier:
    """K-means per class, then nearest centroid at prediction time."""

    def __init__(self, clusters_per_class: int, random_state: int = 42) -> None:
        self.clusters_per_class = clusters_per_class
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.centroids_: np.ndarray | None = None
        self.centroid_labels_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PerClassKMeansClassifier":
        x_scaled = self.scaler.fit_transform(x)
        centroids = []
        centroid_labels = []

        for label in sorted(np.unique(y)):
            class_x = x_scaled[y == label]
            k = min(self.clusters_per_class, len(class_x))
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            model.fit(class_x)
            centroids.append(model.cluster_centers_)
            centroid_labels.extend([label] * k)

        self.centroids_ = np.vstack(centroids)
        self.centroid_labels_ = np.asarray(centroid_labels)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.centroid_labels_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        x_scaled = self.scaler.transform(x)
        # Compute squared Euclidean distance in batches to avoid allocating
        # a large (num_samples, num_centroids, num_features) tensor.
        centroids = self.centroids_
        centroid_sq_norms = np.sum(centroids * centroids, axis=1)
        nearest = np.empty(x_scaled.shape[0], dtype=np.int64)

        batch_size = 2048
        for start in range(0, x_scaled.shape[0], batch_size):
            end = min(start + batch_size, x_scaled.shape[0])
            batch = x_scaled[start:end]
            batch_sq_norms = np.sum(batch * batch, axis=1, keepdims=True)
            distances = batch_sq_norms + centroid_sq_norms[None, :] - 2.0 * (batch @ centroids.T)
            nearest[start:end] = np.argmin(distances, axis=1)

        return self.centroid_labels_[nearest]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2 ReducedMNIST pipeline")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="ReducedMNIST folder to use. If missing, the script generates it from MNIST.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "part2_results",
        help="Where to save the assignment outputs.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Expected image width and height.",
    )
    return parser.parse_args()


def find_split_dir(root: Path, candidates: Iterable[str]) -> Path:
    names = {name.lower() for name in candidates}

    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in names:
            return child

    for child in root.rglob("*"):
        if child.is_dir() and child.name.lower() in names:
            return child

    raise FileNotFoundError(f"Could not find {sorted(names)} under {root}.")


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def looks_like_reduced_mnist(root: Path) -> bool:
    try:
        find_split_dir(root, TRAIN_DIR_NAMES)
        find_split_dir(root, TEST_DIR_NAMES)
    except FileNotFoundError:
        return False
    return True


def sample_balanced_subset(
    images,
    labels,
    samples_per_class: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    selected_indices = []
    for digit in DIGITS:
        digit_indices = (labels == digit).nonzero(as_tuple=True)[0]
        if len(digit_indices) < samples_per_class:
            raise ValueError(
                f"Digit {digit} has only {len(digit_indices)} samples, "
                f"but {samples_per_class} were requested."
            )
        shuffled = digit_indices[torch.randperm(len(digit_indices), generator=generator)]
        selected_indices.append(shuffled[:samples_per_class])

    selected_indices = torch.cat(selected_indices)
    selected_indices = selected_indices[
        torch.randperm(len(selected_indices), generator=generator)
    ]

    return (
        images[selected_indices].numpy().astype(np.uint8),
        labels[selected_indices].numpy().astype(np.int64),
    )


def save_split(images: np.ndarray, labels: np.ndarray, split_dir: Path) -> None:
    counters = {digit: 0 for digit in DIGITS}
    items = zip(images, labels)

    for image, label in with_progress(items, f"Saving {split_dir.name}", len(labels)):
        label = int(label)
        counters[label] += 1
        class_dir = split_dir / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        image_path = class_dir / f"image_{counters[label]:05d}.png"
        Image.fromarray(image, mode="L").save(image_path)


def generate_reduced_mnist(output_root: Path, image_size: int) -> Path:
    if image_size != 28:
        raise ValueError("MNIST generation expects --image-size 28.")

    if output_root.exists():
        if looks_like_reduced_mnist(output_root):
            return output_root
        raise FileExistsError(
            f"{output_root} exists, but it is not a valid ReducedMNIST folder."
        )

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "Generating ReducedMNIST requires torch and torchvision. "
            "Install them with: pip install torch torchvision"
        ) from exc

    print(f"Downloading MNIST to {MNIST_ROOT}")
    train_dataset = datasets.MNIST(root=str(MNIST_ROOT), train=True, download=True)
    test_dataset = datasets.MNIST(root=str(MNIST_ROOT), train=False, download=True)

    train_images, train_labels = sample_balanced_subset(
        train_dataset.data,
        train_dataset.targets,
        TRAIN_SAMPLES_PER_CLASS,
        RANDOM_SEED,
    )
    test_images, test_labels = sample_balanced_subset(
        test_dataset.data,
        test_dataset.targets,
        TEST_SAMPLES_PER_CLASS,
        RANDOM_SEED + 1,
    )

    print(f"Saving generated ReducedMNIST to {output_root}")
    save_split(train_images, train_labels, output_root / "train")
    save_split(test_images, test_labels, output_root / "test")
    return output_root


def resolve_data_root(requested_root: Path | None, image_size: int) -> Path:
    if requested_root is not None:
        if requested_root.exists():
            return requested_root
        print(f"Dataset not found at {requested_root}, generating it from MNIST...")
        return generate_reduced_mnist(requested_root, image_size)

    for candidate in DEFAULT_ROOT_CANDIDATES:
        if candidate.exists() and looks_like_reduced_mnist(candidate):
            return candidate

    print("No local ReducedMNIST found, generating ReducedMNIST_generated from MNIST...")
    return generate_reduced_mnist(GENERATED_ROOT, image_size)


def validate_split(split_dir: Path, expected_count: int, split_name: str) -> list[Path]:
    class_dirs = sorted(
        [folder for folder in split_dir.iterdir() if folder.is_dir() and folder.name.isdigit()],
        key=lambda folder: int(folder.name),
    )
    labels = [int(folder.name) for folder in class_dirs]

    if labels != list(DIGITS):
        raise FileNotFoundError(
            f"{split_name} split at {split_dir} must contain digit folders 0-9 exactly once."
        )

    bad_counts = []
    for folder in class_dirs:
        count = len(list_image_files(folder))
        if count != expected_count:
            bad_counts.append(f"{folder.name}:{count}")

    if bad_counts:
        raise ValueError(
            f"{split_name} split at {split_dir} must contain exactly "
            f"{expected_count} images per digit. Found: {', '.join(bad_counts)}"
        )

    return class_dirs


def read_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32).copy()
    image.close()

    if array.shape != (image_size, image_size):
        raise ValueError(
            f"Expected {(image_size, image_size)} for {path}, but found {array.shape}."
        )

    return array / 255.0


def load_split(
    split_dir: Path,
    image_size: int,
    expected_count: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    class_dirs = validate_split(split_dir, expected_count, split_name)
    samples: list[tuple[int, Path]] = []

    for folder in class_dirs:
        label = int(folder.name)
        samples.extend((label, path) for path in list_image_files(folder))

    images = []
    labels = []
    for label, path in with_progress(samples, f"Loading {split_name}", len(samples)):
        images.append(read_image(path, image_size))
        labels.append(label)

    return np.asarray(images), np.asarray(labels)


def load_dataset(root: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = find_split_dir(root, TRAIN_DIR_NAMES)
    test_dir = find_split_dir(root, TEST_DIR_NAMES)

    train_images, train_labels = load_split(
        train_dir,
        image_size,
        TRAIN_SAMPLES_PER_CLASS,
        "train",
    )
    test_images, test_labels = load_split(
        test_dir,
        image_size,
        TEST_SAMPLES_PER_CLASS,
        "test",
    )
    return train_images, train_labels, test_images, test_labels


def dct2(image: np.ndarray) -> np.ndarray:
    return dct(dct(image.T, norm="ortho").T, norm="ortho")


def extract_dct_features(images: np.ndarray, desc: str) -> np.ndarray:
    features = []
    for image in with_progress(images, desc, len(images)):
        coeffs = dct2(image)
        features.append(coeffs[:15, :15].reshape(-1))
    return np.asarray(features)


def extract_hog_features(images: np.ndarray, desc: str) -> np.ndarray:
    features = []
    for image in with_progress(images, desc, len(images)):
        features.append(
            hog(
                image,
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
        )
    return np.asarray(features)


def extract_pca_features(
    train_images: np.ndarray,
    test_images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    print("  PCA: flattening images...")
    train_flat = train_images.reshape(len(train_images), -1)
    test_flat = test_images.reshape(len(test_images), -1)

    print("  PCA: fitting on train set...")
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    train_pca = pca.fit_transform(train_flat)

    print("  PCA: transforming test set...")
    test_pca = pca.transform(test_flat)
    return train_pca, test_pca


def build_feature_sets(
    train_images: np.ndarray,
    test_images: np.ndarray,
) -> dict[str, FeatureSet]:
    feature_sets: dict[str, FeatureSet] = {}

    start = time.perf_counter()
    train_dct = extract_dct_features(train_images, "DCT train")
    test_dct = extract_dct_features(test_images, "DCT test")
    feature_sets["DCT"] = FeatureSet("DCT", train_dct, test_dct, time.perf_counter() - start)

    start = time.perf_counter()
    train_pca, test_pca = extract_pca_features(train_images, test_images)
    feature_sets["PCA"] = FeatureSet("PCA", train_pca, test_pca, time.perf_counter() - start)

    start = time.perf_counter()
    train_hog = extract_hog_features(train_images, "HOG train")
    test_hog = extract_hog_features(test_images, "HOG test")
    feature_sets["HOG"] = FeatureSet("HOG", train_hog, test_hog, time.perf_counter() - start)

    return feature_sets


def run_experiment(
    classifier_name: str,
    model,
    feature_set: FeatureSet,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> ExperimentResult:
    start = time.perf_counter()
    model.fit(feature_set.train_x, train_y)
    predictions = model.predict(feature_set.test_x)
    total_time_s = feature_set.feature_time_s + (time.perf_counter() - start)

    accuracy = accuracy_score(test_y, predictions)
    confusion = confusion_matrix(test_y, predictions, labels=np.arange(10))
    return ExperimentResult(
        classifier=classifier_name,
        feature_name=feature_set.name,
        accuracy=accuracy,
        total_time_s=total_time_s,
        confusion=confusion,
    )


def build_assignment_rows(results: list[ExperimentResult]) -> list[list[str]]:
    lookup = {(result.classifier, result.feature_name): result for result in results}
    row_specs = [
        ("K-means Clustering", "1", "KMeans_per_class_K=1"),
        ("", "4", "KMeans_per_class_K=4"),
        ("", "16", "KMeans_per_class_K=16"),
        ("", "32", "KMeans_per_class_K=32"),
        ("SVM", "Linear", "SVM_linear"),
        ("", "nonlinear*", "SVM_rbf"),
    ]

    rows = []
    for classifier_label, setting_label, classifier_key in row_specs:
        row = [classifier_label, setting_label]
        for feature_name in ["DCT", "PCA", "HOG"]:
            result = lookup[(classifier_key, feature_name)]
            row.extend(
                [
                    f"{100.0 * result.accuracy:.2f}%",
                    f"{result.total_time_s:.4f} s",
                ]
            )
        rows.append(row)
    return rows


def save_assignment_table(results: list[ExperimentResult], output_dir: Path) -> None:
    path = output_dir / "assignment_style_table.csv"
    rows = build_assignment_rows(results)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["", "", "DCT", "", "PCA", "", "HOG", ""])
        writer.writerow(
            [
                "Classifier",
                "Setting",
                "Accuracy",
                "Processing Time",
                "Accuracy",
                "Processing Time",
                "Accuracy",
                "Processing Time",
            ]
        )
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["Note", "nonlinear kernel = RBF, gamma=scale"])
        writer.writerow(["Note", "Processing Time = feature extraction + training + testing"])


def save_confusion_matrix(result: ExperimentResult, output_dir: Path) -> None:
    safe_name = f"{result.classifier}_{result.feature_name}".replace(" ", "_")
    csv_path = output_dir / f"{safe_name}_confusion.csv"
    np.savetxt(csv_path, result.confusion, fmt="%d", delimiter=",")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(result.confusion, cmap="Blues")
    ax.set_title(f"{result.classifier} with {result.feature_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(result.confusion[row, col]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}_confusion.png", dpi=200)
    plt.close(fig)


def format_result(result: ExperimentResult) -> str:
    return (
        f"{result.classifier} with {result.feature_name} "
        f"({100.0 * result.accuracy:.2f}% accuracy, {result.total_time_s:.4f} s total)"
    )


def save_conclusions(
    results: list[ExperimentResult],
    best_kmeans: ExperimentResult,
    best_svm: ExperimentResult,
    output_dir: Path,
) -> None:
    best_overall = max(results, key=lambda result: (result.accuracy, -result.total_time_s))
    fastest = min(results, key=lambda result: (result.total_time_s, -result.accuracy))
    best_by_feature = {
        feature_name: max(
            [result for result in results if result.feature_name == feature_name],
            key=lambda result: (result.accuracy, -result.total_time_s),
        )
        for feature_name in ["DCT", "PCA", "HOG"]
    }
    accuracy_gap = 100.0 * (best_svm.accuracy - best_kmeans.accuracy)

    lines = [
        "Assignment Part 2 Conclusions",
        "",
        f"1. Best overall result: {format_result(best_overall)}.",
        f"2. Best K-means result: {format_result(best_kmeans)}.",
        f"3. Best SVM result: {format_result(best_svm)}.",
        (
            "4. Classifier comparison: "
            f"the best SVM result was {abs(accuracy_gap):.2f} percentage points "
            f"{'higher' if accuracy_gap >= 0 else 'lower'} than the best K-means result."
        ),
        (
            "5. Feature comparison: "
            f"DCT -> {format_result(best_by_feature['DCT'])}; "
            f"PCA -> {format_result(best_by_feature['PCA'])}; "
            f"HOG -> {format_result(best_by_feature['HOG'])}."
        ),
        f"6. Fastest overall experiment: {format_result(fastest)}.",
        (
            "7. Final takeaway: "
            f"{best_overall.feature_name} features with {best_overall.classifier} "
            "gave the strongest overall result on this ReducedMNIST task."
        ),
    ]
    (output_dir / "conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_data_root(args.data_root, args.image_size)
    print(f"Using dataset folder: {data_root}")

    print("Loading ReducedMNIST...")
    train_images, train_y, test_images, test_y = load_dataset(data_root, args.image_size)
    print(f"Train set: {train_images.shape[0]} images")
    print(f"Test set : {test_images.shape[0]} images")

    print("\nExtracting features...")
    feature_sets = build_feature_sets(train_images, test_images)
    for feature_set in feature_sets.values():
        print(
            f"{feature_set.name:4s} -> {feature_set.train_x.shape[1]} dims, "
            f"feature time = {feature_set.feature_time_s:.4f} s"
        )

    results: list[ExperimentResult] = []

    print("\nRunning K-means experiments...")
    for feature_set in feature_sets.values():
        for k in [1, 4, 16, 32]:
            result = run_experiment(
                classifier_name=f"KMeans_per_class_K={k}",
                model=PerClassKMeansClassifier(clusters_per_class=k, random_state=42),
                feature_set=feature_set,
                train_y=train_y,
                test_y=test_y,
            )
            results.append(result)
            print(f"{result.classifier:24s} + {result.feature_name:4s} -> {100.0 * result.accuracy:.2f}%")

    print("\nRunning SVM experiments...")
    for feature_set in feature_sets.values():
        for kernel in ["linear", "rbf"]:
            result = run_experiment(
                classifier_name=f"SVM_{kernel}",
                model=make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma="scale")),
                feature_set=feature_set,
                train_y=train_y,
                test_y=test_y,
            )
            results.append(result)
            print(f"{result.classifier:24s} + {result.feature_name:4s} -> {100.0 * result.accuracy:.2f}%")

    save_assignment_table(results, args.output_dir)

    best_kmeans = max(
        [result for result in results if result.classifier.startswith("KMeans")],
        key=lambda result: (result.accuracy, -result.total_time_s),
    )
    best_svm = max(
        [result for result in results if result.classifier.startswith("SVM")],
        key=lambda result: (result.accuracy, -result.total_time_s),
    )

    save_confusion_matrix(best_kmeans, args.output_dir)
    save_confusion_matrix(best_svm, args.output_dir)
    save_conclusions(results, best_kmeans, best_svm, args.output_dir)

    print("\nSaved files")
    print(f"- Assignment CSV   : {args.output_dir / 'assignment_style_table.csv'}")
    print(f"- Conclusions text : {args.output_dir / 'conclusions.txt'}")
    print(f"- Confusion files  : {args.output_dir}")


if __name__ == "__main__":
    main()
