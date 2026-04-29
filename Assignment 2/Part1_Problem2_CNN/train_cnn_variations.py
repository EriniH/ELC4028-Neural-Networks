import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from LoadData import get_dataloaders
from model import get_model_variants


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def run_experiment(
    model_name: str,
    model_builder,
    description: str,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    max_train_batches: int | None,
    max_test_batches: int | None,
):
    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n=== {model_name} ===")
    print(f"Description: {description}")

    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=max_train_batches,
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}%"
        )

    training_ms = (time.perf_counter() - train_start) * 1000.0

    test_start = time.perf_counter()
    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=max_test_batches,
    )
    testing_ms = (time.perf_counter() - test_start) * 1000.0

    print(f"Test | loss={test_loss:.4f} | acc={test_acc:.2f}%")
    print(
        f"Timing | training={training_ms:.1f} ms | testing={testing_ms:.1f} ms"
    )

    return {
        "variation": model_name,
        "description": description,
        "accuracy_percent": round(test_acc, 1),
        "training_time_ms": round(training_ms, 1),
        "testing_time_ms": round(testing_ms, 1),
        "raw_test_accuracy": test_acc,
        "raw_test_loss": test_loss,
    }


def default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "Materials" / "ReducedMNIST_generated"


def save_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "cnn_variations_results.csv"
    json_path = output_dir / "cnn_variations_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Variation",
                "Description",
                "Accuracy (%)",
                "Training Time (ms)",
                "Testing Time (ms)",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "Variation": row["variation"],
                    "Description": row["description"],
                    "Accuracy (%)": f"{row['accuracy_percent']:.1f}",
                    "Training Time (ms)": f"{row['training_time_ms']:.1f}",
                    "Testing Time (ms)": f"{row['testing_time_ms']:.1f}",
                }
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline CNN + variations for Assignment 2 Problem 2"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir()),
        help="Directory containing train/ and test/ folders",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="part2_results",
        help="Where to save results files",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on train batches per epoch (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional cap on test batches (useful for smoke tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Point --data-dir to ReducedMNIST_generated."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")

    train_loader, test_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
    )

    variants = get_model_variants()
    results = []

    for model_name, info in variants.items():
        result = run_experiment(
            model_name=model_name,
            model_builder=info["builder"],
            description=info["description"],
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_train_batches=args.max_train_batches,
            max_test_batches=args.max_test_batches,
        )
        results.append(result)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    save_results(results=results, output_dir=output_dir)

    print("\nFinal summary:")
    for row in results:
        print(
            f"- {row['variation']}: "
            f"acc={row['accuracy_percent']:.1f}% | "
            f"train={row['training_time_ms']:.1f} ms | "
            f"test={row['testing_time_ms']:.1f} ms"
        )


if __name__ == "__main__":
    main()
