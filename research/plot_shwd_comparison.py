
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"A:\PPE detection system")
CSV_PATH = ROOT / "runs" / "shwd_comparison" / "shwd_comparison.csv"
OUTPUT_DIR = ROOT / "runs" / "shwd_comparison" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read actual evaluation results
models = []
metrics = {
    "Precision": [],
    "Recall": [],
    "mAP50": [],
    "mAP50-95": []
}

with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        models.append(row["Model"])

        for metric in metrics:
            metrics[metric].append(float(row[metric]) * 100)

if len(models) != 2:
    raise ValueError("Expected exactly two models in the comparison CSV.")

# Consistent publication-style formatting
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 17,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

def create_metric_graph(metric, values):
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        models,
        values,
        width=0.5
    )

    ax.set_title(f"{metric} Comparison on SHWD Test Dataset")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    add_labels(ax, bars)

    fig.tight_layout()

    filename = metric.lower().replace("@", "").replace("-", "_")

    fig.savefig(
        OUTPUT_DIR / f"{filename}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# Generate individual graphs
for metric, values in metrics.items():
    create_metric_graph(metric, values)

# Generate combined 2x2 comparison figure
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

fig.suptitle(
    "YOLO11n vs YOLO12n: SHWD Independent Test Evaluation",
    fontweight="bold"
)

for ax, (metric, values) in zip(axes.flatten(), metrics.items()):
    bars = ax.bar(models, values, width=0.5)

    ax.set_title(metric)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    add_labels(ax, bars)

fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(
    OUTPUT_DIR / "SHWD_Combined_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "SHWD_Combined_Comparison.pdf",
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 55)
print("SHWD COMPARISON GRAPHS GENERATED")
print("=" * 55)

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)

print("\nOutput directory:")
print(OUTPUT_DIR)
