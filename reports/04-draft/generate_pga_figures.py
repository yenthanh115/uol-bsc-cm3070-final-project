"""Generate figures for PGA report: pipeline architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_pipeline_architecture():
    """Draw a pipeline architecture diagram showing the 5 processing stages."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    # Stage definitions
    stages = [
        ("Load &\nTicker\nExtraction", "#4ECDC4"),
        ("Temporal\nWindowing", "#45B7D1"),
        ("Sentiment\nAnalysis", "#96CEB4"),
        ("Composite\nSurge\nLabelling", "#FFEAA7"),
        ("Model\nTraining &\nEvaluation", "#DDA0DD"),
    ]

    # Input/output labels
    input_label = "Reddit\nCSV"
    output_label = "Metrics,\nModels,\nFigures"

    box_width = 1.6
    box_height = 1.8
    spacing = 0.4
    start_x = 1.5

    # Draw input
    ax.text(
        0.5, 0.9, input_label,
        ha="center", va="center", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#333"),
    )

    # Draw stages
    for i, (label, color) in enumerate(stages):
        x = start_x + i * (box_width + spacing)
        rect = mpatches.FancyBboxPatch(
            (x, 0), box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#333", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2, box_height / 2,
            label, ha="center", va="center",
            fontsize=9, fontweight="bold",
        )
        # Stage number
        ax.text(
            x + box_width / 2, box_height - 0.2,
            f"Stage {i + 1}", ha="center", va="top",
            fontsize=7, fontstyle="italic", color="#555",
        )

        # Arrow between stages
        if i == 0:
            # Arrow from input to first stage
            ax.annotate(
                "", xy=(x, 0.9), xytext=(1.0, 0.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
            )
        if i < len(stages) - 1:
            next_x = start_x + (i + 1) * (box_width + spacing)
            ax.annotate(
                "", xy=(next_x, 0.9), xytext=(x + box_width, 0.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
            )

    # Draw output
    last_x = start_x + (len(stages) - 1) * (box_width + spacing) + box_width
    ax.text(
        last_x + 0.6, 0.9, output_label,
        ha="center", va="center", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#333"),
    )
    ax.annotate(
        "", xy=(last_x + 0.1, 0.9), xytext=(last_x, 0.9),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
    )

    # Annotations below stages
    annotations = [
        "CSV parsing\nTimestamp normalisation\nTicker extraction",
        "24h forward/backward\ncounts per ticker",
        "VADER polarity\nscoring",
        "Z-score normalisation\n(train stats only)\nWeighted composite\nBinary threshold τ",
        "LR / RF / XGBoost\nTemporal CV (k=4)\nGrid search\nAUC-ROC evaluation",
    ]
    for i, ann in enumerate(annotations):
        x = start_x + i * (box_width + spacing) + box_width / 2
        ax.text(
            x, -0.3, ann,
            ha="center", va="top", fontsize=7, color="#555",
            linespacing=1.3,
        )

    fig.suptitle(
        "Surge Detection Pipeline Architecture",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pipeline_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pipeline_architecture.png'}")


def draw_expanding_window_cv():
    """Draw expanding-window temporal cross-validation diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4)
    ax.axis("off")

    fold_width = 1.2
    fold_height = 0.6
    y_spacing = 0.9

    train_color = "#4ECDC4"
    val_color = "#FF6B6B"
    unused_color = "#f0f0f0"

    # Labels
    ax.text(-0.3, 3.5, "Fold:", ha="right", va="center", fontsize=9, fontweight="bold")
    for f in range(4):
        x = f * fold_width + 0.5
        ax.text(x + fold_width / 2, 3.5, f"Fold {f + 1}", ha="center", va="center", fontsize=9)

    # Splits
    splits = [
        # (train_folds, val_fold)
        ([0], 1),
        ([0, 1], 2),
        ([0, 1, 2], 3),
    ]

    for split_i, (train_folds, val_fold) in enumerate(splits):
        y = 2.5 - split_i * y_spacing
        ax.text(-0.3, y + fold_height / 2, f"Split {split_i + 1}",
                ha="right", va="center", fontsize=9)

        for f in range(4):
            x = f * fold_width + 0.5
            if f in train_folds:
                color = train_color
            elif f == val_fold:
                color = val_color
            else:
                color = unused_color

            rect = mpatches.FancyBboxPatch(
                (x, y), fold_width - 0.05, fold_height,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="#333", linewidth=1,
            )
            ax.add_patch(rect)

    # Legend
    train_patch = mpatches.Patch(facecolor=train_color, edgecolor="#333", label="Training")
    val_patch = mpatches.Patch(facecolor=val_color, edgecolor="#333", label="Validation")
    unused_patch = mpatches.Patch(facecolor=unused_color, edgecolor="#333", label="Unused")
    ax.legend(
        handles=[train_patch, val_patch, unused_patch],
        loc="lower center", ncol=3, fontsize=9, framealpha=0.9,
    )

    # Time arrow
    ax.annotate(
        "", xy=(4.8, -0.15), xytext=(0.4, -0.15),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#666"),
    )
    ax.text(2.6, -0.35, "Time →", ha="center", va="top", fontsize=9, color="#666")

    fig.suptitle(
        "Expanding-Window Temporal Cross-Validation (k=4 folds, 3 splits)",
        fontsize=11, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expanding_window_cv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'expanding_window_cv.png'}")


if __name__ == "__main__":
    draw_pipeline_architecture()
    draw_expanding_window_cv()
    print("All PGA figures generated.")
