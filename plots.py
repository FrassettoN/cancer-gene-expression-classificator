import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def roc_plot(y_true, y_score, auc_value, model_name, cancer_type, charts_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, color="b", label=f"ROC (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve - {model_name} ({cancer_type})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(charts_dir, f"roc_curve_{cancer_type}_{model_name}.png")
    plt.savefig(output_path, dpi=1200)
    plt.close()


def accuracy_bar_chart(file_paths, models, accuracies, std_errors, charts_dir):
    labels = list(file_paths.values())
    x = np.arange(len(labels))
    width = 0.15

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(models)))

    fig, ax = plt.subplots(figsize=(12, 8))

    for y in [0.6, 0.7, 0.8, 0.9, 1.0]:
        ax.axhline(y=y, color="#dfdfdf", linestyle="-", linewidth=1, zorder=0)

    n_models = len(models)
    center_offset = (n_models - 1) / 2

    for i, model in enumerate(models):
        xpos = x + (i - center_offset) * width
        ax.bar(
            xpos,
            accuracies[model],
            width,
            yerr=std_errors[model],
            capsize=3,
            label=model,
            color=colors[i],
            ecolor="#727272",
            zorder=2,
        )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.6, 1)
    ax.set_title("Model Performance Across Cancer Types (SEM)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=5)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    output_path = os.path.join(charts_dir, "bar_chart.png")
    plt.savefig(output_path, dpi=1200)
    plt.close()
    print(f"The chart has been saved as 'bar_chart.png' in: {charts_dir}")
