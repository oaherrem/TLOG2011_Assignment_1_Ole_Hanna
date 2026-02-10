import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix plot to {save_path}")


def plot_model_comparison(metrics_dict, save_path):
    """
    metrics_dict = {
        "Logistic Regression": {"accuracy":..., "precision":..., "recall":...},
        "Random Forest": {...},
        "Gradient Boosting": {...}
    }
    """
    models = list(metrics_dict.keys())
    accuracy = [metrics_dict[m]["accuracy"] for m in models]
    precision = [metrics_dict[m]["precision"] for m in models]
    recall = [metrics_dict[m]["recall"] for m in models]

    x = range(len(models))

    plt.figure(figsize=(7, 4))
    plt.bar(x, accuracy, width=0.25, label="Accuracy")
    plt.bar([i + 0.25 for i in x], precision, width=0.25, label="Precision")
    plt.bar([i + 0.5 for i in x], recall, width=0.25, label="Recall")

    plt.xticks([i + 0.25 for i in x], models)
    plt.ylabel("Score")
    plt.title("Model performance comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved model comparison plot to {save_path}")