import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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


def plot_roc_curves(models_dict, X_test, y_test, save_path="results/roc_curve.png"):
    """
    Plots ROC curves for multiple trained models.

    Parameters:
    - models_dict: dictionary {"Model name": trained_model}
    - X_test: test feature set
    - y_test: true labels
    - save_path: where to save the png file
    """

    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to: {save_path}")
