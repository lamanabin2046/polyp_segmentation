import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import torch
from torch.utils.data import DataLoader
from dataset import PolypDataset, get_val_transform
from model import build_unetpp_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs/checkpoints/best_model.pth"

SAVE_DIR = "outputs/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Load Model
# -------------------------
def load_model():
    model = build_unetpp_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model


# -------------------------
# Collect predictions
# -------------------------
def collect_predictions():
    model = load_model()

    dataset = PolypDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        transform=get_val_transform()
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_probs = []
    all_preds = []
    all_masks = []

    for batch in loader:
        img, mask = batch
        img = img.to(DEVICE)
        mask = mask.cpu().numpy().astype(np.uint8)

        with torch.no_grad():
            pred = model(img)
            prob = torch.sigmoid(pred)[0, 0].cpu().numpy()
            pred_bin = (prob > 0.5).astype(np.uint8)

        all_probs.extend(prob.flatten())
        all_preds.extend(pred_bin.flatten())
        all_masks.extend(mask.flatten())

    return np.array(all_probs), np.array(all_preds), np.array(all_masks)


# -------------------------
# PLOT 1 — Confusion Matrix
# -------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Polyp", "Polyp"],
                yticklabels=["No Polyp", "Polyp"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png", dpi=300)
    plt.close()


# -------------------------
# PLOT 2 — ROC Curve
# -------------------------
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/roc_curve.png", dpi=300)
    plt.close()


# -------------------------
# PLOT 3 — Precision–Recall Curve
# -------------------------
def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.savefig(f"{SAVE_DIR}/pr_curve.png", dpi=300)
    plt.close()


# -------------------------
# PLOT 4 — Dice & IoU Distributions
# -------------------------
def compute_dice(pred, mask):
    smooth = 1e-6
    pred = pred.flatten()
    mask = mask.flatten()
    inter = (pred * mask).sum()
    return (2*inter + smooth) / (pred.sum() + mask.sum() + smooth)


def compute_iou(pred, mask):
    smooth = 1e-6
    pred = pred.flatten()
    mask = mask.flatten()
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum() - inter
    return (inter + smooth) / (union + smooth)


def plot_dice_iou_distribution(y_pred, y_true):
    dice_scores = []
    iou_scores = []

    # Compute image-level Dice/IoU (requires reshaping)
    # Here we do pixel-level distribution
    for p, m in zip(y_pred, y_true):
        dice_scores.append(2*p*m / (p + m + 1e-6))
        iou_scores.append(p*m / (p + m - p*m + 1e-6))

    plt.figure()
    sns.histplot(dice_scores, bins=30, kde=True, color="green")
    plt.title("Dice Score Distribution")
    plt.savefig(f"{SAVE_DIR}/dice_distribution.png", dpi=300)
    plt.close()

    plt.figure()
    sns.histplot(iou_scores, bins=30, kde=True, color="purple")
    plt.title("IoU Distribution")
    plt.savefig(f"{SAVE_DIR}/iou_distribution.png", dpi=300)
    plt.close()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("Collecting predictions...")
    probs, preds, masks = collect_predictions()

    print("Generating plots...")
    plot_confusion_matrix(masks, preds)
    plot_roc_curve(masks, probs)
    plot_pr_curve(masks, probs)
    plot_dice_iou_distribution(preds, masks)

    print(f"\nPlots saved to: {SAVE_DIR}")
