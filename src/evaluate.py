import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import build_unetpp_model
from dataset import PolypDataset, get_val_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs/checkpoints/best_model.pth"


def load_model():
    model = build_unetpp_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model


def dice_score(pred, mask):
    smooth = 1e-6
    pred = pred.flatten()
    mask = mask.flatten()
    intersection = (pred * mask).sum()
    return (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth)


def iou_score(pred, mask):
    smooth = 1e-6
    pred = pred.flatten()
    mask = mask.flatten()
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate():
    model = load_model()

    dataset = PolypDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        transform=get_val_transform()      # <-- IMPORTANT FIX
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds, all_masks = [], []
    dice_scores, iou_scores = [], []

    for batch in loader:

        # Unpack tuple (image, mask)
        img, mask = batch
        img = img.to(DEVICE)
        mask = mask.cpu().numpy().astype(np.uint8)

        with torch.no_grad():
            pred = model(img)
            pred = torch.sigmoid(pred)[0, 0].cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

        all_preds.extend(pred_bin.flatten())
        all_masks.extend(mask.flatten())

        dice_scores.append(dice_score(pred_bin, mask))
        iou_scores.append(iou_score(pred_bin, mask))

    tn, fp, fn, tp = confusion_matrix(all_masks, all_preds).ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    print("====== Segmentation Metrics ======")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-score:        {f1:.4f}")
    print(f"Dice Score:      {np.mean(dice_scores):.4f}")
    print(f"IoU (Jaccard):   {np.mean(iou_scores):.4f}")

    print("\n====== Confusion Matrix ======")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")


if __name__ == "__main__":
    evaluate()
