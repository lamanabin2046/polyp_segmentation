import cv2
import torch
import numpy as np
from model import build_unetpp_model
from dataset import get_val_transform
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs/checkpoints/best_model.pth"
SAVE_DIR = "outputs/predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_model():
    model = build_unetpp_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model

def predict_image(model, img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_val_transform()
    aug = transform(image=img_rgb)
    tensor = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)[0,0].cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8)

    return img, mask

def save_overlay(img, mask, save_path):
    # Resize mask back to original image size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_color = np.zeros_like(img)
    mask_color[:, :, 1] = mask * 255   # green mask

    overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
    cv2.imwrite(save_path, overlay)

def main():
    model = load_model()
    test_image = "test.jpg"   # change this

    img, mask = predict_image(model, test_image)

    save_overlay(img, mask, f"{SAVE_DIR}/overlay.png")
    cv2.imwrite(f"{SAVE_DIR}/mask.png", mask * 255)

    print("Prediction saved in outputs/predictions")

if __name__ == "__main__":
    main()
