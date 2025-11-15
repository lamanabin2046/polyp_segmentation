import sys
sys.path.append("src")

import os
import time
import zipfile
import io
import csv
from datetime import datetime
import gradio as gr
import cv2
import torch
import numpy as np
from model import build_unetpp_model
from dataset import get_val_transform
from gradcam import GradCAM

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs/checkpoints/best_model.pth"
CASES_DIR = "outputs/cases"
os.makedirs(CASES_DIR, exist_ok=True)

# -------------------------
# Load model + CAM
# -------------------------
def load_model():
    model = build_unetpp_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model

model = load_model()
# pick final encoder block for EfficientNet
target_layer = model.encoder._blocks[-1]
cam_generator = GradCAM(model, target_layer)
transform_fn = get_val_transform()


# -------------------------
# Utility: analysis computation (same logic as app)
# -------------------------
def compute_polyp_analysis(mask, pred_prob):
    # mask: binary (HxW) resized to original resolution
    area = int(mask.sum())
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return area, 0, 0.0, 0.0, "No polyp detected"
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    diameter = int(max(w, h))
    perimeter = cv2.arcLength(cnt, True)
    irregularity = float(perimeter**2 / (4 * np.pi * area)) if area > 0 else 0.0
    confidence = float(pred_prob.mean())
    # safe, non-diagnostic risk heuristic
    if area < 500:
        risk = "Low"
    elif area < 3000:
        risk = "Medium"
    else:
        risk = "High (not diagnostic)"
    return area, diameter, irregularity, confidence, risk


# -------------------------
# Core prediction function returning all displays + structured data
# -------------------------
def run_analysis(image):
    if image is None:
        return (None, None, None, None, "No image uploaded", {})
    # image: numpy RGB from gradio
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_rgb = image.copy()
    aug = transform_fn(image=img_rgb)
    tensor = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        prob_map = torch.sigmoid(pred)[0, 0].cpu().numpy()
        mask = (prob_map > 0.5).astype(np.uint8)

    # resize mask to original resolution
    mask_resized = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # overlay
    mask_color = np.zeros_like(img_bgr)
    mask_color[:, :, 1] = mask_resized * 255
    overlay = cv2.addWeighted(img_bgr, 0.7, mask_color, 0.3, 0)

    # gradcam
    cam = cam_generator(tensor)
    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

    # convert to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    cam_overlay_rgb = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)
    mask_display = (mask_resized * 255).astype(np.uint8)

    # compute analysis
    area, diameter, irregularity, confidence, risk = compute_polyp_analysis(mask_resized, prob_map)
    analysis_text = (
        f"Polyp Area: {area} px\n"
        f"Approx Diameter: {diameter} px\n"
        f"Irregularity: {irregularity:.3f}\n"
        f"Model Confidence: {confidence:.3f}\n"
        f"Risk Indicator: {risk}\n\n"
        f"*This is NOT a medical diagnosis*"
    )

    structured = {
        "area": area,
        "diameter": diameter,
        "irregularity": irregularity,
        "confidence": confidence,
        "risk": risk
    }

    # return images (as numpy arrays) and text + structured
    return mask_display, overlay_rgb, heatmap_rgb, cam_overlay_rgb, analysis_text, structured


# -------------------------
# Save case: create folder with timestamp, save images and analysis, return zip path
# -------------------------
def save_case(image, overlay_np, mask_np, heatmap_np, cam_overlay_np, analysis_text, structured):
    # ensure inputs exist
    if image is None:
        return None
    ts = datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = os.path.join(CASES_DIR, f"case_{ts}")
    os.makedirs(case_dir, exist_ok=True)

    # Save original (RGB -> BGR for cv2)
    orig_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(case_dir, "original.jpg"), orig_bgr)

    # Save overlay / mask / heatmap / cam_overlay (they are RGB arrays except mask)
    cv2.imwrite(os.path.join(case_dir, "overlay.jpg"), cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(case_dir, "mask.png"), mask_np)
    cv2.imwrite(os.path.join(case_dir, "heatmap.jpg"), cv2.cvtColor(heatmap_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(case_dir, "gradcam_overlay.jpg"), cv2.cvtColor(cam_overlay_np, cv2.COLOR_RGB2BGR))

    # Save analysis text
    with open(os.path.join(case_dir, "analysis.txt"), "w") as f:
        f.write(analysis_text)

    # Save structured CSV
    csv_path = os.path.join(case_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["area", "diameter", "irregularity", "confidence", "risk"])
        writer.writerow([structured["area"], structured["diameter"], f"{structured['irregularity']:.6f}", f"{structured['confidence']:.6f}", structured["risk"]])

    # Create zip
    zip_path = os.path.join(CASES_DIR, f"case_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(case_dir):
            zf.write(os.path.join(case_dir, fname), arcname=fname)

    return zip_path, case_dir


# -------------------------
# List history: scan CASES_DIR for zip files and return decent list
# -------------------------
def list_saved_cases():
    zips = sorted([f for f in os.listdir(CASES_DIR) if f.endswith(".zip")], reverse=True)
    # return list of tuples (name, path)
    result = []
    for z in zips:
        path = os.path.join(CASES_DIR, z)
        # make a nice label
        label = z.replace(".zip", "")
        result.append((label, path))
    return result


# -------------------------
# Gradio wrapper functions
# -------------------------
# We'll keep the last prediction in memory for saving
LAST_PRED = {"image": None, "overlay": None, "mask": None, "heatmap": None, "cam_overlay": None, "analysis": "", "structured": {}}

def gr_predict(image):
    mask, overlay, heatmap, cam_overlay, analysis_text, structured = run_analysis(image)
    # store numpy images and original image for save
    LAST_PRED["image"] = image
    LAST_PRED["overlay"] = overlay
    LAST_PRED["mask"] = mask
    LAST_PRED["heatmap"] = heatmap
    LAST_PRED["cam_overlay"] = cam_overlay
    LAST_PRED["analysis"] = analysis_text
    LAST_PRED["structured"] = structured
    # return images and text
    return mask, overlay, heatmap, cam_overlay, analysis_text

def gr_save_case():
    if LAST_PRED["image"] is None:
        return None, "No prediction to save."
    zip_path, case_dir = save_case(LAST_PRED["image"], LAST_PRED["overlay"], LAST_PRED["mask"], LAST_PRED["heatmap"], LAST_PRED["cam_overlay"], LAST_PRED["analysis"], LAST_PRED["structured"])
    # refresh history
    history = list_saved_cases()
    return zip_path, f"Saved: {case_dir}"

def gr_list_history():
    return list_saved_cases()

# -------------------------
# Build Gradio layout
# -------------------------
with gr.Blocks(title="Polyp Clinical Dashboard") as demo:
    gr.Markdown("# 🏥 Polyp Clinical Dashboard — Segmentation + Explainability")
    gr.Markdown("Upload an image (or paste). Use **Run Prediction**. Save a case to archive and download zipped evidence for reports.")

    with gr.Row():
        inp = gr.Image(type="numpy", label="Upload Colonoscopy Image")
        with gr.Column():
            run_btn = gr.Button("Run Prediction")
            save_btn = gr.Button("Save Case")
            history_btn = gr.Button("Refresh History")
            download_output = gr.File(label="Download last saved case (zip)")

            gr.Markdown("### Saved Cases")
            history = gr.Dropdown(choices=[x[0] for x in list_saved_cases()], label="Saved cases", interactive=True)
            dl_btn = gr.Button("Download Selected Case")

    with gr.Row():
        mask_out = gr.Image(label="Segmentation Mask", interactive=False)
        overlay_out = gr.Image(label="Overlay (Mask on Image)", interactive=False)

    with gr.Row():
        heatmap_out = gr.Image(label="Grad-CAM Heatmap", interactive=False)
        cam_overlay_out = gr.Image(label="Grad-CAM Overlay", interactive=False)

    analysis_txt = gr.Textbox(label="Polyp Analysis", lines=8)

    # Wiring
    run_btn.click(fn=gr_predict, inputs=inp, outputs=[mask_out, overlay_out, heatmap_out, cam_overlay_out, analysis_txt])
    save_btn.click(fn=gr_save_case, inputs=None, outputs=[download_output, gr.Textbox(label="Save status")])
    history_btn.click(fn=lambda: [x[0] for x in list_saved_cases()], inputs=None, outputs=[history])

    def download_selected_case(name):
        if not name:
            return None
        # find path
        zname = f"{name}.zip" if not name.endswith(".zip") else name
        path = os.path.join(CASES_DIR, zname)
        if os.path.exists(path):
            return path
        return None

    dl_btn.click(fn=download_selected_case, inputs=[history], outputs=[download_output])

demo.launch()
