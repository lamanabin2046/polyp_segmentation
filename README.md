# 🩺 Polyp Segmentation & Explainability System  
### UNet++ (EfficientNet-B3) • Medical Image Segmentation • GradCAM • Polyp Size Analysis • Gradio Dashboard

This repository provides a complete **medical AI pipeline** for colon polyp detection and segmentation using deep learning.  
The system includes:

- ⚡ High-accuracy UNet++ segmentation  
- 🔥 GradCAM explainability  
- 📏 Morphological analysis (size, diameter, irregularity)  
- 🧠 Confidence & risk scoring (non-diagnostic)  
- 🖥️ Interactive Gradio web dashboard  
- 🧪 Training + Evaluation scripts  
- 🌍 Ready for deployment on Hugging Face Spaces  
- 📚 Full documentation & reproducible environment  

---

# 📘 **Abstract**

Colorectal cancer often begins as benign polyps that are frequently missed during colonoscopy.  
This project presents a **UNet++ segmentation model with EfficientNet-B3 encoder**, trained on the **Kvasir-SEG dataset** for accurate polyp identification. In addition to segmentation, the system provides:

- **GradCAM-based visual explainability**
- **Polyp area estimation**
- **Approximate diameter**
- **Shape irregularity score**
- **Model confidence**
- **A non-diagnostic risk indicator**

The goal is **research and educational use**, not clinical diagnosis.  
A fully interactive dashboard built with **Gradio** enables real-time visual analysis and interpretability.  

---

# 🌟 **Features**

### ✔ UNet++ Polyp Segmentation
- EfficientNet-B3 encoder  
- Hybrid loss (Dice + BCE)  
- Achieves **Dice ≈ 0.95**, **IoU ≈ 0.90**

### ✔ Grad-CAM Explainability  
Highlights **where the model focuses**, helping clinicians and students understand model reasoning.

### ✔ Polyp Morphology Analysis  
Automatically computes:

- 🟩 Polyp Area (in pixels)  
- ▫️ Approx Diameter  
- 🌀 Shape Irregularity Score  
- 🔍 Model Confidence  
- 🚨 Risk Indicator (Low / Medium / High — NOT diagnostic)

### ✔ Gradio Dashboard
Upload an image → Get:

| Output | Description |
|--------|------------|
| 🧩 Segmentation Mask | Predicted polyp mask |
| 🎨 Overlay | Mask + original image |
| 🔥 GradCAM Heatmap | Model attention |
| 🔥+📸 CAM Overlay | Heatmap + original |
| 📄 Analysis Panel | Size, diameter, confidence, risk |

### ✔ Full ML Pipeline
- Dataset loader  
- Augmentations  
- Training  
- Evaluation (Dice, IoU, F1, CM)  
- Visualization scripts  

---

# 🧪 **Sample Outputs**

### Segmentation Mask
![mask](assets/mask_example.png)

### Overlay
![overlay](assets/overlay_example.png)

### GradCAM Heatmap
![gradcam](assets/gradcam_example.png)

### Analysis Panel
