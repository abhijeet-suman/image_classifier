# Image Classification System

An interactive deep learning web application powered by **PyTorch** and **Gradio** to detect and classify images into 7 categories from uploaded images.
Categories: Miscellaneous', 'Candle', 'Snowman', 'Airplane', 'Christmas_Tree', 'Jacket'

---

## 🚀 Features

* **Multi-Image Upload**: Process multiple images simultaneously.
* **Deep Learning Inference**: Uses a pre-trained TorchScript model similar to DenseNet architecture.
* **Interactive UI**: Clean, view with thumbnail gallery and confidence predictions.
* **GPU Accelerated**: Optimized for ZeroGPU via Hugging Face.

---

## 📂 Project Structure

```text
├── app.py                     # Main Gradio application
├── best_model.pth             # TorchScript model weights
├── classes.pkl                # Data classes
└── requirements.txt           # Project dependencies
