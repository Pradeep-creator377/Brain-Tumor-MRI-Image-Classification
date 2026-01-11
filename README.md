# 🧠 Brain Tumor MRI Image Classification

## 📌 Project Overview

This project uses a **ResNet50 deep learning model** to classify brain MRI images into four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary

The application is built with **Streamlit** to provide an interactive web interface where users can upload MRI images and receive predictions along with confidence scores and visualizations.

---

## 🚀 Features

- Upload MRI images (`.jpg`, `.jpeg`, `.png`) via a simple web interface.
- Automatic preprocessing (resize, RGB conversion, ResNet50 normalization).
- Model prediction with confidence scores for each tumor type.
- Visualization of confidence scores using bar charts.
- Error handling for missing files or invalid inputs.

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras (ResNet50 model)
- Streamlit (web app framework)
- NumPy, PIL, Matplotlib (image handling & visualization)

---

## 📂 Project Structure 

```
Brain-Tumor-Classification/
│
├── best_ResNet50.h5     # Trained ResNet50 model
├── app.py               # Streamlit application code
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
pip install tensorflow streamlit numpy pillow matplotlib
streamlit run app.py
```

---

## 📊 Usage

- Launch the app with `streamlit run app.py`.
- Upload an MRI image.
- View the predicted tumor type and confidence scores.
- Check the bar chart visualization for probability distribution across classes.

---

## 📈 Model Details

- Base model: ResNet50 (pretrained on ImageNet).
- Fine-tuned for brain tumor classification.
- Input size: 224 × 224 × 3 (RGB).
- Output: 4 softmax probabilities corresponding to tumor classes.
