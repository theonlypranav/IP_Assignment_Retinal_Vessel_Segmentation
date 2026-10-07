# 🩺 Retinal Vessel Segmentation
### AI-Powered Detection of Blood Vessels in Fundus Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen?style=flat-square&logo=streamlit)](https://theonlypranav-ip-assignment-retinal-vessel-segmentation.streamlit.app)
[![Course](https://img.shields.io/badge/Course-BITS%20F311-orange?style=flat-square)](https://www.bits-pilani.ac.in/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Automated diagnosis assistance for diabetic retinopathy, glaucoma, and hypertension through classical image processing and machine learning.**

---

## 🚀 Live Demo
**Try it now:** [Retinal Vessel Segmentation App](https://theonlypranav-ip-assignment-retinal-vessel-segmentation.streamlit.app)

Upload a fundus image and get instant vessel segmentation results!

---

## 🎯 Problem Statement

Manual retinal vessel inspection is:
- ⏱️ **Time-consuming** — requires trained ophthalmologists
- 🔴 **Inconsistent** — varies between analysts  
- ❌ **Error-prone** — difficult to spot thin vessels (critical for early diagnosis)

**Our solution:** An automated, accurate pipeline that detects blood vessels in retinal images to assist in diagnosing:
- 🩸 Diabetic Retinopathy
- 👁️ Glaucoma
- 💔 Hypertension

---

## 🏗️ Architecture & Pipeline

```
📸 Input Image
    ↓
🎨 Pre-processing (Green channel + CLAHE)
    ↓
🔍 Matched Filtering (Multi-orientation vessel detection)
    ↓
📊 Feature Extraction (Intensity + Texture + Structural)
    ↓
🤖 AdaBoost Classification (Pixel-wise prediction)
    ↓
✨ Post-processing (Morphological refinement)
    ↓
✅ Vessel Segmentation Output
```

### 📋 Dataset
- **DRIVE Dataset** — 40 retinal fundus images with expert-annotated ground truth

### 🔧 Pipeline Stages

#### 1. Pre-processing
- ✓ Extract green channel (best vessel contrast)
- ✓ Apply CLAHE (adaptive histogram equalization)
- ✓ Median filtering for noise reduction

#### 2. Vessel Enhancement  
- ✓ Multi-orientation matched filtering (0°-165° at 15° intervals)
- ✓ Detects Gaussian vessel-like structures

#### 3. Feature Extraction (4 categories)
| Category | Features | Count |
|----------|----------|-------|
| **Intensity** | Mean, variance, skewness, kurtosis | 4 |
| **Texture** | Contrast, correlation, entropy, homogeneity | 4 |
| **Structural** | Run-length emphasis metrics | 3 |
| **Frequency** | Gabor filter responses | 24 |
| | **Total selected** | **10** |

#### 4. Classification
- ✓ **AdaBoost** classifier (pixel-wise prediction)
- ✓ Compared vs. SVM & k-NN (5-fold CV)
- ✓ **Best performer** across all metrics

#### 5. Post-processing (Our Innovation 💡)
- ✓ Morphological closing (reconnect broken segments)
- ✓ Dilation (enhance thin vessels)
- ✓ Remove small noise artifacts

---

## 📊 Results & Performance

| Metric | AdaBoost | SVM | k-NN |
|--------|----------|-----|------|
| **Accuracy** | 94.2% | 92.8% | 89.5% |
| **Sensitivity** | 0.87 | 0.84 | 0.78 |
| **Specificity** | 0.96 | 0.94 | 0.92 |
| **AUC-ROC** | 0.941 | 0.923 | 0.897 |

✅ **AdaBoost wins on generalization and real-world performance**

---

## ✨ Key Innovations

### Our Contribution: Advanced Post-Processing
Instead of just removing noise, we **reconstruct vessel geometry**:

1. **Morphological Closing** → Reconnect broken vessel segments
2. **Selective Dilation** → Enhance thin vessels without over-smoothing
3. **Artifact Removal** → Smart filtering of false positives

**Result:** Better vessel continuity and preservation of clinically important thin vessels.

---

## ⚡ Strengths & Limitations

### ✅ Strengths
- **Robust hybrid approach** — combines classical image processing + ML
- **Handles real-world challenges** — noise, lighting variation, vessel thickness
- **Fast inference** — no GPU needed
- **Explainable pipeline** — each step is interpretable

### ❌ Limitations  
- **Manual feature engineering** — requires domain expertise
- **Hard to extend** — adding new features requires pipeline redesign
- **Scale bias** — optimized for DRIVE dataset; may need tuning for other sources
- **Deep learning comparison** — modern U-Net/SegNet models are more flexible

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/theonlypranav/IP_Assignment_Retinal_Vessel_Segmentation.git
cd IP_Assignment_Retinal_Vessel_Segmentation
pip install -r requirements.txt
```

### Run Streamlit App (Live Demo)
```bash
streamlit run streamlit_app.py
```
Then open: `http://localhost:8501`

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/main.ipynb
```

---

## 🌐 Deployment

**Live on Streamlit Cloud:**
🔗 https://theonlypranav-ip-assignment-retinal-vessel-segmentation.streamlit.app

**Features:**
- 📸 Upload retinal fundus image
- ⚡ Instant AI-powered segmentation
- 🎨 Side-by-side comparison (original vs segmented)
- 🌙 Dark mode UI

---

## 📁 Project Structure

```
.
├── streamlit_app.py              # 🌐 Live web app
├── requirements.txt              # 📦 Dependencies
├── models/
│   └── model.pkl                # 🤖 Trained AdaBoost classifier
├── data/
│   └── DRIVE/                   # 📊 Dataset
├── notebooks/
│   └── main.ipynb               # 📓 Full pipeline & analysis
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── classification.py
└── README.md
```

---

## 📚 References

- **Memari et al.** — Original retinal vessel segmentation paper
- **U-Net Architecture** — https://arxiv.org/abs/1505.04597
- **DRIVE Dataset** — https://drive.grand-challenge.org/
- **AdaBoost Overview** — Scikit-learn documentation

---

## 👥 Team

| Name | Role | Contribution |
|------|------|--------------|
| **Pranav Deshpande** | Lead Developer | Pipeline implementation, AdaBoost classifier, performance analysis |
| **Mehul Goel** | Documentation & Strategy | Report structure, deployment planning, Streamlit integration |
| **Nakshatara Garg** | Research & Design | Pre/post-processing research, presentation design |

**Course:** BITS F311 – Image Processing  
**Academic Year:** 2025-2026

---

## 🎓 Conclusion

This project demonstrates a **production-grade implementation** of retinal vessel segmentation combining:
- ✅ Classical image processing expertise
- ✅ Machine learning best practices  
- ✅ Real-world deployment on Streamlit Cloud
- ✅ Novel post-processing innovations

**Key Takeaway:** While modern deep learning (U-Net) offers flexibility, our classical approach delivers:
- Interpretability
- Fast inference
- Low computational requirements
- Reliable performance on the DRIVE dataset

**Future Work:**
- 🔮 Ensemble models (classical + deep learning)
- 🔮 Domain adaptation for other retinal datasets
- 🔮 3D vessel structure reconstruction
- 🔮 Real-time video analysis for clinical use

---

## 📜 License

This project is licensed under the MIT License — see LICENSE file for details.

---

<div align="center">

### ⭐ If you found this useful, please star the repo!

Made with ❤️ by the IP Assignment team at BITS Pilani

</div>
