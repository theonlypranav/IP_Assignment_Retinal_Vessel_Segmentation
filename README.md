# Supervised Retinal Vessel Segmentation using Matched Filtering and AdaBoost

**Course:** BITS F311 – Image Processing  
**Assignment:** PID 24  
**Date:** 25 April 2026  

## Team Members
- Pranav Deshpande (2022B3A70446P)  
- Mehul Goel (2023A7PS0446P)  
- Nakshatara Garg (2023A7PS0634P)  

---

## 📌 Problem Statement

This project focuses on automated retinal vessel segmentation from color fundus images, based on the work by Memari et al. It lies in the domain of ophthalmoscopy and aims to assist in diagnosing diseases such as diabetes, glaucoma, and hypertension.

Manual inspection of retinal vessels is time-consuming and inconsistent. Variations in lighting, noise, and subtle vessel structures make detection difficult, especially for thin vessels that are critical for early diagnosis.

The goal is to automate this process using image processing and machine learning, enabling accurate extraction and analysis of vessel features such as thickness, curvature, and abnormalities.

---

## ⚙️ Methodology

### Dataset
- Used the **DRIVE dataset** (40 retinal images with ground truth annotations).

### Pipeline Overview
1. **Pre-processing**
   - Extract Green channel (best contrast for vessels)
   - Apply Median Filter (noise reduction)
   - Apply CLAHE (contrast enhancement)
   - Apply Retinex (illumination correction)

2. **Vessel Enhancement**
   - Matched Filtering (detects vessel-like structures in all directions)

3. **Feature Extraction**
   - Intensity features: mean, variance, skewness, kurtosis  
   - Texture features: contrast, correlation, entropy, homogeneity  
   - Structural features: run-length metrics  
   - Frequency features: Gabor filters  

   → Top 10 features selected from 35

4. **Classification**
   - AdaBoost classifier used for pixel-wise classification

5. **Post-processing**
   - Removes small noisy regions
   - Improves segmentation quality

6. **Evaluation**
   - Compared with SVM, k-NN using 5-fold cross-validation  
   - AdaBoost performed best

---

## ✅ Pros and ❌ Cons

### Pros
- Well-designed hybrid pipeline (image processing + ML)
- Handles noise, lighting variation, and vessel thickness effectively
- Good generalization across datasets
- Efficient feature selection reduces computation
- AdaBoost is fast and effective for structured features

### Cons
- Heavy reliance on manual feature engineering
- Complex and hard to modify
- Feature extraction is time-consuming
- Modern deep learning models (e.g., U-Net) are more powerful and flexible

---

## 🔍 Research Gaps

- Small vessel segments are often removed as noise → loss of important details  
- Lack of vessel continuity in final segmentation  
- Pixel-wise classification does not guarantee structural correctness  
- No strong post-processing for reconstructing vessel geometry  

---

## 💡 Novelty

We introduce improvements in the **post-processing stage**:

- Apply **Morphological Closing** to connect broken vessel segments  
- Apply **Dilation** to enhance thin vessels  

### Key Idea:
Shift focus from *noise removal* → *structure reconstruction*

### Important Note:
- AdaBoost model is treated as a **black box**
- No changes made to classifier
- Improvements are purely image-processing based

---

## 🛠️ Our Implementation

- Built a **simplified version** of the original pipeline
- Focused on:
  - Pre-processing (Green channel + CLAHE)
  - Matched filtering
  - Feature extraction
  - AdaBoost classification
  - Post-processing (our contribution)

### Additional Work
- Implemented dataset handling (masking, sampling)
- Developed complete pipeline in Jupyter Notebook

---

## 🌐 Deployment

- Deployed using **Streamlit**
- Features:
  - Upload retinal image
  - Get segmented vessel output instantly
- Makes the model easy to demonstrate and use

---

## 🤖 AI Usage Notes

- Used AI for:
  - Matched filter kernel generation (Gaussian profiles, rotations)
  - AdaBoost implementation
  - Post-processing method selection

- Reason:
  - Limited prior experience with ML and deep learning concepts

---

## 👥 Member Contributions

### Mehul Goel
- Documentation and report structuring  
- Planned presentation flow  
- Suggested Streamlit deployment  

### Nakshatara Garg
- Designed presentation  
- Researched pre-processing and post-processing steps  

### Pranav Deshpande
- Implemented AdaBoost and feature pipeline  
- Analyzed pros/cons and identified research gaps  

---

## 📚 References

- Memari et al. – Original Research Paper  
- U-Net Paper: https://arxiv.org/abs/1505.04597  
- Optimized U-Net: https://www.nature.com/articles/s41598-026-48475-6  

---

## 🚀 Conclusion

This project demonstrates a practical implementation of retinal vessel segmentation using classical image processing and machine learning techniques. While effective, it highlights the importance of structural preservation and opens avenues for improvement using better post-processing or deep learning approaches.
