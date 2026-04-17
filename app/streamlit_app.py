import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Retinal Vessel Segmentation", layout="wide")

# -------- LOAD MODEL --------
model = joblib.load("models/model.pkl")

# -------- FUNCTIONS --------

def get_clahe_image(img):
    green = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(green)


def create_matched_filter(size=15, sigma=2):
    kernel = np.zeros((size, size))
    center = size // 2

    for x in range(size):
        for y in range(size):
            kernel[x, y] = -np.exp(-((y - center)**2) / (2 * sigma**2))

    kernel = kernel - np.mean(kernel)
    return kernel


def apply_matched_filter(img):
    responses = []
    kernel = create_matched_filter()

    for angle in range(0, 180, 15):
        M = cv2.getRotationMatrix2D((kernel.shape[1]//2, kernel.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(kernel, M, (kernel.shape[1], kernel.shape[0]))

        res = cv2.filter2D(img, cv2.CV_32F, rotated)
        responses.append(res)

    response = np.max(responses, axis=0)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    return response.astype(np.uint8)


def extract_features(img):
    clahe_img = get_clahe_image(img)

    mf_img = apply_matched_filter(clahe_img)

    sobelx = cv2.Sobel(clahe_img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(clahe_img, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur = cv2.GaussianBlur(clahe_img, (5,5), 0)
    variance = cv2.absdiff(clahe_img, blur)

    return np.stack([clahe_img, mf_img, gradient, variance], axis=-1)


def predict_image(img):
    feat = extract_features(img)
    h, w, _ = feat.shape

    pred = model.predict(feat.reshape(-1, feat.shape[-1]))
    return pred.reshape(h, w)


def post_process(pred):
    pred = pred.astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)

    closed = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    return dilated

# -------- UI --------

# Header
st.title("🩺 Retinal Vessel Segmentation")
st.caption("AI-based detection of blood vessels in fundus images • BITS F311 Image Processing Coursework")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.caption("Developed as part of BITS F311 Image Processing coursework")
    st.write("This app segments retinal blood vessels using classical image processing + AdaBoost.")

    st.subheader("👥 Team")
    st.write("Pranav Deshpande")
    st.write("Mehul Goel")
    st.write("Nakshatara Garg")

    st.subheader("⚙️ Pipeline")
    st.markdown("""
    - CLAHE enhancement
    - Matched filtering
    - Gradient + variance features
    - AdaBoost classification
    - Morphological refinement
    """)

# Main section
st.markdown("### Upload Fundus Image")
uploaded_file = st.file_uploader("", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with st.spinner("Processing image..."):
        pred = predict_image(img)
        clean = post_process(pred)

    st.success("Segmentation complete!")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(clean, caption="Vessel Segmentation", use_container_width=True)

else:
    st.info("Upload an image to get started.")
