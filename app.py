import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import cv2
import shutil

# --- 1. 모델 로딩 ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 분석 함수 ---
def get_features(image):
    w, h = image.size
    # 중앙 크롭 (배경 노이즈 제거)
    image = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def get_color_score(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    # 상품 본연의 색상 추출 (중앙 60% 영역)
    center = img_cv[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    hist = cv2.calcHist([center], [0, 1, 2], None, [12, 12, 12], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. 데이터 설정 ---
IMAGE_DIR = "uploaded_samples"
if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)

@st.cache_data
def build_index(image_list):
    if not image_list: return None, None
    ai_vecs, color_vecs = [], []
    for p in image_list:
        img = Image.open(