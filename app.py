import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image, ImageOps
import cv2

# --- 1. 고정밀 모델 로드 (ViT-L/14) ---
@st.cache_resource
def load_high_res_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 가장 정밀도가 높은 모델 사용
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess, device

model, preprocess, device = load_high_res_model()

# --- 2. 정밀 분석 함수 ---
def get_advanced_features(image):
    # 중앙부 크롭 (배경 영향 최소화, 상품 중심 분석)
    w, h = image.size
    image = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

# 색상 분석 가중치 강화 (중앙 영역 집중)
def get_center_color(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    # 중앙 50% 영역만 추출하여 배경색 간섭 차단
    center_img = img_cv[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    hist = cv2.calcHist([center_img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. 종합 점수 산출 (가중치 조정) ---
# 디자인 일치도를 85%, 색상 일치도를 15%로 배분
# (디자인이 우선이고, 색상은 보조 지표로 사용)
def calculate_hybrid_score(ai_score, color_score):
    return (ai_score * 0.85) + (max(0, color_score) * 0.15)

# (이하 이미지 업로드 및 결과 출력 로직은 이전과 동일하되 위 함수들을 적용)