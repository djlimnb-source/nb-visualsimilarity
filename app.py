import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import cv2
import shutil

# --- 1. 모델 로드 (서버 안정을 위해 B/32 사용) ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # L/14 대신 B/32를 쓰고 알고리즘으로 정확도를 보강합니다.
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 정밀 분석 로직 ---
def get_features(image):
    # [정밀도 향상 비결 1] 중앙 크롭
    # 배경 노이즈를 줄이기 위해 이미지의 중앙 80%만 사용합니다.
    w, h = image.size
    image = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def get_color_score(image):
    # [정밀도 향상 비결 2] 중앙 영역 색상 분석
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    center = img_cv[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    hist = cv2.calcHist([center], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. 데이터 및 폴더 설정 ---
IMAGE_DIR = "uploaded_samples"
if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)

@st.cache_data
def build_index(image_list):
    if not image_list: return None, None
    ai_vecs, color_vecs = [], []
    for p in image_list:
        img = Image.open(p).convert("RGB")
        ai_vecs.append(get_features(img))
        color_vecs.append(get_color_score(img))
    
    ai_vecs = np.vstack(ai_vecs).astype('float32')
    index = faiss.IndexFlatIP(ai_vecs.shape[1])
    index.add(ai_vecs)
    return index, color_vecs

# --- 4. UI 구성 ---
st.set_page_config(layout="wide")
st.title("🎯 정밀 상품 추천 시스템")

with st.sidebar:
    st.header("DB 관리")
    ups = st.file_uploader("상품 등록", type=['jpg','png'], accept_multiple_files=True)
    if st.button("등록"):
        for u in ups:
            with open(os.path.join(IMAGE_DIR, u.name), "wb") as f: f.write(u.getbuffer())
        st.cache_data.clear(); st.rerun()
    if st.button("초기화"):
        shutil.rmtree(IMAGE_DIR); os.makedirs(IMAGE_DIR); st.cache_data.clear(); st.rerun()

all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.info("왼쪽에서 상품 사진을 먼저 등록하세요.")
else:
    idx, c_db = build_index(all_files)
    st.subheader("검색 이미지 업로드")
    q_file = st.file_uploader("이미지 선택", type=['jpg','png'])

    if q_file:
        q_img = Image.open(q_file).convert("RGB")
        q_ai = get_features(q_img).astype('float32')
        q_col = get_color_score(q_img)
        
        # 전체 검색 후 가중치 합산 (디자인 80% + 색상 20%)
        dists, inds = idx.search(q_ai, len(all_files))
        results = []
        for i in range(len(all_files)):
            curr_idx = inds[0][i]
            ai_s = dists[0][i]
            col_s = cv2.compareHist(q_col, c_db[curr_idx], cv2.HISTCMP_CORREL)
            final_s = (ai_s * 0.8) + (max(0, col_s) * 0.2)
            results.append((curr_idx, final_s, ai_s, col_s))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        cols = st.columns(3)
        for i in range(min(3, len(results))):
            idx_res, f_s, a_s, c_s = results[i]
            with cols[i]:
                st.image(all_files[idx_res], use_container_width=True)
                st.metric(f"순위 {i+1}", f"{f_s:.2f}")
                st.caption(f"디자인: {a_s:.2f} / 색상: {c_s:.2f}")