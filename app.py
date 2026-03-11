import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import shutil

# --- 1. 환경 설정 및 모델 로드 ---
st.set_page_config(page_title="AI 상품 추천 시스템", layout="wide")
st.title("🖼️ AI 비주얼 검색 & 상품 관리 시스템")

# 배포 안정성을 위해 ViT-B/32 사용 (ViT-L/14는 서버 메모리 초과 위험)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 설정 및 폴더 준비 ---
IMAGE_DIR = "uploaded_samples"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# --- 3. 핵심 로직 함수 ---
def get_vector(image):
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

@st.cache_data
def build_index(image_list):
    if not image_list:
        return None
    vectors = []
    for img_path in image_list:
        img = Image.open(img_path).convert("RGB")
        vectors.append(get_vector(img))
    
    vectors = np.vstack(vectors).astype('float32')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

# --- 4. 사이드바: 샘플 이미지 관리 (DB 등록) ---
with st.sidebar:
    st.header("📦 상품 DB 관리")
    uploaded_samples = st.file_uploader("샘플 이미지들을 업로드하세요 (여러 장 가능)", 
                                        type=['jpg', 'jpeg', 'png'], 
                                        accept_multiple_files=True)
    
    if st.button("서버에 상품 등록/업데이트"):
        if uploaded_samples:
            for uploaded_item in uploaded_samples:
                file_path = os.path.join(IMAGE_DIR, uploaded_item.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_item.getbuffer())
            st.success(f"{len(uploaded_samples)}개의 상품이 등록되었습니다!")
            st.cache_data.clear() # 인덱스 갱신을 위해 캐시 삭제
            st.rerun()
        else:
            st.error("업로드할 파일을 먼저 선택해주세요.")

    if st.button("모든 샘플 삭제"):
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
            os.makedirs(IMAGE_DIR)
            st.cache_data.clear()
            st.warning("DB가 초기화되었습니다.")
            st.rerun()

# --- 5. 메인 화면: 검색 및 결과 ---
all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.info("👈 왼쪽 사이드바에서 비교할 상품 샘플들을 먼저 업로드해주세요.")
else:
    st.write(f"현재 등록된 상품 수: **{len(all_files)}개**")
    
    with st.spinner("이미지 분석 중..."):
        index = build_index(all_files)

    st.divider()
    
    # 기준 이미지 업로드
    st.subheader("🔎