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

# 배포 안정성을 위해 ViT-B/32 사용 (무료 서버 메모리 최적화)
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

# --- 4. 사이드바: 상품 DB 관리 ---
with st.sidebar:
    st.header("📦 상품 DB 관리")
    uploaded_samples = st.file_uploader("샘플 이미지 업로드 (다중 선택 가능)", 
                                        type=['jpg', 'jpeg', 'png'], 
                                        accept_multiple_files=True)
    
    if st.button("서버에 상품 등록"):
        if uploaded_samples:
            for uploaded_item in uploaded_samples:
                file_path = os.path.join(IMAGE_DIR, uploaded_item.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_item.getbuffer())
            st.success(f"{len(uploaded_samples)}개의 상품이 등록되었습니다!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("파일을 먼저 선택해주세요.")

    if st.button("DB 초기화 (전체 삭제)"):
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
            os.makedirs(IMAGE_DIR)
            st.cache_data.clear()
            st.warning("모든 데이터가 삭제되었습니다.")
            st.rerun()

# --- 5. 메인 화면: 검색 및 결과 ---
all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.info("👈 왼쪽 사이드바에서 비교할 상품들을 먼저 업로드해 주세요.")
else:
    st.write(f"현재 등록된 상품 수: **{len(all_files)}개**")
    
    with st.spinner("이미지 분석 중..."):
        index = build_index(all_files)

    st.divider()
    
    st.subheader("🔎 검색할 이미지 올리기")
    search_file = st.file_uploader("이 이미지와 닮은 상품 찾기", type=['jpg', 'jpeg', 'png'], key="search")

    if search_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**내가 올린 이미지**")
            query_img = Image.open(search_file).convert("RGB")
            st.image(query_img, use_container_width=True)

        with col2:
            st.write("**추천 결과**")
            query_vec = get_vector(query_img).astype('float32')
            
            k = min(len(all_files), 4)
            distances, indices = index.search(query_vec, k)
            
            res_cols = st.columns(k)
            for i, idx in enumerate(indices[0]):
                with res_cols[i]:
                    matched_img = Image.open(all_files[idx])
                    st.image(matched_img, caption=f"순위 {i+1} ({distances[0][i]:.2f})", use_container_width=True)