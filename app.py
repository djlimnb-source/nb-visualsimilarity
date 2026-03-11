import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image

# --- 1. 환경 설정 및 모델 로드 ---
st.set_page_config(page_title="AI 상품 추천 데모", layout="wide")
st.title("🔍 AI 시각적 유사도 상품 추천")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 이미지 벡터 추출 함수 ---
def get_vector(image):
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

# --- 3. 데이터 인덱싱 (이미 업로드된 상품들 분석) ---
IMAGE_DIR = "sample_images"

@st.cache_data
def build_index(image_list):
    vectors = []
    for img_path in image_list:
        img = Image.open(img_path).convert("RGB")
        vectors.append(get_vector(img))
    
    vectors = np.vstack(vectors).astype('float32')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

# --- 4. 메인 화면 구성 ---
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# 비교 대상 파일 목록
all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.warning(f"⚠️ '{IMAGE_DIR}' 폴더에 상품 이미지를 먼저 넣어주세요.")
else:
    # 인덱스 구축
    with st.spinner("상품 데이터를 분석 중입니다..."):
        index = build_index(all_files)
    
    # 이미지 업로드 UI
    uploaded_file = st.file_uploader("찾고 싶은 상품 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("내가 올린 상품")
            query_img = Image.open(uploaded_file).convert("RGB")
            st.image(query_img, use_container_width=True)

        with col2:
            st.subheader("추천 결과 (가장 닮은 상품)")
            query_vec = get_vector(query_img).astype('float32')
            
            # 유사도 검색 (상위 4개)
            k = 4
            distances, indices = index.search(query_vec, k)
            
            # 결과 출력
            res_cols = st.columns(k)
            for i, idx in enumerate(indices[0]):
                with res_cols[i]:
                    matched_img = Image.open(all_files[idx])
                    st.image(matched_img, caption=f"추천 {i+1}순위", use_container_width=True)
                    st.write(f"유사도: {distances[0][i]:.2f}")