import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import shutil
import cv2  # 색상 분석용

# --- 1. 환경 설정 및 모델 로드 ---
st.set_page_config(page_title="AI 상품 추천 시스템", layout="wide")
st.title("🎨 AI 스타일 + 색상 분석 통합 추천")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 안정적인 ViT-B/32 모델 사용
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 설정 및 폴더 준비 ---
IMAGE_DIR = "uploaded_samples"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# --- 3. 핵심 로직 함수 ---

# [색상 분석용] 이미지에서 컬러 히스토그램 추출
def get_color_histogram(image):
    # PIL 이미지를 OpenCV 형식(BGR)으로 변환
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 이미지 크기 축소 (계산 속도 향상)
    img_cv = cv2.resize(img_cv, (100, 100))
    # 히스토그램 계산 (3채널)
    hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def get_vector(image):
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

@st.cache_data
def build_index(image_list):
    if not image_list: return None, None
    
    ai_vectors = []
    color_vectors = []
    
    for img_path in image_list:
        img = Image.open(img_path).convert("RGB")
        ai_vectors.append(get_vector(img))
        color_vectors.append(get_color_histogram(img))
        
    ai_vectors = np.vstack(ai_vectors).astype('float32')
    # AI 스타일 인덱스 (Faiss)
    index = faiss.IndexFlatIP(ai_vectors.shape[1])
    index.add(ai_vectors)
    
    return index, color_vectors

# --- 4. 사이드바: 관리 기능 생략 (이전과 동일) ---
with st.sidebar:
    st.header("📦 상품 DB 관리")
    uploaded_samples = st.file_uploader("샘플 이미지 업로드", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if st.button("서버에 상품 등록"):
        if uploaded_samples:
            for item in uploaded_samples:
                with open(os.path.join(IMAGE_DIR, item.name), "wb") as f:
                    f.write(item.getbuffer())
            st.cache_data.clear()
            st.rerun()
    if st.button("DB 초기화"):
        shutil.rmtree(IMAGE_DIR); os.makedirs(IMAGE_DIR)
        st.cache_data.clear(); st.rerun()

# --- 5. 메인 화면: 하이브리드 검색 ---
all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.info("👈 왼쪽에서 상품 샘플을 먼저 등록해 주세요.")
else:
    with st.spinner("상품 데이터 정밀 분석 중..."):
        ai_index, color_db = build_index(all_files)

    search_file = st.file_uploader("이 이미지와 닮은 상품 찾기", type=['jpg', 'jpeg', 'png'])

    if search_file:
        query_img = Image.open(search_file).convert("RGB")
        query_ai_vec = get_vector(query_img).astype('float32')
        query_color_vec = get_color_histogram(query_img)

        # 1. AI 유사도 계산
        ai_scores, ai_indices = ai_index.search(query_ai_vec, len(all_files))
        
        # 2. 색상 유사도 계산 및 점수 합산
        final_results = []
        for i in range(len(all_files)):
            idx = ai_indices[0][i]
            ai_score = ai_scores[0][i]
            # 히스토그램 상관관계 계산 (0~1 사이)
            color_score = cv2.compareHist(query_color_vec, color_db[idx], cv2.HISTCMP_CORREL)
            
            # 최종 점수 = AI(70%) + 색상(30%)
            total_score = (ai_score * 0.7) + (max(0, color_score) * 0.3)
            final_results.append((idx, total_score, ai_score, color_score))

        # 점수 기준 정렬
        final_results.sort(key=lambda x: x[1], reverse=True)

        # 결과 출력
        st.subheader("🚀 하이브리드 추천 결과")
        cols = st.columns(3)
        for i in range(min(3, len(final_results))):
            idx, t_score, a_score, c_score = final_results[i]
            with cols[i]:
                st.image(all_files[idx], use_container_width=True)
                st.write(f"**추천 {i+1}순위**")
                st.caption(f"종합 일치도: {t_score:.2f}")
                
                # 분석 근거
                with st.expander("분석 데이터 보기"):
                    st.write(f"- 스타일 점수: {a_score:.2f}")
                    st.write(f"- 색상 일치도: {c_score:.2f}")
                    if c_score > 0.8: st.write("✅ 색상이 매우 유사함")
                    if a_score > 0.8: st.write("✅ 디자인이 매우 유사함")