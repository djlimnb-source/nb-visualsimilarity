import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import cv2
import shutil
from rembg import remove  # 배경 제거 라이브러리

# --- 1. 모델 및 설정 (캐싱 적용) ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 이미지 분석 핵심 로직 ---

def get_clean_image(image):
    """배경을 제거하여 '상품 자체'의 특징만 남김 (빨간 소매 등 상품 요소는 보존)"""
    # rembg는 알파 채널(RGBA)을 생성하므로 RGB로 변환하여 분석에 사용
    rmbg_img = remove(image)
    return rmbg_img.convert("RGB")

def get_features(image):
    """CLIP 기반 디자인/스타일 특징 추출"""
    clean_img = get_clean_image(image)
    w, h = clean_img.size
    # [정밀도] 상품 본체에 집중하기 위해 중앙 80% 크롭 (배경 노이즈 최소화)
    img_cropped = clean_img.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    
    img_input = preprocess(img_cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def get_color_score(image):
    """3D RGB 히스토그램 기반 색상 분석"""
    clean_img = get_clean_image(image)
    img_cv = cv2.cvtColor(np.array(clean_img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    # [정밀도] 그래픽 및 핵심 색상 파악을 위해 중앙 60% 집중 분석
    center = img_cv[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    hist = cv2.calcHist([center], [0, 1, 2], None, [12, 12, 12], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. 데이터베이스 구축 ---
IMAGE_DIR = "uploaded_samples"
if not os.path.exists(IMAGE_DIR): 
    os.makedirs(IMAGE_DIR)

@st.cache_data
def build_index(image_list):
    if not image_list: return None, None
    ai_vecs, color_vecs = [], []
    for p in image_list:
        try:
            img = Image.open(p).convert("RGB")
            ai_vecs.append(get_features(img))
            color_vecs.append(get_color_score(img))
        except Exception as e:
            continue
    
    if not ai_vecs: return None, None
    ai_vecs = np.vstack(ai_vecs).astype('float32')
    index = faiss.IndexFlatIP(ai_vecs.shape[1])
    index.add(ai_vecs)
    return index, color_vecs

# --- 4. Streamlit UI 레이아웃 ---
st.set_page_config(page_title="AI 상품 정밀 분석 시스템", layout="wide")
st.title("🎨 색상 중심 초정밀 상품 검색")
st.info("배경 제거(RemBG) 기술을 통해 상품 본연의 색상과 디자인을 분석합니다.")

with st.sidebar:
    st.header("📦 상품 DB 관리")
    ups = st.file_uploader("상품 이미지 등록", type=['jpg','png','jpeg'], accept_multiple_files=True)
    if st.button("서버에 신규 등록"):
        if ups:
            with st.status("이미지 분석 및 배경 제거 중...", expanded=False) as status:
                for u in ups:
                    with open(os.path.join(IMAGE_DIR, u.name), "wb") as f:
                        f.write(u.getbuffer())
                st.cache_data.clear()
                status.update(label="등록 완료!", state="complete")
            st.rerun()

    if st.button("DB 전체 삭제"):
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
            os.makedirs(IMAGE_DIR)
            st.cache_data.clear()
            st.rerun()

all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.warning("👈 먼저 사이드바에서 상품 이미지를 등록해 주세요.")
else:
    idx_db, c_db = build_index(all_files)
    
    st.subheader("🔎 검색 기준 이미지")
    q_file = st.file_uploader("유사한 상품을 찾을 이미지를 올려주세요", type=['jpg','png','jpeg'])

    if q_file:
        q_img = Image.open(q_file).convert("RGB")
        
        # 기준 이미지 표시
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.image(q_img, caption="검색 기준 (원본)", use_container_width=True)
        st.divider()

        # 분석 진행
        q_ai = get_features(q_img).astype('float32')
        q_col = get_color_score(q_img)
        
        total_count = len(all_files)
        dists, inds = idx_db.search(q_ai, total_count)
        
        results = []
        for i in range(total_count):
            curr_idx = inds[0][i]
            ai_s = float(dists[0][i])
            col_s = float(cv2.compareHist(q_col, c_db[curr_idx], cv2.HISTCMP_CORREL))
            
            # [가중치 고도화] 색상 90% : 디자인 10%
            final_s = (max(0, col_s) * 0.9) + (ai_s * 0.1)
            results.append((curr_idx, final_s, ai_s, col_s))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        st.subheader(f"🏆 유사도 랭킹 (총 {total_count}개 상품)")
        
        # 4열 그리드 출력
        for i in range(0, total_count, 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < total_count:
                    res_idx, f_s, a_s, c_s = results[i + j]
                    with cols[j]:
                        st.image(all_files[res_idx], use_container_width=True)
                        st.markdown(f"**{i + j + 1}위** (적합도: {f_s:.2%})")
                        
                        # 점수 시각화
                        st.caption(f"🎨 색상 일치도: {max(0, c_s):.2f}")
                        st.progress(max(0.0, min(1.0, c_s)))
                        st.caption(f"👗 디자인 일치도: {a_s:.2f}")
                        st.progress(max(0.0, min(1.0, a_s)))
                        st.divider()