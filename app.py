import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import cv2
import shutil

# --- 1. 모델 로딩 (서버 안정성 최적화) ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 무료 서버의 메모리 한계를 고려하여 B/32 모델 사용
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 정밀 분석 함수 (중앙 집중형) ---
def get_features(image):
    w, h = image.size
    # [정밀도] 테두리 배경 노이즈 제거를 위한 10% 크롭
    image = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def get_color_score(image):
    # [정밀도] 그래픽의 색상을 정확히 잡기 위해 중앙 영역(60%) 집중 분석
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    center = img_cv[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    hist = cv2.calcHist([center], [0, 1, 2], None, [12, 12, 12], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. 데이터 저장 설정 ---
IMAGE_DIR = "uploaded_samples"
if not os.path.exists(IMAGE_DIR): 
    os.makedirs(IMAGE_DIR)

@st.cache_data
def build_index(image_list):
    if not image_list: 
        return None, None
    ai_vecs, color_vecs = [], []
    for p in image_list:
        try:
            img = Image.open(p).convert("RGB")
            ai_vecs.append(get_features(img))
            color_vecs.append(get_color_score(img))
        except Exception:
            continue
    
    if not ai_vecs: return None, None
    ai_vecs = np.vstack(ai_vecs).astype('float32')
    index = faiss.IndexFlatIP(ai_vecs.shape[1])
    index.add(ai_vecs)
    return index, color_vecs

# --- 4. 메인 레이아웃 및 UI ---
st.set_page_config(page_title="AI 정밀 상품 추천", layout="wide")
st.title("🎨 색상 중심 전체 상품 랭킹 분석")

# 사이드바: 상품 DB 관리
with st.sidebar:
    st.header("📦 상품 DB 관리")
    ups = st.file_uploader("상품 이미지를 등록하세요", type=['jpg','png','jpeg'], accept_multiple_files=True)
    
    # 버튼 작동 안정성 강화 섹션
    if st.button("서버에 등록"):
        if ups:
            with st.status("파일 저장 중...", expanded=False) as status:
                for u in ups:
                    file_path = os.path.join(IMAGE_DIR, u.name)
                    with open(file_path, "wb") as f:
                        f.write(u.getbuffer())
                st.cache_data.clear()
                status.update(label="등록 완료!", state="complete", expanded=False)
            st.rerun()
        else:
            st.warning("파일을 먼저 선택해 주세요.")

    if st.button("전체 삭제"):
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
            os.makedirs(IMAGE_DIR)
            st.cache_data.clear()
            st.rerun()

# 메인 화면 처리
all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.info("👈 왼쪽 사이드바에서 비교 대상 상품들을 먼저 등록해 주세요.")
else:
    with st.spinner("상품 데이터를 분석 중입니다..."):
        idx_db, c_db = build_index(all_files)
    
    st.subheader("🔎 검색 이미지 업로드")
    q_file = st.file_uploader("비교할 기준 이미지를 선택하세요", type=['jpg','png','jpeg'])

    if q_file:
        st.divider()
        
        # --- 기준 이미지 표시 섹션 ---
        st.subheader("🎯 비교 기준 이미지")
        q_img = Image.open(q_file).convert("RGB")
        
        m_col1, m_col2, m_col3 = st.columns([1, 1, 1])
        with m_col2:
            st.image(q_img, caption="검색의 기준 이미지", use_container_width=True)
        
        st.divider()

        # 분석 진행
        q_ai = get_features(q_img).astype('float32')
        q_col = get_color_score(q_img)
        
        total_count = len(all_files)
        dists, inds = idx_db.search(q_ai, total_count)
        
        results = []
        for i in range(total_count):
            curr_idx = inds[0][i]
            ai_s = dists[0][i]
            col_s = cv2.compareHist(q_col, c_db[curr_idx], cv2.HISTCMP_CORREL)
            # 요청하신 색상 가중치 70% 반영
            final_s = (max(0, col_s) * 0.7) + (ai_s * 0.3)
            results.append((curr_idx, final_s, ai_s, col_s))
        
        # 총점 순으로 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 결과 리스트 출력
        st.subheader(f"🏆 유사도 순위 (전체 {total_count}개)")
        
        # 4열 그리드 출력
        rows = (total_count // 4) + (1 if total_count % 4 != 0 else 0)
        for r in range(rows):
            cols = st.columns(4)
            for c in range(4):
                idx_res = r * 4 + c
                if idx_res < total_count:
                    res_idx, f_s, a_s, c_s = results[idx_res]
                    with cols[c]:
                        st.image(all_files[res_idx], use_container_width=True)
                        st.markdown(f"**{idx_res + 1}위** (총점: {f_s:.2f})")
                        st.caption(f"🎨 색상: {c_s:.2f} / 👗 디자인: {a_s:.2f}")