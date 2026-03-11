import streamlit as st
import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import cv2
import shutil

# --- 1. 모델 로드 ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 서버 안정성을 위해 B/32 사용 + 알고리즘 보완
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# --- 2. 분석 로직 (중앙 집중형) ---
def get_features(image):
    w, h = image.size
    image = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def get_color_score(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    center = img_cv[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    hist = cv2.calcHist([center], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. 폴더 및 인덱스 설정 ---
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

# --- 4. UI 레이아웃 ---
st.set_page_config(layout="wide")
st.title("📊 등록 상품 전체 유사도 랭킹")

with st.sidebar:
    st.header("📦 DB 관리")
    ups = st.file_uploader("상품 등록", type=['jpg','png','jpeg'], accept_multiple_files=True)
    if st.button("서버에 등록"):
        for u in ups:
            with open(os.path.join(IMAGE_DIR, u.name), "wb") as f: f.write(u.getbuffer())
        st.cache_data.clear(); st.rerun()
    if st.button("전체 삭제"):
        shutil.rmtree(IMAGE_DIR); os.makedirs(IMAGE_DIR); st.cache_data.clear(); st.rerun()

all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not all_files:
    st.info("👈 왼쪽 사이드바에서 상품들을 먼저 등록해 주세요.")
else:
    idx, c_db = build_index(all_files)
    st.write(f"현재 총 **{len(all_files)}개**의 상품이 분석 대상입니다.")
    
    st.divider()
    
    st.subheader("🔎 검색 이미지 업로드")
    q_file = st.file_uploader("비교할 이미지를 선택하세요", type=['jpg','png','jpeg'])

    if q_file:
        q_img = Image.open(q_file).convert("RGB")
        q_ai = get_features(q_img).astype('float32')
        q_col = get_color_score(q_img)
        
        # 1. 모든 상품에 대해 거리 계산 (k를 전체 파일 수로 설정)
        total_count = len(all_files)
        dists, inds = idx.search(q_ai, total_count)
        
        results = []
        for i in range(total_count):
            curr_idx = inds[0][i]
            ai_s = dists[0][i]
            col_s = cv2.compareHist(q_col, c_db[curr_idx], cv2.HISTCMP_CORREL)
            # 종합 점수 계산 (디자인 85% + 색상 15%)
            final_s = (ai_s * 0.85) + (max(0, col_s) * 0.15)
            results.append((curr_idx, final_s, ai_s, col_s))
        
        # 최종 점수순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 2. 결과 출력 (그리드 형태)
        st.subheader(f"🏆 전체 유사도 순위 (1위 ~ {total_count}위)")
        
        # 한 줄에 4개씩 출력
        rows = (total_count // 4) + (1 if total_count % 4 != 0 else 0)
        for r in range(rows):
            cols = st.columns(4)
            for c in range(4):
                idx_in_res = r * 4 + c
                if idx_in_res < total_count:
                    res_idx, f_s, a_s, c_s = results[idx_in_res]
                    with cols[c]:
                        st.image(all_files[res_idx], use_container_width=True)
                        st.markdown(f"**{idx_in_res + 1}위** (점수: {f_s:.2f})")
                        with st.expander("상세 데이터"):
                            st.write(f"디자인: {a_s:.2f}")
                            st.write(f"색상: {c_s:.2f}")