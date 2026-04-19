import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import base64
import requests
import io
import time

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────
# API 配置
# ─────────────────────────────────────────────
API_URL = "https://z8t5weoff78et3d2.aistudio-app.com/layout-parsing"
TOKEN   = "d46b18fed8ee6704eacbf91c022cc549e0c7515c"

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Content-Type":  "application/json",
}


# ─────────────────────────────────────────────
# 图像预处理（保持不变）
# ─────────────────────────────────────────────
def preprocess_image(img, mode="extreme"):
    if mode == "off":
        return img
    img = img.convert("L")
    if mode == "standard":
        return ImageOps.autocontrast(img, cutoff=0)
    if mode == "extreme":
        img = ImageOps.autocontrast(img, cutoff=0)
        arr = np.array(img)
        arr = np.where(arr < 128, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img


# ─────────────────────────────────────────────
# 智能长图切片（保持不变）
# ─────────────────────────────────────────────
def smart_slice_image(img, target_height=2500, search_window=300):
    width, height = img.size
    if height <= target_height + search_window:
        return [img]

    img_gray  = img.convert('L')
    img_array = np.array(img_gray)
    chunks, current_y = [], 0

    while current_y < height:
        if current_y + target_height >= height:
            chunks.append(img.crop((0, current_y, width, height)))
            break
        search_start     = current_y + target_height - search_window
        search_end       = current_y + target_height
        window           = img_array[search_start:search_end, :]
        best_cut_relative = np.argmin(np.var(window, axis=1))
        cut_y            = search_start + best_cut_relative
        chunks.append(img.crop((0, current_y, width, cut_y)))
        current_y = cut_y

    return chunks


# ─────────────────────────────────────────────
# 核心：调用 PaddleOCR-VL-1.5 API
# ─────────────────────────────────────────────
def ocr_chunk_via_api(chunk_img: Image.Image) -> str:
    """
    将单块 PIL Image 编码为 base64 PNG，POST 到 VL-1.5 接口。
    返回该块识别出的 Markdown 文本（多栏文档结构完整保留）。
    """
    # PIL → bytes → base64
    buf = io.BytesIO()
    # 若图像为灰度/二值，先转 RGB 确保 JPEG 编码兼容
    chunk_img.convert("RGB").save(buf, format="PNG")
    file_data = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "file":     file_data,
        "fileType": 1,          # 1 = 图片
        # 关闭耗时可选项，最大化速度
        "useDocOrientationClassify": False,
        "useDocUnwarping":           False,
        "useChartRecognition":       False,
    }

    resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"API 返回错误 {resp.status_code}: {resp.text[:200]}")

    results = resp.json().get("result", {}).get("layoutParsingResults", [])
    # 一张图片对应一个解析结果，取第一条的 markdown 文本
    if not results:
        return ""
    return results[0]["markdown"]["text"].strip()


# ─────────────────────────────────────────────
# UI 配置
# ─────────────────────────────────────────────
st.set_page_config(page_title="Vision OCR | 长图提取", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 3rem; padding-bottom: 2rem; max-width: 800px; }
    .main-title { font-size: 2.2rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem; letter-spacing: -0.02em; }
    .sub-title { font-size: 1rem; color: #6B7280; margin-bottom: 2.5rem; font-weight: 400; }
    .stSelectbox > div > div > div { border-radius: 8px; border: 1px solid #E5E7EB; box-shadow: none !important; }
    .stButton > button[kind="primary"] { border-radius: 8px; background-color: #111827; color: #FFFFFF; border: none; font-weight: 500; transition: all 0.2s ease; }
    .stButton > button[kind="primary"]:hover { background-color: #374151; }
    .stButton > button[kind="secondary"] { border-radius: 8px; border: 1px solid #E5E7EB; color: #374151; background-color: transparent; }
    .stDownloadButton > button { border-radius: 8px; border: 1px solid #E5E7EB; color: #111827; width: 100%; background-color: #FFFFFF; }
    .result-box { border: 1px solid #E5E7EB; border-radius: 8px; background: #FAFAFA; padding: 1.5rem 1.75rem; line-height: 1.8; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Vision OCR.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">PaddleOCR-VL-1.5 · 版面理解 · Markdown 结构化输出</div>', unsafe_allow_html=True)

CONTRAST_MAP = {
    "不处理 (截图推荐)": "off",
    "标准增强":          "standard",
    "最高对比度":        "extreme",
}

# ─────────────────────────────────────────────
# 上传与设置
# ─────────────────────────────────────────────
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "选择图片文件", type=["png", "jpg", "jpeg", "webp"],
        label_visibility="hidden"
    )
    col_contrast, col_slice = st.columns(2)
    with col_contrast:
        contrast_choice = st.selectbox("图像预处理", list(CONTRAST_MAP.keys()))
    with col_slice:
        slice_height = st.slider(
            "切片高度 (px)", min_value=1000, max_value=4000, value=2500, step=500,
            help="超长图会按此高度切片后分批发送，建议保持默认值"
        )

st.write("")

# ─────────────────────────────────────────────
# 执行识别
# ─────────────────────────────────────────────
if uploaded_file:
    if st.button("开始提取文本", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0)
            status_text  = st.empty()

            img = Image.open(uploaded_file)

            # 预处理
            contrast_mode = CONTRAST_MAP[contrast_choice]
            if contrast_mode != "off":
                status_text.caption("正在增强图像对比度...")
                img_processed = preprocess_image(img, mode=contrast_mode)
            else:
                img_processed = img

            # 切片
            status_text.caption(f"分析图片尺寸: {img.width} × {img.height} px")
            chunks       = smart_slice_image(img_processed, target_height=slice_height, search_window=400)
            total_chunks = len(chunks)

            full_md_parts = []

            for i, chunk in enumerate(chunks):
                status_text.caption(
                    f"调用 VL-1.5 识别区块 {i+1} / {total_chunks}..."
                    + (" （首块含模型冷启动，稍等）" if i == 0 else "")
                )
                md_text = ocr_chunk_via_api(chunk)
                if md_text:
                    full_md_parts.append(md_text)
                progress_bar.progress((i + 1) / total_chunks)

            status_text.empty()
            progress_bar.empty()

            final_md = "\n\n---\n\n".join(full_md_parts)  # 切片间加分割线

            if not final_md.strip():
                st.error("未能识别到有效文本，请检查图片质量或网络连接。")
            else:
                st.session_state['ocr_result'] = final_md
                st.rerun()

        except requests.exceptions.Timeout:
            st.error("API 请求超时（>120s），图片可能过大，请尝试减小切片高度。")
        except Exception as e:
            st.error(f"处理异常: {e}")

# ─────────────────────────────────────────────
# 结果展示（Markdown 渲染 + 纯文本导出）
# ─────────────────────────────────────────────
if 'ocr_result' in st.session_state:
    st.write("")
    st.markdown("### 提取结果")

    # Tab 1：渲染视图  Tab 2：原始 Markdown
    tab_render, tab_raw = st.tabs(["渲染视图", "原始 Markdown"])

    with tab_render:
        st.markdown(
            f'<div class="result-box">{st.session_state["ocr_result"]}</div>',
            unsafe_allow_html=True
        )
    with tab_raw:
        st.code(st.session_state['ocr_result'], language="markdown")

    st.write("")
    col_dl_md, col_dl_txt, col_clear = st.columns([1, 1, 1])
    with col_dl_md:
        st.download_button(
            "导出 Markdown", st.session_state['ocr_result'],
            "ocr_result.md", mime="text/markdown"
        )
    with col_dl_txt:
        # 导出纯文本：去掉 Markdown 标记符
        plain = "\n".join(
            line.lstrip("#").strip()
            for line in st.session_state['ocr_result'].splitlines()
        )
        st.download_button("导出纯文本 (TXT)", plain, "ocr_result.txt")
    with col_clear:
        if st.button("清除结果", type="secondary", use_container_width=True):
            del st.session_state['ocr_result']
            st.rerun()

    st.write("")
    with st.expander("查看原始图片", expanded=False):
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)