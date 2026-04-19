import streamlit as st
from rapidocr import RapidOCR
from PIL import Image, ImageOps
import numpy as np
import time

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────
# RapidOCR 实例缓存
# ─────────────────────────────────────────────
@st.cache_resource
def load_ocr_engine():
    return RapidOCR()


# ─────────────────────────────────────────────
# 图像对比度预处理
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
# 智能长图切片
# ─────────────────────────────────────────────
def smart_slice_image(img, target_height=2500, search_window=300):
    width, height = img.size
    if height <= target_height + search_window:
        return [img]

    img_gray = img.convert("L")
    img_array = np.array(img_gray)
    chunks = []
    current_y = 0

    while current_y < height:
        if current_y + target_height >= height:
            chunks.append(img.crop((0, current_y, width, height)))
            break

        search_start = current_y + target_height - search_window
        search_end   = current_y + target_height
        window = img_array[search_start:search_end, :]
        row_variances = np.var(window, axis=1)
        best_cut_relative = np.argmin(row_variances)

        cut_y = search_start + best_cut_relative
        chunks.append(img.crop((0, current_y, width, cut_y)))
        current_y = cut_y

    return chunks


# ─────────────────────────────────────────────
# RapidOCR 结果 → 纯文本（按 Y 轴排序）
# ─────────────────────────────────────────────
def result_to_text(result, conf_threshold=0.5):
    if not result or not result.boxes:
        return ""

    lines = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        if score >= conf_threshold:
            top_y = box[0][1]
            lines.append((top_y, text))

    lines.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in lines)


# ─────────────────────────────────────────────
# UI 配置与 CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Vision OCR | 长图提取", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        box-shadow: none !important;
    }
    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        color: #374151;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #9CA3AF;
        color: #111827;
        background-color: #F9FAFB;
    }
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        background-color: #111827;
        color: #FFFFFF;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #374151;
        box-shadow: 0 4px 12px rgba(0,0,0,.1);
    }
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        color: #111827;
        width: 100%;
        background-color: #FFFFFF;
    }
    .stDownloadButton > button:hover {
        border-color: #9CA3AF;
        background-color: #F9FAFB;
    }
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 页面 Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Vision OCR.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">针对超长图特化的极简字符提取工具，由 RapidOCR 驱动，支持最高三万像素智能无损切片。</div>', unsafe_allow_html=True)

# 参数映射
CONTRAST_MAP = {
    "最高对比度 (推荐)": "extreme",
    "标准增强":           "standard",
    "不处理":             "off",
}
CONF_MAP = {
    "严格 (≥ 0.8)": 0.8,
    "标准 (≥ 0.5)": 0.5,
    "宽松 (≥ 0.3)": 0.3,
}

# ─────────────────────────────────────────────
# 工作区：上传与设置
# ─────────────────────────────────────────────
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "选择图片文件",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="hidden",
    )

    col_contrast, col_conf, col_cls = st.columns(3)
    with col_contrast:
        contrast_choice = st.selectbox("图像预处理", list(CONTRAST_MAP.keys()))
    with col_conf:
        conf_choice = st.selectbox("置信度阈值", list(CONF_MAP.keys()), index=1)
    with col_cls:
        angle_cls = st.toggle("自动纠正方向", value=False, help="适用于拍照倾斜或旋转的图片")

st.write("")

# ─────────────────────────────────────────────
# 执行操作
# ─────────────────────────────────────────────
if uploaded_file:
    if st.button("开始提取文本", type="primary", use_container_width=True):
        try:
            img = Image.open(uploaded_file).convert("RGB")

            progress_bar = st.progress(0)
            status_text  = st.empty()

            # ── 加载引擎 ────────────────────────────
            status_text.caption("加载 RapidOCR 模型（首次运行需下载，请稍候）...")
            ocr = load_ocr_engine()

            # ── 预处理 ──────────────────────────────
            contrast_mode = CONTRAST_MAP[contrast_choice]
            if contrast_mode != "off":
                status_text.caption("正在增强图像对比度...")
                img_processed = preprocess_image(img, mode=contrast_mode)
                img_processed = img_processed.convert("RGB")
            else:
                img_processed = img

            # ── 切片 ────────────────────────────────
            status_text.caption(f"分析图片尺寸：{img.width} × {img.height} px")
            chunks       = smart_slice_image(img_processed, target_height=2500, search_window=400)
            total_chunks = len(chunks)

            conf_threshold = CONF_MAP[conf_choice]
            full_text = []

            for i, chunk in enumerate(chunks):
                status_text.caption(f"正在识别区块 {i+1} / {total_chunks}...")
                chunk_array = np.array(chunk)
                result = ocr(
                    chunk_array,
                    use_cls=angle_cls,
                    use_det=True,
                    use_rec=True,
                )
                text = result_to_text(result, conf_threshold=conf_threshold)
                if text.strip():
                    full_text.append(text.strip())
                progress_bar.progress((i + 1) / total_chunks)

            status_text.caption("整合文本数据...")
            time.sleep(0.3)
            status_text.empty()
            progress_bar.empty()

            final_result = "\n\n".join(full_text)

            if not final_result.strip():
                st.error("未能识别到有效文本，请检查图片质量或尝试更换预处理 / 置信度设置。")
            else:
                st.session_state["ocr_result"] = final_result
                st.rerun()

        except Exception as e:
            st.error(f"处理异常：{e}")

# ─────────────────────────────────────────────
# 结果展示区
# ─────────────────────────────────────────────
if "ocr_result" in st.session_state:
    st.write("")
    st.markdown("### 提取结果")

    st.text_area(
        "Result",
        value=st.session_state["ocr_result"],
        height=400,
        label_visibility="hidden",
    )

    col_dl, col_clear = st.columns([1, 1])
    with col_dl:
        st.download_button(
            "导出文本 (TXT)",
            st.session_state["ocr_result"],
            "vision_ocr_result.txt",
        )
    with col_clear:
        if st.button("清除结果", type="secondary", use_container_width=True):
            del st.session_state["ocr_result"]
            st.rerun()

    st.write("")
    with st.expander("查看原始图片", expanded=False):
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)