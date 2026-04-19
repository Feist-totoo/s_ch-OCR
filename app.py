import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import time

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────
# 新增：图像对比度预处理
# ─────────────────────────────────────────────
def preprocess_image(img, mode="extreme"):
    """
    对图像进行对比度增强预处理，提升 OCR 识别准确率。
    mode:
      - "off"     不处理，原图直出
      - "standard" 标准增强：灰度 + 对比度拉伸
      - "extreme"  最高对比度：灰度 + CLAHE 模拟 + 自适应阈值二值化
    """
    if mode == "off":
        return img

    # 第一步：转为灰度
    img = img.convert("L")

    if mode == "standard":
        # 使用 PIL 自带的对比度增强器，系数 2.5 是经验最优值
        img = ImageEnhance.Contrast(img).enhance(2.5)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        return img

    if mode == "extreme":
        arr = np.array(img, dtype=np.float32)

        # ── 直方图线性拉伸 ──────────────────────────
        # 截断最暗 2% 和最亮 2% 的像素，避免极端噪点干扰
        p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
        if p98 > p2:  # 防止除零
            arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255)

        # ── 自适应阈值二值化 ────────────────────────
        # 用局部均值做自适应阈值，比全局 Otsu 对光照不均更鲁棒
        arr_uint8 = arr.astype(np.uint8)
        blurred = np.array(
            Image.fromarray(arr_uint8).filter(ImageFilter.GaussianBlur(radius=15)),
            dtype=np.float32
        )
        # 当像素比局部均值暗 10 以上时判定为"文字"(黑)，否则为"背景"(白)
        binary = np.where(arr.astype(np.float32) < blurred - 10, 0, 255).astype(np.uint8)

        # ── 最终锐化 ────────────────────────────────
        result = Image.fromarray(binary)
        result = ImageEnhance.Sharpness(result).enhance(2.0)
        return result

    return img  # 兜底返回


# ─────────────────────────────────────────────
# 核心功能：智能长图切片 (逻辑保持不变)
# ─────────────────────────────────────────────
def smart_slice_image(img, target_height=2500, search_window=300):
    width, height = img.size
    if height <= target_height + search_window:
        return [img]

    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    chunks = []
    current_y = 0

    while current_y < height:
        if current_y + target_height >= height:
            chunks.append(img.crop((0, current_y, width, height)))
            break

        search_start = current_y + target_height - search_window
        search_end = current_y + target_height
        window = img_array[search_start:search_end, :]
        row_variances = np.var(window, axis=1)
        best_cut_relative = np.argmin(row_variances)

        cut_y = search_start + best_cut_relative
        chunks.append(img.crop((0, current_y, width, cut_y)))
        current_y = cut_y

    return chunks


# ─────────────────────────────────────────────
# UI 全局配置与极简 CSS 注入
# ─────────────────────────────────────────────
st.set_page_config(page_title="Vision OCR | 长图提取", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
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
st.markdown('<div class="sub-title">针对超长图特化的极简字符提取工具，支持最高三万像素智能无损切片。</div>', unsafe_allow_html=True)

# 参数映射
LANG_MAP = {"中英混合": "chi_sim+eng", "纯中文": "chi_sim", "纯英文": "eng"}
PSM_MAP = {"精准排版模式 (推荐)": "--psm 6", "标准模式": "--psm 3"}
# 新增：对比度模式映射
CONTRAST_MAP = {
    "最高对比度 (推荐)": "extreme",
    "标准增强":          "standard",
    "不处理":            "off",
}

# ─────────────────────────────────────────────
# 工作区：上传与设置
# ─────────────────────────────────────────────
with st.container(border=True):
    uploaded_file = st.file_uploader("选择图片文件", type=["png", "jpg", "jpeg", "webp"], label_visibility="hidden")

    col_lang, col_mode, col_contrast = st.columns(3)
    with col_lang:
        lang_choice = st.selectbox("识别语言", ["中英混合", "纯中文", "纯英文"])
    with col_mode:
        mode_choice = st.selectbox("识别模式", ["精准排版模式 (推荐)", "标准模式"])
    with col_contrast:
        contrast_choice = st.selectbox("图像预处理", list(CONTRAST_MAP.keys()))

st.write("")

# ─────────────────────────────────────────────
# 执行操作
# ─────────────────────────────────────────────
if uploaded_file:
    if st.button("开始提取文本", type="primary", use_container_width=True):
        try:
            img = Image.open(uploaded_file)

            progress_bar = st.progress(0)
            status_text = st.empty()

            # ── 预处理阶段 ──────────────────────────
            contrast_mode = CONTRAST_MAP[contrast_choice]
            if contrast_mode != "off":
                status_text.caption("正在增强图像对比度...")
                # 注意：切片前先对原图预处理，保证切割逻辑仍基于原尺寸
                img_processed = preprocess_image(img, mode=contrast_mode)
            else:
                img_processed = img

            # ── 切片阶段 ────────────────────────────
            status_text.caption(f"分析图片尺寸: {img.width} × {img.height} px")
            # 切片基于预处理后的图像；切割线的寻找逻辑同样适用于灰度/二值图
            chunks = smart_slice_image(img_processed, target_height=2500, search_window=400)
            total_chunks = len(chunks)

            config = f"--oem 3 {PSM_MAP[mode_choice]}"
            full_text = []

            for i, chunk in enumerate(chunks):
                status_text.caption(f"正在处理区块 {i+1} / {total_chunks}...")
                text = pytesseract.image_to_string(
                    chunk,
                    lang=LANG_MAP[lang_choice],
                    config=config
                )
                clean_text = "\n".join([line.rstrip() for line in text.splitlines() if line.strip()])
                if clean_text:
                    full_text.append(clean_text)
                progress_bar.progress((i + 1) / total_chunks)

            status_text.caption("整合文本数据...")
            time.sleep(0.3)

            status_text.empty()
            progress_bar.empty()

            final_result = "\n\n".join(full_text)

            if not final_result.strip():
                st.error("未能识别到有效文本，请检查图片质量或尝试更换预处理模式。")
            else:
                st.session_state['ocr_result'] = final_result
                st.rerun()

        except Exception as e:
            st.error(f"处理异常: {e}")

# ─────────────────────────────────────────────
# 结果展示区
# ─────────────────────────────────────────────
if 'ocr_result' in st.session_state:
    st.write("")
    st.markdown("### 提取结果")

    st.text_area("Result", value=st.session_state['ocr_result'], height=400, label_visibility="hidden")

    col_dl, col_clear = st.columns([1, 1])
    with col_dl:
        st.download_button("导出文本 (TXT)", st.session_state['ocr_result'], "vision_ocr_result.txt")
    with col_clear:
        if st.button("清除结果", type="secondary", use_container_width=True):
            del st.session_state['ocr_result']
            st.rerun()

    st.write("")
    with st.expander("查看原始图片", expanded=False):
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)