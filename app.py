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
    if result is None:
        return ""

    boxes  = result.boxes
    txts   = result.txts
    scores = result.scores

    if boxes is None or txts is None or scores is None:
        return ""

    lines = []
    for box, text, score in zip(boxes, txts, scores):
        if score is not None and score >= conf_threshold:
            top_y = float(box[0][1])
            lines.append((top_y, text))

    lines.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in lines)

# ─────────────────────────────────────────────
# UI 配置与自适应 CSS
# ─────────────────────────────────────────────
# 设置 layout="wide" 充分利用屏幕，便于左侧边栏和主界面的配合
st.set_page_config(page_title="Vision OCR | 批量长图提取", layout="wide", initial_sidebar_state="expanded")

# 移除硬编码颜色，依赖 Streamlit 原生变量适应系统的深浅色模式
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 2rem;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 初始化 Session State
# ─────────────────────────────────────────────
if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = {}

# ─────────────────────────────────────────────
# 页面 Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Vision OCR.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">针对超长图特化的极简批量字符提取工具，支持最高三万像素智能无损切片，自适应深色/浅色模式。</div>', unsafe_allow_html=True)

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
# 工作区：设置与上传
# ─────────────────────────────────────────────
with st.container(border=True):
    col_upload, col_settings = st.columns([1, 1])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "上传图片 (支持多选批量提取)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="visible"
        )
        
    with col_settings:
        contrast_choice = st.selectbox("图像预处理", list(CONTRAST_MAP.keys()))
        conf_choice = st.selectbox("置信度阈值", list(CONF_MAP.keys()), index=1)
        angle_cls = st.toggle("自动纠正方向", value=False, help="适用于拍照倾斜或旋转的图片")

# ─────────────────────────────────────────────
# 左侧边栏：原图预览
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🖼️ 原图预览")
    st.caption("在此预览您上传的待处理图片，支持侧边栏隐藏收起。")
    if uploaded_files:
        for f in uploaded_files:
            with st.expander(f"预览: {f.name}", expanded=False):
                st.image(f, use_container_width=True)
    else:
        st.info("尚未上传任何图片。")

# ─────────────────────────────────────────────
# 核心执行操作 (批量处理)
# ─────────────────────────────────────────────
if uploaded_files:
    if st.button("🚀 开始批量提取文本", type="primary", use_container_width=True):
        st.session_state["ocr_results"].clear()
        
        # 整体进度与状态
        overall_progress = st.progress(0)
        status_box = st.empty()
        
        try:
            status_box.info("加载 RapidOCR 模型中...")
            ocr = load_ocr_engine()
            
            total_files = len(uploaded_files)
            contrast_mode = CONTRAST_MAP[contrast_choice]
            conf_threshold = CONF_MAP[conf_choice]

            for file_idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_box.info(f"正在处理图片 ({file_idx+1}/{total_files}): {filename} ...")
                
                img = Image.open(uploaded_file).convert("RGB")
                
                # 预处理
                if contrast_mode != "off":
                    img_processed = preprocess_image(img, mode=contrast_mode).convert("RGB")
                else:
                    img_processed = img

                # 切片
                chunks = smart_slice_image(img_processed, target_height=2500, search_window=400)
                total_chunks = len(chunks)
                full_text = []

                for i, chunk in enumerate(chunks):
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
                
                # 保存单张图结果
                final_result = "\n\n".join(full_text)
                if final_result.strip():
                    st.session_state["ocr_results"][filename] = final_result
                else:
                    st.session_state["ocr_results"][filename] = "[未识别到有效文本]"

                overall_progress.progress((file_idx + 1) / total_files)

            status_box.success(f"✅ 成功完成 {total_files} 张图片的文本提取！")
            time.sleep(1)
            status_box.empty()
            overall_progress.empty()

        except Exception as e:
            st.error(f"处理异常：{e}")

# ─────────────────────────────────────────────
# 结果展示与批量导出区 (主界面展示，视作右侧结果面板)
# ─────────────────────────────────────────────
if st.session_state.get("ocr_results"):
    st.divider()
    
    col_title, col_export = st.columns([1, 1], vertical_alignment="bottom")
    with col_title:
        st.markdown("### 📑 提取结果预览")
    with col_export:
        # 生成批量导出文本
        batch_export_text = ""
        for name, text in st.session_state["ocr_results"].items():
            batch_export_text += f"====== {name} ======\n{text}\n\n"
            
        st.download_button(
            label="📥 批量导出全部文本 (TXT)",
            data=batch_export_text,
            file_name="vision_ocr_batch_result.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    # 结果展示面板（使用 expander 保持界面简洁，可单独复制每个文件）
    for name, text in st.session_state["ocr_results"].items():
        with st.expander(f"📄 {name}", expanded=True):
            st.text_area(
                "Result",
                value=text,
                height=250,
                key=f"text_{name}",
                label_visibility="hidden",
            )
            
    if st.button("🗑️ 清除所有结果", type="secondary"):
        st.session_state["ocr_results"].clear()
        st.rerun()
