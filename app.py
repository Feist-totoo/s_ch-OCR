import streamlit as st
import pytesseract
from PIL import Image
import io
import numpy as np
import time

# 解除 PIL 对超大尺寸图片的限制
Image.MAX_IMAGE_PIXELS = None

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
    /* 隐藏默认头部和尾部，保持页面纯净 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局背景与排版微调 */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* 标题样式定制 */
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

    /* 输入框与下拉菜单极简圆角化 */
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        box-shadow: none !important;
    }
    
    /* 普通按钮 (次要操作) */
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

    /* 主操作按钮 (深色极简风) */
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
    
    /* 下载按钮统一为极简风格 */
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

    /* 文本域去边框感 */
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

# ─────────────────────────────────────────────
# 工作区：上传与设置
# ─────────────────────────────────────────────
# 使用 border=True 的 container 创建一个卡片式的视觉区域
with st.container(border=True):
    uploaded_file = st.file_uploader("选择图片文件", type=["png", "jpg", "jpeg", "webp"], label_visibility="hidden")
    
    # 选项区
    col_lang, col_mode = st.columns(2)
    with col_lang:
        lang_choice = st.selectbox("识别语言", ["中英混合", "纯中文", "纯英文"])
    with col_mode:
        mode_choice = st.selectbox("识别模式", ["精准排版模式 (推荐)", "标准模式"])

# ─────────────────────────────────────────────
# 执行操作
# ─────────────────────────────────────────────
# 增加一点留白
st.write("") 

if uploaded_file:
    # 使用 type="primary" 和 use_container_width=True 打造宽屏主按钮
    if st.button("开始提取文本", type="primary", use_container_width=True):
        try:
            img = Image.open(uploaded_file)
            
            # 进度反馈区，保持文字极简
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.caption(f"分析图片尺寸: {img.width} × {img.height} px")
            chunks = smart_slice_image(img, target_height=2500, search_window=400)
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
            
            # 结束后清空进度条，保持页面整洁
            status_text.empty()
            progress_bar.empty()
            
            final_result = "\n\n".join(full_text)
            
            if not final_result.strip():
                st.error("未能识别到有效文本，请检查图片质量。")
            else:
                st.session_state['ocr_result'] = final_result
                st.rerun() # 重新运行以平滑显示结果
                
        except Exception as e:
            st.error(f"处理异常: {e}")

# ─────────────────────────────────────────────
# 结果展示区
# ─────────────────────────────────────────────
if 'ocr_result' in st.session_state:
    st.write("") # 留白
    st.markdown("### 提取结果")
    
    st.text_area("Result", value=st.session_state['ocr_result'], height=400, label_visibility="hidden")
    
    # 底部操作栏
    col_dl, col_clear = st.columns([1, 1])
    with col_dl:
        st.download_button("导出文本 (TXT)", st.session_state['ocr_result'], "vision_ocr_result.txt")
    with col_clear:
        # 使用次要按钮样式
        if st.button("清除结果", type="secondary", use_container_width=True):
            del st.session_state['ocr_result']
            st.rerun()

    # 将图片预览折叠，保持主界面清爽
    st.write("")
    with st.expander("查看原始图片", expanded=False):
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
