import streamlit as st
import pytesseract
from PIL import Image
import io

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(page_title="精准长图 OCR", page_icon="🔍", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        width: 100%; 
        background-color: #007bff; 
        color: white; 
        border-radius: 8px;
        height: 3em;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 精准中英 OCR (长图强化版)")

# ─────────────────────────────────────────────
# 第一步：上传与核心设置
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("📥 第一步：上传图片（支持长图）", type=["png", "jpg", "jpeg", "webp"])

# 将选项并排显示，节省空间
col1, col2 = st.columns(2)
with col1:
    lang_choice = st.selectbox("🔤 识别语言", ["中英混合", "纯中文", "纯英文"])
with col2:
    mode_choice = st.selectbox("📏 识别模式", ["标准模式", "长图/精准换行模式"], 
                             help="长图模式会强制保留识别到的物理换行")

# 映射参数
LANG_MAP = {"中英混合": "chi_sim+eng", "纯中文": "chi_sim", "纯英文": "eng"}
# PSM 6 对长图中的段落块和换行识别效果通常更好
PSM_MAP = {"标准模式": "--psm 3", "长图/精准换行模式": "--psm 6"}

# ─────────────────────────────────────────────
# 第二步：操作按钮（位置上移）
# ─────────────────────────────────────────────
# 只有上传了文件才显示转换按钮
if uploaded_file:
    if st.button("🚀 开始转换"):
        with st.spinner("正在精准解析中..."):
            try:
                img = Image.open(uploaded_file)
                
                # 配置 Tesseract: --oem 3 使用最新神经网络引擎
                config = f"--oem 3 {PSM_MAP[mode_choice]}"
                
                # 执行识别
                text = pytesseract.image_to_string(
                    img, 
                    lang=LANG_MAP[lang_choice], 
                    config=config
                )
                
                # 后处理：清理空行但保留必要换行
                clean_text = "\n".join([line.rstrip() for line in text.splitlines() if line.strip()])
                
                if not clean_text:
                    st.warning("未能识别到文字，请检查图片清晰度。")
                else:
                    st.session_state['ocr_result'] = clean_text
                    st.success("识别完成！结果已显示在下方。")
                    
            except Exception as e:
                st.error(f"发生错误: {e}")

# ─────────────────────────────────────────────
# 第三步：结果展示
# ─────────────────────────────────────────────
if 'ocr_result' in st.session_state:
    st.divider()
    st.subheader("📄 识别结果")
    # 使用 text_area 方便复制，且能展示长文本
    st.text_area("识别文本", value=st.session_state['ocr_result'], height=400)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 下载 TXT", st.session_state['ocr_result'], "result.txt")
    with col_dl2:
        if st.button("🗑 清除结果"):
            del st.session_state['ocr_result']
            st.rerun()

# 最后显示图片预览（放在底部，避免遮挡操作区）
if uploaded_file:
    with st.expander("👁 查看原始图片预览"):
        st.image(uploaded_file, use_container_width=True)