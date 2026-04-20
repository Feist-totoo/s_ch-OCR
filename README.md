# Vision OCR · s_ch-OCR

> 针对**超长图**特化的极简字符提取工具。

---

## ✨ 特性

- 🖼 **智能无损切片** — 分析行方差自动寻找最优断点，支持最高约 30,000 px 的长截图
- 🔍 **图像预处理** — 提供标准增强 / 极限二值化两档对比度优化，有效对抗低质量扫描件
- 🎯 **置信度过滤** — 严格 / 标准 / 宽松三档阈值，按需平衡召回与准确率
- 🔄 **自动方向校正** — 可选方向分类器，兼容拍照倾斜场景
- 📤 **一键导出** — 结果以 `.txt` 直接下载

---

## 🛠 技术栈

| 层次 | 技术 |
|------|------|
| UI 框架 | [Streamlit](https://streamlit.io) |
| OCR 引擎 | [RapidOCR](https://github.com/RapidAI/RapidOCR) |
| 图像处理 | Pillow · NumPy |
| 运行环境 | Python 3.10+ |

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Feist-totoo/s_ch-OCR.git
cd s_ch-OCR

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动
streamlit run app.py
```

---

## 📖 使用说明

1. 上传 PNG / JPG / JPEG / WebP 图片
2. 按需调整**图像预处理**、**置信度阈值**、**自动方向校正**
3. 点击「开始提取文本」
4. 查看结果，可直接导出 `.txt`

---

## 📁 项目结构

```
s_ch-OCR/
├── app.py          # 主程序
└── requirements.txt
```