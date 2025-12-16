import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io

# =========================
# 0. 页面配置与安全验证
# =========================
st.set_page_config(
    page_title="AI 评论精细化分析系统 (NLP Engineer Ver.)",
    page_icon="🔬",
    layout="wide"
)

# 解决 Matplotlib 中文乱码问题 (尝试使用系统通用字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

ACCESS_PASSWORD = "admin123" 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    if st.session_state["password_input"] == ACCESS_PASSWORD:
        st.session_state.logged_in = True
    else:
        st.error("密码错误")

if not st.session_state.logged_in:
    st.markdown("## 🔒 系统锁定")
    st.text_input("请输入访问密码", type="password", key="password_input", on_change=check_password)
    st.stop() 

# =========================
# 1. 标签库定义 (严格遵守)
# =========================

# 好评标签库 (固定集合)
POS_LABELS_LIST = [
    "面料舒适", "质量很好", "有助于锻炼", "有助于缓解疼痛", "保暖", "舒适贴合", 
    "有压缩感", "抓握式有效", "合身", "有助于关节炎/扳机指", "增加手指灵活", 
    "促进血液循环", "耐用", "缓解不适", "轻盈", "覆盖整个手指", "有助于防止受伤"
]

# 差评标签库 (沿用旧版逻辑，补充完整以覆盖常见差评)
NEG_LABELS_LIST = [
    "无效/没有作用", "缝线开裂/破损", "收到二手/脏污", "面料质量差/廉价", 
    "尺码太小/太紧", "尺码太大/太松", "接缝处磨手/不适", "不耐用/一次性", 
    "过敏/皮疹/发痒", "太滑/没有抓握力", "数量不符/发错货", "导致血液循环受阻"
]

# =========================
# 2. AI 模型加载
# =========================
@st.cache_resource
def load_model():
    # 使用多语言模型处理中英文语义
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# =========================
# 3. 核心 NLP 引擎：拆句与匹配
# =========================

def split_into_sentences(text):
    """
    语义拆解：将长评论拆分为独立句子/语义单元。
    支持中英文标点及换行符。
    """
    if not isinstance(text, str):
        return []
    # 使用正则按 . ! ? ; 。 ！？ ；以及换行符进行切分
    sentences = re.split(r'[.!?;。！？；\n]+', text)
    # 过滤空字符串并去除首尾空格
    return [s.strip() for s in sentences if s.strip()]

def analyze_single_review(row_idx, rating, full_text, model, threshold=0.40):
    """
    对单条评论进行细粒度分析，返回多个结构化结果。
    """
    sentences = split_into_sentences(full_text)
    analyzed_results = []
    
    # 预编码标签库 (Tensor)
    pos_embeddings = model.encode(POS_LABELS_LIST, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS_LIST, convert_to_tensor=True)

    # 评论整体情感基调 (简单规则：<=3星为负向，>=4星为正向)
    review_polarity_base = "negative" if rating <= 3 else "positive"

    if not sentences:
        # 如果评论为空或无法拆分，直接返回整句的兜底
        return [{
            "review_id": row_idx,
            "original_review": full_text,
            "sentence": str(full_text),
            "polarity": review_polarity_base,
            "label": "差评其他" if review_polarity_base == "negative" else "好评其他",
            "evidence": str(full_text),
            "confidence": 0.5
        }]

    for sent in sentences:
        # 忽略太短的无意义片段 (如 "OK", "嗯")
        if len(sent) < 2:
            continue

        # 编码当前句子
        sent_embedding = model.encode(sent, convert_to_tensor=True)

        # 计算相似度
        pos_scores = util.cos_sim(sent_embedding, pos_embeddings)[0]
        neg_scores = util.cos_sim(sent_embedding, neg_embeddings)[0]

        best_pos_score = torch.max(pos_scores).item()
        best_pos_idx = torch.argmax(pos_scores).item()
        
        best_neg_score = torch.max(neg_scores).item()
        best_neg_idx = torch.argmax(neg_scores).item()

        # 决策逻辑 (Winner Takes All for this sentence)
        matched_label = None
        matched_polarity = None
        confidence = 0.0

        # 1. 比较正向和负向的最高分
        if best_pos_score > best_neg_score:
            # 倾向于好评
            if best_pos_score > threshold:
                matched_label = POS_LABELS_LIST[best_pos_idx]
                matched_polarity = "positive"
                confidence = best_pos_score
            else:
                # 没过阈值，但句子看起来是中性/正向的
                # 这里我们利用整条评论的星级做兜底
                if review_polarity_base == "positive":
                    matched_label = "好评其他"
                    matched_polarity = "positive"
                    confidence = 0.3 # 低置信度
                else:
                    # 星级是差评，但这句话没匹配到差评库，可能是一句废话或“其他”
                    # 暂时忽略，除非它是该评论唯一的句子
                    pass 
        else:
            # 倾向于差评
            if best_neg_score > threshold:
                matched_label = NEG_LABELS_LIST[best_neg_idx]
                matched_polarity = "negative"
                confidence = best_neg_score
            else:
                if review_polarity_base == "negative":
                    matched_label = "差评其他"
                    matched_polarity = "negative"
                    confidence = 0.3
                else:
                    pass

        # 如果句子没匹配到任何具体标签，且被判定为“其他”，存入结果
        if matched_label:
            analyzed_results.append({
                "review_id": row_idx,
                "original_review": full_text,
                "sentence": sent,
                "polarity": matched_polarity,
                "label": matched_label,
                "evidence": sent, # 强证据：直接引用原句
                "confidence": round(confidence, 4)
            })

    # 兜底逻辑：如果整条评论拆完后，连一个标签都没打上（所有句子都低于阈值且被忽略）
    if not analyzed_results:
        fallback_label = "差评其他" if review_polarity_base == "negative" else "好评其他"
        analyzed_results.append({
            "review_id": row_idx,
            "original_review": full_text,
            "sentence": "(整段语义模糊)",
            "polarity": review_polarity_base,
            "label": fallback_label,
            "evidence": full_text,
            "confidence": 0.0
        })

    return analyzed_results

# =========================
# 4. 辅助工具
# =========================
def load_file(f):
    if f.name.lower().endswith(".csv"):
        try: return pd.read_csv(f, encoding="utf-8")
        except: return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating_strict(x):
    """强制提取评分整数"""
    if pd.isna(x): return np.nan
    s = str(x)
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        val = float(m.group(1))
        val_int = int(round(val))
        return max(1, min(5, val_int))
    return np.nan

# =========================
# 5. 主程序 UI
# =========================
st.title("🔬 AI 评论精细化分析系统")
st.markdown("""
**核心逻辑更新：**
1. **语义拆解**：自动将长评论拆分为独立句子，分别打标（解决一条评论既好又坏的问题）。
2. **强证据约束**：标签必须对应原文的具体句子 (`evidence`)。
3. **兜底规则**：未匹配到库的语义，依据星级归入“好评其他”或“差评其他”。
""")

with st.spinner("正在加载 NLP 语义模型..."):
    model = load_model()

uploaded = st.file_uploader("上传评论文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('正在逐句拆解并分析语义...'):
        df_raw = load_file(uploaded)
        
        # 1. 字段识别
        all_cols = df_raw.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in str(c) or "rating" in str(c).lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in str(c) or "review" in str(c).lower() or "text" in str(c).lower()), all_cols[1])
        
        # 2. 清洗
        df_raw["rating_clean"] = df_raw[rating_col].apply(parse_rating_strict)
        df_raw = df_raw.dropna(subset=["rating_clean"])
        df_raw["text_clean"] = df_raw[text_col].astype(str).fillna("")
        
        # 3. 核心运算：生成结构化打标表 (Granular DataFrame)
        all_structured_data = []
        
        # 进度条
        progress_bar = st.progress(0)
        total_rows = len(df_raw)
        
        for idx, row in df_raw.iterrows():
            if idx % 10 == 0: progress_bar.progress(idx / total_rows)
            
            # 调用拆句分析函数
            results = analyze_single_review(
                row_idx=idx, # 使用索引作为 ID
                rating=row["rating_clean"],
                full_text=row["text_clean"],
                model=model
            )
            all_structured_data.extend(results)
            
        progress_bar.empty()
        
        # 生成最终 DataFrame
        detailed_df = pd.DataFrame(all_structured_data)
        
    st.success(f"✅ 分析完成！原数据 {len(df_raw)} 条，拆解出 {len(detailed_df)} 个语义单元。")

    # =========================
    # A: 结构化数据展示
    # =========================
    st.markdown("---")
    st.header("1. 结构化打标结果 (Structured Data)")
    st.markdown("每一行代表一个“语义单元”，而非一条完整的评论。")
    
    st.dataframe(
        detailed_df[["review_id", "label", "evidence", "sentence", "confidence"]], 
        use_container_width=True,
        height=400
    )

    # =========================
    # B: 统计可视化 (Matplotlib 降级方案)
    # =========================
    st.markdown("---")
    st.header("2. 标签分布统计")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 标签分布")
        # 统计标签频率
        label_counts = detailed_df["label"].value_counts().head(10)
        
        # 使用 Matplotlib 绘图
        fig, ax = plt.subplots(figsize=(8, 5))
        # 颜色映射：好评绿，差评红，其他灰
        colors = []
        for lbl in label_counts.index:
            if "其他" in lbl: colors.append("#95a5a6")
            elif lbl in POS_LABELS_LIST: colors.append("#2ecc71")
            else: colors.append("#e74c3c")
            
        bars = ax.barh(label_counts.index, label_counts.values, color=colors)
        ax.invert_yaxis() # 翻转Y轴让第一名在上面
        ax.set_xlabel("Mentions")
        ax.set_title("Label Frequency")
        
        # 在柱状图上添加数值
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
            
        st.pyplot(fig)

    with col2:
        st.subheader("情感占比 (拆句后)")
        polarity_counts = detailed_df["polarity"].value_counts()
        
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(
            polarity_counts.values, 
            labels=polarity_counts.index, 
            autopct='%1.1f%%', 
            colors=["#e74c3c", "#2ecc71", "#3498db"],
            startangle=90
        )
        ax2.set_title("Polarity Distribution (Sentence Level)")
        st.pyplot(fig2)

    # =========================
    # C: 证据回溯工具
    # =========================
    st.markdown("---")
    st.header("3. 证据回溯 (Traceability)")
    
    selected_label = st.selectbox("选择一个标签查看证据:", detailed_df["label"].unique())
    
    evidence_df = detailed_df[detailed_df["label"] == selected_label][["review_id", "evidence", "original_review"]]
    
    if not evidence_df.empty:
        st.write(f"共找到 {len(evidence_df)} 条证据：")
        for i, row in evidence_df.head(5).iterrows():
            with st.expander(f"Review #{row['review_id']}: \"{row['evidence']}\""):
                st.info(f"**完整原文:** {row['original_review']}")
    else:
        st.write("无数据")

    # =========================
    # 下载区
    # =========================
    st.markdown("---")
    
    # 导出 CSV
    csv_buffer = detailed_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="⬇️ 下载结构化打标结果 (CSV)",
        data=csv_buffer,
        file_name="structured_analysis_result.csv",
        mime="text/csv"
    )
    
    # 导出 Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        detailed_df.to_excel(writer, index=False, sheet_name='Structured_Data')
        # 同时也把原始数据放进去方便对比
        df_raw.to_excel(writer, index=False, sheet_name='Raw_Data')
        
    st.download_button(
        label="⬇️ 下载完整分析报表 (Excel)",
        data=buffer.getvalue(),
        file_name="structured_analysis_report.xlsx",
        mime="application/vnd.ms-excel"
    )
