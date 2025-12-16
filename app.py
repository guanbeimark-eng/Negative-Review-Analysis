import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io

# =========================
# 0. 页面配置与基础设置
# =========================
st.set_page_config(
    page_title="AI 深度语义分析看板 (可视化增强版)",
    page_icon="📊",
    layout="wide"
)

# Matplotlib 中文支持与样式设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif'] # 适配 Windows/Mac
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot') # 使用更好看的绘图风格

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
# 1. 标签库定义
# =========================
POS_LABELS_LIST = [
    "面料舒适", "质量很好", "有助于锻炼", "有助于缓解疼痛", "保暖", "舒适贴合", 
    "有压缩感", "抓握式有效", "合身", "有助于关节炎/扳机指", "增加手指灵活", 
    "促进血液循环", "耐用", "缓解不适", "轻盈", "覆盖整个手指", "有助于防止受伤"
]

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
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# =========================
# 3. 核心 NLP 引擎
# =========================
def split_into_sentences(text):
    """拆句"""
    if not isinstance(text, str): return []
    sentences = re.split(r'[.!?;。！？；\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 1]

def analyze_single_review(row_idx, rating, full_text, model, threshold=0.35):
    """拆句并打标"""
    sentences = split_into_sentences(full_text)
    analyzed_results = []
    
    pos_embeddings = model.encode(POS_LABELS_LIST, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS_LIST, convert_to_tensor=True)
    
    review_polarity_base = "negative" if rating <= 3 else "positive"

    # 如果无法拆句，整句处理
    if not sentences:
        fallback_label = "差评其他" if review_polarity_base == "negative" else "好评其他"
        return [{
            "review_id": row_idx,
            "rating": rating,
            "original_review": full_text,
            "sentence": full_text,
            "polarity": review_polarity_base,
            "label": fallback_label,
            "evidence": full_text,
            "confidence": 0.5
        }]

    for sent in sentences:
        sent_embedding = model.encode(sent, convert_to_tensor=True)
        pos_scores = util.cos_sim(sent_embedding, pos_embeddings)[0]
        neg_scores = util.cos_sim(sent_embedding, neg_embeddings)[0]

        best_pos_score = torch.max(pos_scores).item()
        best_pos_idx = torch.argmax(pos_scores).item()
        best_neg_score = torch.max(neg_scores).item()
        best_neg_idx = torch.argmax(neg_scores).item()

        matched_label = None
        matched_polarity = None
        confidence = 0.0

        if best_pos_score > best_neg_score:
            if best_pos_score > threshold:
                matched_label = POS_LABELS_LIST[best_pos_idx]
                matched_polarity = "positive"
                confidence = best_pos_score
        else:
            if best_neg_score > threshold:
                matched_label = NEG_LABELS_LIST[best_neg_idx]
                matched_polarity = "negative"
                confidence = best_neg_score

        if matched_label:
            analyzed_results.append({
                "review_id": row_idx,
                "rating": rating,
                "original_review": full_text,
                "sentence": sent,
                "polarity": matched_polarity,
                "label": matched_label,
                "evidence": sent,
                "confidence": round(confidence, 4)
            })

    # 兜底：如果没有任一句子匹配到标签
    if not analyzed_results:
        fallback_label = "差评其他" if review_polarity_base == "negative" else "好评其他"
        analyzed_results.append({
            "review_id": row_idx,
            "rating": rating,
            "original_review": full_text,
            "sentence": "(无明确特征语义)",
            "polarity": review_polarity_base,
            "label": fallback_label,
            "evidence": full_text,
            "confidence": 0.0
        })

    return analyzed_results

# =========================
# 4. 辅助函数
# =========================
def load_file(f):
    if f.name.lower().endswith(".csv"):
        try: return pd.read_csv(f, encoding="utf-8")
        except: return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating_strict(x):
    if pd.isna(x): return np.nan
    s = str(x)
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        val = int(round(float(m.group(1))))
        return max(1, min(5, val))
    return np.nan

# =========================
# 5. 主程序 UI
# =========================
st.title("📊 AI 深度语义分析看板")
st.markdown("""
**核心能力：**
1. **语义拆解**：解决“一条评论既说好又说坏”的分析难题。
2. **强证据关联**：所有分析结果均可回溯到具体的原文句子。
3. **多维可视化**：无需 Plotly，使用原生 Matplotlib 绘制高级嵌套图表。
""")

with st.spinner("AI 模型加载中..."):
    model = load_model()

uploaded = st.file_uploader("上传文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('AI 正在逐句阅读分析...'):
        df_raw = load_file(uploaded)
        
        all_cols = df_raw.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in str(c) or "rating" in str(c).lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in str(c) or "review" in str(c).lower() or "text" in str(c).lower()), all_cols[1])
        
        df_raw["rating_clean"] = df_raw[rating_col].apply(parse_rating_strict)
        df_raw = df_raw.dropna(subset=["rating_clean"])
        df_raw["text_clean"] = df_raw[text_col].astype(str).fillna("")
        
        all_results = []
        progress_bar = st.progress(0)
        total = len(df_raw)
        
        for idx, row in df_raw.iterrows():
            if idx % 10 == 0: progress_bar.progress(idx / total)
            res = analyze_single_review(idx, row["rating_clean"], row["text_clean"], model)
            all_results.extend(res)
        
        progress_bar.empty()
        detailed_df = pd.DataFrame(all_results)

    st.success(f"✅ 分析完成！解析出 {len(detailed_df)} 个语义切片。")

    # =========================
    # 可视化 A: 宏观与星级
    # =========================
    st.markdown("---")
    st.header("1. 宏观数据概览")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("总评论数", len(df_raw))
    col_kpi2.metric("平均评分", f"{df_raw['rating_clean'].mean():.2f} ⭐")
    neg_rate = (len(df_raw[df_raw['rating_clean']<=3])/len(df_raw))*100
    col_kpi3.metric("差评率 (<=3星)", f"{neg_rate:.1f}%", delta_color="inverse")

    # 星级分布图 (Matplotlib)
    fig_stars, ax_stars = plt.subplots(figsize=(10, 3))
    star_counts = df_raw['rating_clean'].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
    colors_stars = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71'] # 红到绿
    bars = ax_stars.bar(star_counts.index, star_counts.values, color=colors_stars, alpha=0.8)
    ax_stars.set_title('星级评分分布')
    ax_stars.set_xticks([1,2,3,4,5])
    ax_stars.set_ylabel('评论数量')
    ax_stars.grid(axis='y', linestyle='--', alpha=0.3)
    # 标数值
    for bar in bars:
        height = bar.get_height()
        ax_stars.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    st.pyplot(fig_stars)

    # =========================
    # 可视化 B: 情感与标签 (嵌套环形图)
    # =========================
    st.markdown("---")
    st.header("2. 市场深度分析 (Nested Analysis)")
    
    col_viz1, col_viz2 = st.columns([1.5, 1])

    with col_viz1:
        st.subheader("情感与标签构成 (嵌套环形图)")
        st.caption("内圈：情感 (正/负) | 外圈：具体标签")
        
        # 准备数据
        # 1. 情感分布
        polarity_counts = detailed_df['polarity'].value_counts()
        # 2. 标签分布
        label_counts = detailed_df.groupby(['polarity', 'label']).size()
        
        # 绘图数据准备
        inner_labels = polarity_counts.index
        inner_sizes = polarity_counts.values
        inner_colors = ['#2ecc71' if l=='positive' else '#e74c3c' for l in inner_labels]
        
        # 外圈数据对齐
        outer_sizes = []
        outer_colors = []
        outer_labels_text = []
        
        for pol in inner_labels:
            if pol in label_counts:
                sub_labels = label_counts[pol].sort_values(ascending=False)
                # 只显示Top N标签，其他的归为"其他"以防图表太乱
                top_n = sub_labels.head(6)
                others = sub_labels.iloc[6:].sum()
                
                # 基础颜色
                base_color = '#27ae60' if pol=='positive' else '#c0392b'
                alphas = np.linspace(0.9, 0.3, len(top_n) + (1 if others > 0 else 0))
                
                for idx, (lbl, count) in enumerate(top_n.items()):
                    outer_sizes.append(count)
                    outer_labels_text.append(lbl if count/len(detailed_df) > 0.02 else "") # 占比太小不显示文字
                    # 变色处理
                    outer_colors.append(base_color) # 简化：使用纯色，或者可以调整透明度
                
                if others > 0:
                    outer_sizes.append(others)
                    outer_labels_text.append("")
                    outer_colors.append(base_color)

        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        
        # 外圈
        ax_pie.pie(outer_sizes, labels=outer_labels_text, radius=1, 
                   colors=outer_colors, wedgeprops=dict(width=0.3, edgecolor='w'),
                   textprops={'fontsize': 9}, labeldistance=1.05)
        
        # 内圈
        ax_pie.pie(inner_sizes, labels=[l.upper() for l in inner_labels], radius=0.7, 
                   colors=inner_colors, wedgeprops=dict(width=0.3, edgecolor='w'),
                   textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}, labeldistance=0.6)
        
        ax_pie.set(aspect="equal")
        st.pyplot(fig_pie)

    with col_viz2:
        st.subheader("标签排行榜 (Top 10)")
        
        top_labels = detailed_df['label'].value_counts().head(10).sort_values()
        
        fig_barh, ax_barh = plt.subplots(figsize=(6, 8))
        # 颜色映射
        bar_colors = []
        for l in top_labels.index:
            if l in POS_LABELS_LIST: bar_colors.append('#2ecc71')
            elif l in NEG_LABELS_LIST: bar_colors.append('#e74c3c')
            else: bar_colors.append('#95a5a6')
            
        ax_barh.barh(top_labels.index, top_labels.values, color=bar_colors)
        ax_barh.set_xlabel("提及次数")
        
        # 图例
        pos_patch = mpatches.Patch(color='#2ecc71', label='好评')
        neg_patch = mpatches.Patch(color='#e74c3c', label='差评')
        other_patch = mpatches.Patch(color='#95a5a6', label='其他')
        ax_barh.legend(handles=[pos_patch, neg_patch, other_patch], loc='lower right')
        
        st.pyplot(fig_barh)

    # =========================
    # C: 证据回溯与原声
    # =========================
    st.markdown("---")
    st.header("3. 痛点原声透视")
    st.caption("基于语义拆解，直接定位到差评的具体句子")
    
    # 筛选差评标签
    neg_options = detailed_df[detailed_df['polarity']=='negative']['label'].unique()
    if len(neg_options) > 0:
        selected_neg = st.selectbox("选择差评问题:", neg_options)
        
        evidence_data = detailed_df[detailed_df['label'] == selected_neg]
        st.write(f"共发现 {len(evidence_data)} 处相关反馈：")
        
        for i, row in evidence_data.head(5).iterrows():
            with st.expander(f"来自评分 {row['rating']}星的评论"):
                st.markdown(f"**原声证据:** :red[{row['evidence']}]")
                st.caption(f"**完整上下文:** {row['original_review']}")
    else:
        st.info("数据中未发现明显差评。")

    # =========================
    # 下载区
    # =========================
    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        detailed_df.to_excel(writer, index=False, sheet_name='Detailed_Analysis')
        df_raw.to_excel(writer, index=False, sheet_name='Raw_Data')
        
    st.download_button(
        label="⬇️ 下载完整 Excel 分析报表",
        data=buffer.getvalue(),
        file_name="sentiment_analysis_report.xlsx",
        mime="application/vnd.ms-excel"
    )
