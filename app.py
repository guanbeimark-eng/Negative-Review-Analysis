import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io

# =========================
# 0. 页面配置与安全验证
# =========================
st.set_page_config(
    page_title="AI 市场洞察系统 (线上版)",
    page_icon="🧠",
    layout="wide"
)

# --- 🔒 密码保护 (线上部署必备) ---
# 默认密码是 admin123，您可以修改
ACCESS_PASSWORD = "admin123" 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    if st.session_state["password_input"] == ACCESS_PASSWORD:
        st.session_state.logged_in = True
    else:
        st.error("密码错误，请重试")

if not st.session_state.logged_in:
    st.markdown("## 🔒 系统锁定 (线上部署模式)")
    st.markdown("该分析系统包含敏感市场数据，请输入密码访问。")
    st.text_input("访问密码", type="password", key="password_input", on_change=check_password)
    st.stop() 

# =========================
# 1. 标签库定义 (标准库)
# =========================
POS_LABELS = [
    "面料舒适/柔软", "做工质量好", "缓解疼痛/医疗效果", "保暖性能好", 
    "尺码合身/舒适贴合", "提供压缩感/支撑力", "增加抓握力/防滑", 
    "关节炎/扳机指辅助", "灵活性好", "耐用性强", "轻盈透气"
]

NEG_LABELS = [
    "无效/没有作用", "缝线开裂/破损", "收到二手/脏污", "面料质量差/廉价", 
    "尺码太小/太紧", "尺码太大/太松", "接缝处磨手/不适", "不耐用/一次性", 
    "过敏/皮疹/发痒", "太滑/没有抓握力", "数量不符/发错货", "导致血液循环受阻"
]

POS_OTHER = "其他好评"
NEG_OTHER = "其他差评"

# =========================
# 2. AI 模型加载 (针对云端优化)
# =========================
# 注意：使用 @st.cache_resource 确保模型只加载一次，节省云端内存
@st.cache_resource
def load_model():
    # all-MiniLM-L6-v2 模型很小 (~80MB)，非常适合 Streamlit Cloud 免费版
    return SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 3. 语义打标逻辑
# =========================
def semantic_classify(df, model, threshold=0.25):
    """
    使用向量相似度进行高精度打标
    """
    reviews = df['text'].tolist()
    
    # 1. 批量编码评论
    review_embeddings = model.encode(reviews, convert_to_tensor=True)
    
    # 2. 编码标签库
    pos_embeddings = model.encode(POS_LABELS, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS, convert_to_tensor=True)
    
    # 3. 计算相似度矩阵
    pos_sims = util.cos_sim(review_embeddings, pos_embeddings)
    neg_sims = util.cos_sim(review_embeddings, neg_embeddings)
    
    final_labels = []
    
    # 为了显示进度条
    progress_bar = st.progress(0)
    total = len(df)
    
    for i in range(total):
        # 每处理10%更新一次进度条，避免UI卡顿
        if i % (total // 10 + 1) == 0:
            progress_bar.progress(i / total)

        rating = df.iloc[i]['rating']
        
        # 获取最高分
        p_scores = pos_sims[i]
        n_scores = neg_sims[i]
        
        best_pos_idx = torch.argmax(p_scores).item()
        best_pos_score = p_scores[best_pos_idx].item()
        
        best_neg_idx = torch.argmax(n_scores).item()
        best_neg_score = n_scores[best_neg_idx].item()
        
        # --- 决策逻辑 ---
        # 1-3星：强制匹配差评库
        if rating <= 3:
            if best_neg_score > threshold:
                final_labels.append(NEG_LABELS[best_neg_idx])
            else:
                final_labels.append(NEG_OTHER)
        # 5星：强制匹配好评库
        elif rating == 5:
            if best_pos_score > threshold:
                final_labels.append(POS_LABELS[best_pos_idx])
            else:
                final_labels.append(POS_OTHER)
        # 4星：摇摆逻辑
        else:
            if best_neg_score > threshold and best_neg_score > best_pos_score:
                final_labels.append(NEG_LABELS[best_neg_idx])
            elif best_pos_score > threshold:
                final_labels.append(POS_LABELS[best_pos_idx])
            else:
                final_labels.append(POS_OTHER)
                
    progress_bar.empty() # 清除进度条
    return final_labels

# =========================
# 4. 辅助工具
# =========================
def load_file(f):
    if f.name.lower().endswith(".csv"):
        try: return pd.read_csv(f, encoding="utf-8")
        except: return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating(x):
    if pd.isna(x): return np.nan
    m = re.search(r"(\d+(\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan

def get_sentiment_type(tag):
    if tag in POS_LABELS or tag == POS_OTHER: return "Positive"
    if tag in NEG_LABELS or tag == NEG_OTHER: return "Negative"
    return "Unknown"

# =========================
# 5. 主程序 UI
# =========================
st.title("🧠 AI 深度语义分析系统 (Cloud Ver.)")
st.markdown("此版本运行在云端，第一次加载 AI 模型可能需要 10-20 秒，请耐心等待。")

# 懒加载模型
with st.spinner("正在唤醒 AI 引擎..."):
    model = load_model()

uploaded = st.file_uploader("上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('AI 正在逐行阅读并理解评论...'):
        df = load_file(uploaded)
        
        # 字段自动识别
        all_cols = df.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in c or "rating" in c.lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in c or "review" in c.lower()), all_cols[1])
        
        # 基础清洗
        df["rating"] = df[rating_col].apply(parse_rating).round().astype(int)
        df = df[df["rating"].between(1, 5)]
        df["text"] = df[text_col].astype(str).fillna("")
        
        # AI 打标
        df["Tag_Label"] = semantic_classify(df, model)
        df["Sentiment_Type"] = df["Tag_Label"].apply(get_sentiment_type)
        
    st.success(f"✅ 分析完成！已处理 {len(df)} 条评论。")

    # =========================
    # 模块 A: 宏观概览
    # =========================
    st.markdown("---")
    st.header("1. 市场宏观概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("平均评分", f"{df['rating'].mean():.2f} ⭐")
    c2.metric("好评率 (5星)", f"{(len(df[df['rating']==5])/len(df)*100):.1f}%")
    c3.metric("差评率 (1-3星)", f"{(len(df[df['rating']<=3])/len(df)*100):.1f}%", delta_color="inverse")

    # =========================
    # 模块 B: 痛点分析
    # =========================
    st.markdown("---")
    st.header("2. 痛点分析 (Top Complaints)")
    
    neg_df = df[df["Sentiment_Type"] == "Negative"]
    if not neg_df.empty:
        viz_neg_df = neg_df[neg_df["Tag_Label"] != NEG_OTHER]
        if viz_neg_df.empty: viz_neg_df = neg_df

        neg_counts = viz_neg_df["Tag_Label"].value_counts().reset_index()
        neg_counts.columns = ["Issue", "Count"]
        
        fig_neg = px.bar(neg_counts, x="Count", y="Issue", orientation='h', 
                         title="主要投诉分布", color="Count", color_continuous_scale="Reds")
        fig_neg.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_neg, use_container_width=True)
        
        st.subheader("🔍 痛点原声透视")
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            sel_neg_tag = st.selectbox("选择痛点标签:", neg_counts["Issue"].unique())
        with col_n2:
            st.markdown(f"**用户抱怨 '{sel_neg_tag}' 的原话:**")
            sample_neg = neg_df[neg_df["Tag_Label"] == sel_neg_tag].sort_values(by="text", key=lambda x: x.str.len(), ascending=False).head(5)
            for i, row in sample_neg.iterrows():
                with st.expander(f"💔 {row['rating']}星: ...{row['text'][:50]}..."):
                    st.write(row['text'])
    else:
        st.info("暂无明显差评数据。")

    # =========================
    # 模块 C: 卖点挖掘
    # =========================
    st.markdown("---")
    st.header("3. 卖点挖掘 (Selling Points)")
    
    pos_df = df[df["Sentiment_Type"] == "Positive"]
    if not pos_df.empty:
        viz_pos_df = pos_df[pos_df["Tag_Label"] != POS_OTHER]
        if viz_pos_df.empty: viz_pos_df = pos_df
        
        pos_counts = viz_pos_df["Tag_Label"].value_counts().reset_index()
        pos_counts.columns = ["Feature", "Count"]
        
        fig_tree = px.treemap(pos_counts, path=['Feature'], values='Count',
                              title="卖点权重分布",
                              color='Count', color_continuous_scale='Greens')
        st.plotly_chart(fig_tree, use_container_width=True)
        
        st.subheader("💡 卖点原声透视")
        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            sel_pos_tag = st.selectbox("选择卖点标签:", pos_counts["Feature"].unique())
        with col_p2:
            st.markdown(f"**用户夸赞 '{sel_pos_tag}' 的原话:**")
            sample_pos = pos_df[pos_df["Tag_Label"] == sel_pos_tag].sort_values(by="text", key=lambda x: x.str.len(), ascending=False).head(5)
            for i, row in sample_pos.iterrows():
                with st.expander(f"❤️ 5星: ...{row['text'][:50]}..."):
                    st.write(row['text'])
    else:
        st.info("暂无好评数据。")

    # =========================
    # 模块 D: 机会挖掘
    # =========================
    st.markdown("---")
    st.header("4. 机会挖掘 (4-Star Analysis)")
    four_star = df[df['rating'] == 4]
    if not four_star.empty:
        f_counts = four_star["Tag_Label"].value_counts().reset_index()
        f_counts.columns = ["Label", "Count"]
        f_counts["Type"] = f_counts["Label"].apply(get_sentiment_type)
        
        fig_sun = px.sunburst(f_counts, path=['Type', 'Label'], values='Count',
                              title="4星评价成分分析",
                              color='Type', 
                              color_discrete_map={'Positive':'#66c2a5', 'Negative':'#d53e4f', 'Unknown':'#999999'})
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.write("暂无4星评论。")

    # =========================
    # 下载区
    # =========================
    st.markdown("---")
    
    # CSV 下载
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "⬇️ 下载分析报表 (CSV)",
        data=csv_data,
        file_name="ai_analysis_report.csv",
        mime="text/csv"
    )
    
    # Excel 下载 (解决乱码最稳妥的方式)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
    
    st.download_button(
        label="⬇️ 下载分析报表 (Excel - 推荐)",
        data=buffer.getvalue(),
        file_name="ai_analysis_report.xlsx",
        mime="application/vnd.ms-excel"
    )
