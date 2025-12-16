import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="评论市场洞察系统",
    page_icon="📈",
    layout="wide"
)

# =========================
# 1. 标签库配置 (保持不变)
# =========================
POS_LABELS = [
    "面料舒适","质量很好","有助于锻炼","有助于缓解疼痛","保暖","舒适贴合",
    "有压缩感","抓握式有效","合身","有助于关节炎/扳机指","增加手指灵活",
    "促进血液循环","耐用","缓解不适","轻盈","覆盖整个手指","有助于防止肿胀"
]

NEG_LABELS = [
    "没有作用/没有效果","缝线裂开","二手商品","质量问题","不适合",
    "尺码太小","尺码不对","接缝处不舒适","不耐用",
    "尺码太大","过敏","光滑/没有抓握","实物与购买数量不一致"
]

POS_OTHER = "好评其他"
NEG_OTHER = "差评其他"

# =========================
# 2. Seed 词 (保持不变)
# =========================
SEEDS_POS = {
    "面料舒适": ["comfortable", "soft"],
    "质量很好": ["well made", "good quality"],
    "有助于缓解疼痛": ["pain relief", "arthritis"],
    "舒适贴合": ["fits well", "snug"],
    "有压缩感": ["compression"],
    "抓握式有效": ["grip"],
    "耐用": ["durable"]
}

SEEDS_NEG = {
    "没有作用/没有效果": ["no effect", "doesn't work"],
    "尺码太小": ["too small", "tight"],
    "尺码太大": ["too big", "loose"],
    "质量问题": ["poor quality", "cheap"],
    "不耐用": ["broke", "tear"],
    "光滑/没有抓握": ["slippery", "no grip"],
    "过敏": ["allergy", "rash"]
}

# =========================
# 3. 核心工具函数 (保持不变)
# =========================
def load_file(f):
    if f.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating(x):
    if pd.isna(x): 
        return np.nan
    m = re.search(r"(\d+(\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan

def tokenize(text):
    if not text:
        return []
    text = str(text).lower() # 强制转str防止报错
    eng = re.findall(r"[a-z]+", text)
    bigram = [f"{eng[i]} {eng[i+1]}" for i in range(len(eng)-1)]
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return eng + bigram + zh

# =========================
# 4 & 5. 学习权重与关键词 (保持不变)
# =========================
def learn_polarity_weights(texts, ratings, min_df=3):
    neg, pos = Counter(), Counter()
    for t, r in zip(texts, ratings):
        toks = tokenize(t)
        if r <= 3:
            neg.update(toks)
        elif r == 5:
            pos.update(toks)

    weights = {}
    for tok in set(neg) | set(pos):
        fn, fp = neg[tok], pos[tok]
        if fn + fp < min_df:
            continue
        # 平滑处理
        weights[tok] = math.log((fn + 1) / (fp + 1))
    return weights

def learn_label_kw(df, polarity_weights, topk=40):
    label_docs = defaultdict(list)

    for _, r in df.iterrows():
        toks = tokenize(r["text"])
        if r["rating"] <= 3:
            for lb, seeds in SEEDS_NEG.items():
                if any(s in r["text"].lower() for s in seeds):
                    label_docs[lb].append(toks)
        elif r["rating"] == 5:
            for lb, seeds in SEEDS_POS.items():
                if any(s in r["text"].lower() for s in seeds):
                    label_docs[lb].append(toks)

    label_kw = {}
    for lb, docs in label_docs.items():
        c = Counter()
        for d in docs:
            c.update(d)
        scores = {}
        for tok, f in c.items():
            if tok in polarity_weights:
                pol = polarity_weights[tok]
                # 过滤逻辑
                if (lb in NEG_LABELS and pol > 0) or (lb in POS_LABELS and pol < 0):
                    scores[tok] = abs(pol) * f
        label_kw[lb] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk])

    for lb in POS_LABELS + NEG_LABELS:
        label_kw.setdefault(lb, {})
    return label_kw

# =========================
# 6. 打标逻辑 (保持不变)
# =========================
def score_label(tokens, kw_map):
    return sum(kw_map.get(t, 0) for t in tokens)

def choose_tag(text, rating, label_kw):
    toks = tokenize(text)

    if rating <= 3:
        scores = {lb: score_label(toks, label_kw[lb]) for lb in NEG_LABELS}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else NEG_OTHER

    if rating == 5:
        scores = {lb: score_label(toks, label_kw[lb]) for lb in POS_LABELS}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else POS_OTHER

    # 4星：先差评
    neg_scores = {lb: score_label(toks, label_kw[lb]) for lb in NEG_LABELS}
    best_neg = max(neg_scores, key=neg_scores.get)
    if neg_scores[best_neg] > 0:
        return best_neg

    pos_scores = {lb: score_label(toks, label_kw[lb]) for lb in POS_LABELS}
    best_pos = max(pos_scores, key=pos_scores.get)
    return best_pos if pos_scores[best_pos] > 0 else POS_OTHER

# =========================
# 7. UI与高级可视化 (新增/修改部分)
# =========================
st.title("📈 评论市场洞察系统")
st.markdown("自动打标 + 商业可视化分析")

uploaded = st.file_uploader("上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('正在分析数据...'):
        df = load_file(uploaded)
        
        # 字段识别
        all_cols = df.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in c or "rating" in c.lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in c or "review" in c.lower()), all_cols[1])

        # 数据预处理
        df["rating"] = df[rating_col].apply(parse_rating).round().astype(int)
        df = df[df["rating"].between(1, 5)]
        df["text"] = df[text_col].astype(str)

        # 核心计算
        polarity_weights = learn_polarity_weights(df["text"], df["rating"])
        label_kw = learn_label_kw(df, polarity_weights)
        df["Tag_Label"] = df.apply(lambda r: choose_tag(r["text"], r["rating"], label_kw), axis=1)

        # 增加分类列辅助绘图
        def get_sentiment_type(tag):
            if tag in POS_LABELS or tag == POS_OTHER: return "Positive"
            if tag in NEG_LABELS or tag == NEG_OTHER: return "Negative"
            return "Unknown"
        df["Sentiment_Type"] = df["Tag_Label"].apply(get_sentiment_type)

    st.success("✅ 数据分析完成！")

    # =========================
    # 模块 A: 宏观市场概览
    # =========================
    st.markdown("---")
    st.header("1. 宏观市场概览")
    
    col1, col2, col3 = st.columns(3)
    
    # KPI 计算
    avg_rating = df["rating"].mean()
    neg_rate = (len(df[df["rating"]<=3]) / len(df)) * 100
    pos_rate = (len(df[df["rating"]==5]) / len(df)) * 100
    
    col1.metric("平均评分 (CSAT)", f"{avg_rating:.2f} ⭐")
    col2.metric("好评率 (5星)", f"{pos_rate:.1f}%")
    col3.metric("差评率 (1-3星)", f"{neg_rate:.1f}%", delta_color="inverse")

    # 图表：评分分布 (交互式)
    rating_counts = df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    fig_rating = px.bar(rating_counts, x="Rating", y="Count", 
                        title="用户评分分布", color="Count", 
                        color_continuous_scale="Blues")
    st.plotly_chart(fig_rating, use_container_width=True)

    # =========================
    # 模块 B: 痛点与改进 (Negative)
    # =========================
    st.markdown("---")
    st.header("2. 痛点分析：用户为什么流失？")
    st.caption("基于差评 (1-3星) 及部分4星负面反馈的数据")

    neg_df = df[df["Sentiment_Type"] == "Negative"]
    
    if not neg_df.empty:
        neg_counts = neg_df["Tag_Label"].value_counts().reset_index()
        neg_counts.columns = ["Issue", "Count"]
        
        # 1. 帕累托图 (Pareto Chart) 风格的柱状图
        fig_neg = px.bar(neg_counts, x="Count", y="Issue", orientation='h',
                         title="主要投诉问题排行 (Top Issues)",
                         color="Count", color_continuous_scale="Reds")
        fig_neg.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_neg, use_container_width=True)
        
        # 2. 深入挖掘具体问题
        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            selected_issue = st.selectbox("选择一个问题深入分析关键词:", neg_counts["Issue"].unique())
        
        with col_b2:
            if selected_issue in label_kw:
                keywords = label_kw[selected_issue]
                if keywords:
                    kw_df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Weight"]).head(15)
                    fig_kw = px.bar(kw_df, x="Keyword", y="Weight", 
                                    title=f"'{selected_issue}' 的高频触发词",
                                    color="Weight", color_continuous_scale="Reds")
                    st.plotly_chart(fig_kw, use_container_width=True)
                else:
                    st.info("该标签为通用标签或未提取到显著特征词。")
            else:
                st.info("该标签属于'其他'类，暂无特定特征词。")
    else:
        st.write("暂无差评数据，产品表现完美！")

    # =========================
    # 模块 C: 卖点与营销 (Positive)
    # =========================
    st.markdown("---")
    st.header("3. 卖点挖掘：广告语该怎么写？")
    st.caption("基于好评 (5星) 及部分4星正面反馈的数据")

    pos_df = df[df["Sentiment_Type"] == "Positive"]
    
    if not pos_df.empty:
        pos_counts = pos_df["Tag_Label"].value_counts().reset_index()
        pos_counts.columns = ["Selling Point", "Count"]

        # 树状图 (Treemap)：适合展示层级占比，很有营销感
        fig_tree = px.treemap(pos_counts, path=['Selling Point'], values='Count',
                              title="用户最欣赏的功能点 (Treemap)",
                              color='Count', color_continuous_scale='Greens')
        st.plotly_chart(fig_tree, use_container_width=True)
        
        # 关键词提取用于文案
        st.subheader("💡 营销文案灵感 (Copywriting Ideas)")
        top_pos_tag = pos_counts.iloc[0]["Selling Point"]
        st.markdown(f"用户最常提到的优点是 **{top_pos_tag}**。")
        
        if top_pos_tag in label_kw:
            top_words = list(label_kw[top_pos_tag].keys())[:10]
            st.info(f"推荐广告高频词: {', '.join(top_words)}")

    # =========================
    # 模块 D: 机会挖掘 (The 4-Star Gap)
    # =========================
    st.markdown("---")
    st.header("4. 机会挖掘：如何拯救摇摆用户 (4星分析)")
    st.caption("4星用户通常对产品大体满意，但有一两个具体抱怨。解决这些问题最能提升评分。")

    four_star_df = df[df["rating"] == 4]
    if not four_star_df.empty:
        # 统计4星里的差评标签 vs 好评标签
        fs_counts = four_star_df["Tag_Label"].value_counts().reset_index()
        fs_counts.columns = ["Label", "Count"]
        fs_counts["Type"] = fs_counts["Label"].apply(get_sentiment_type)
        
        fig_4s = px.sunburst(fs_counts, path=['Type', 'Label'], values='Count',
                             title="4星用户评价构成 (Sunburst)",
                             color='Type', color_discrete_map={'Positive':'#66c2a5', 'Negative':'#d53e4f', 'Unknown':'#grey'})
        st.plotly_chart(fig_4s, use_container_width=True)
        
        # 找出4星用户最主要的抱怨
        fs_neg = fs_counts[fs_counts["Type"] == "Negative"]
        if not fs_neg.empty:
            top_complaint = fs_neg.iloc[0]["Label"]
            st.warning(f"⚠️ 阻碍4星用户给出满分的最大障碍是：**{top_complaint}**")
    else:
        st.write("样本中没有4星评价。")

    # =========================
    # 数据下载区
    # =========================
    st.markdown("---")
    st.subheader("📥 数据导出")
    st.dataframe(df[[rating_col, "Tag_Label", "text"]].head(50))
    
    st.download_button(
        "⬇️ 下载完整分析报表 (CSV)",
        df.to_csv(index=False).encode("utf-8-sig"),
        "market_insight_report.csv",
        "text/csv"
    )
