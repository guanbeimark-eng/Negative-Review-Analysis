import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import torch
import re
import io

# =========================
# 0. 页面配置与安全验证
# =========================
st.set_page_config(
    page_title="智能评论标签分析系统 (可视化版)",
    page_icon="📊",
    layout="wide"
)

# 访问密码
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
# 1. 您的标准标签库 (中文版)
# =========================
# AI 将自动计算这些中文标签与英文/中文评论的相似度
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

# =========================
# 2. AI 模型加载 (升级为多语言版)
# =========================
@st.cache_resource
def load_model():
    # 使用多语言模型，让 AI 能理解 "Soft" = "柔软"
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# =========================
# 3. 核心功能：标签匹配与发现
# =========================
def extract_dynamic_label(text, model, ngram_range=(2, 3)):
    """
    当评论不匹配标准库时，提取原文核心短语作为标签
    """
    try:
        # 简单判断是否包含中文，调整分词策略
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        analyzer_type = 'char' if is_chinese else 'word'
        
        # 提取候选词
        count = CountVectorizer(ngram_range=ngram_range, analyzer=analyzer_type, stop_words='english').fit([text])
        candidates = count.get_feature_names_out()
        
        if len(candidates) == 0: return "其他未分类"

        # 计算最核心的短语
        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)
        distances = util.cos_sim(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-1:]]
        
        # 格式化输出
        tag = keywords[0]
        return tag.replace(" ", "") if is_chinese else tag.title()
        
    except:
        return "其他(文本过短)"

def semantic_classify_and_discover(df, model, match_threshold=0.40):
    """
    主逻辑：标准库匹配 -> 情感判断 -> 新词发现
    """
    reviews = df['text'].tolist()
    
    # 批量编码 (速度最快的方式)
    review_embeddings = model.encode(reviews, convert_to_tensor=True)
    pos_embeddings = model.encode(POS_LABELS, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS, convert_to_tensor=True)
    
    pos_sims = util.cos_sim(review_embeddings, pos_embeddings)
    neg_sims = util.cos_sim(review_embeddings, neg_embeddings)
    
    final_labels = []
    sentiment_display = [] # 用于显示中文情感
    is_new_label = []

    # 进度条
    progress_bar = st.progress(0)
    total = len(df)
    
    for i in range(total):
        if i % 10 == 0: progress_bar.progress(i / total)

        rating = df.iloc[i]['rating']
        text = df.iloc[i]['text']
        
        best_pos_idx = torch.argmax(pos_sims[i]).item()
        best_pos_score = pos_sims[i][best_pos_idx].item()
        
        best_neg_idx = torch.argmax(neg_sims[i]).item()
        best_neg_score = neg_sims[i][best_neg_idx].item()
        
        label = None
        s_display = "未知"
        is_new = False
        is_negative = False

        # --- 情感判定 ---
        if rating <= 3:
            is_negative = True
        elif rating == 4:
            # 4星摇摆：谁分数高听谁的
            if best_neg_score > best_pos_score: is_negative = True
            else: is_negative = False
        else:
            is_negative = False

        # --- 匹配逻辑 ---
        if is_negative:
            s_display = "差评"
            # 1. 尝试匹配标准差评库
            if best_neg_score > match_threshold:
                label = NEG_LABELS[best_neg_idx]
            else:
                # 2. 挖掘新标签
                label = extract_dynamic_label(text, model)
                is_new = True
        else:
            s_display = "好评"
            # 1. 尝试匹配标准好评库
            if best_pos_score > match_threshold:
                label = POS_LABELS[best_pos_idx]
            else:
                # 2. 挖掘新标签
                label = extract_dynamic_label(text, model)
                is_new = True
        
        final_labels.append(label)
        sentiment_display.append(s_display)
        is_new_label.append(is_new)

    progress_bar.empty()
    return final_labels, sentiment_display, is_new_label

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

# =========================
# 5. 主程序 UI
# =========================
st.title("📊 智能评论标签分析系统")
st.markdown("AI 模型：**Multilingual-MiniLM** (支持中英互译匹配)")

with st.spinner("正在加载多语言 AI 模型..."):
    model = load_model()

uploaded = st.file_uploader("上传评论文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('AI 正在进行跨语言语义分析...'):
        df = load_file(uploaded)
        
        # 字段识别
        all_cols = df.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in c or "rating" in c.lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in c or "review" in c.lower()), all_cols[1])
        
        # 清洗
        df["rating"] = df[rating_col].apply(parse_rating).round().astype(int)
        df = df[df["rating"].between(1, 5)]
        df["text"] = df[text_col].astype(str).fillna("")
        
        # 核心运算
        labels, sentiments, is_new = semantic_classify_and_discover(df, model)
        df["标签"] = labels
        df["情感分类"] = sentiments
        df["是否新标签"] = is_new
        
    st.success("✅ 分析完成！")

    # =========================
    # 可视化模块 A: 宏观概览
    # =========================
    st.markdown("---")
    st.header("1. 数据概览")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("评论总数", len(df))
    k1.caption("有效数据行")
    
    avg_score = df['rating'].mean()
    k2.metric("平均评分", f"{avg_score:.2f} ⭐")
    
    neg_count = len(df[df['情感分类']=="差评"])
    neg_rate = neg_count / len(df) * 100
    k3.metric("差评占比", f"{neg_rate:.1f}%")
    
    new_tag_count = sum(is_new)
    k4.metric("新发现问题点", new_tag_count)
    k4.caption("标准库未覆盖的分类")

    # =========================
    # 可视化模块 B: 图表分析
    # =========================
    st.markdown("---")
    st.header("2. 可视化深度分析")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("情感构成分析")
        # 环形图 (Donut Chart)
        sent_counts = df["情感分类"].value_counts().reset_index()
        sent_counts.columns = ["情感", "数量"]
        fig_donut = px.pie(sent_counts, values="数量", names="情感", hole=0.4,
                           color="情感",
                           color_discrete_map={"好评": "#2ecc71", "差评": "#e74c3c"})
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("标签层级分布 (旭日图)")
        # 旭日图 (Sunburst)：展示 情感 -> 标签 的层级
        # 过滤掉数量太少的标签，防止图表太乱
        viz_df = df.copy()
        tag_counts = viz_df["标签"].value_counts()
        # 把出现少于2次的标签归为"其他"
        viz_df["显示标签"] = viz_df["标签"].apply(lambda x: x if tag_counts[x] > 1 else "其他低频标签")
        
        count_df = viz_df.groupby(["情感分类", "显示标签"]).size().reset_index(name="数量")
        fig_sun = px.sunburst(count_df, path=['情感分类', '显示标签'], values='数量',
                              color='情感分类',
                              color_discrete_map={"好评": "#2ecc71", "差评": "#e74c3c"})
        st.plotly_chart(fig_sun, use_container_width=True)

    # =========================
    # 可视化模块 C: 详细排行
    # =========================
    st.markdown("---")
    st.subheader("🏆 热门标签排行 (Top Issues)")
    
    # 区分颜色：标准库 vs 新发现
    std_set = set(POS_LABELS + NEG_LABELS)
    
    top_tags = df["标签"].value_counts().head(15).reset_index()
    top_tags.columns = ["标签名", "提及次数"]
    top_tags["类型"] = top_tags["标签名"].apply(lambda x: "标准库" if x in std_set else "新发现")
    
    fig_bar = px.bar(top_tags, x="提及次数", y="标签名", orientation='h', 
                     color="类型",
                     text="提及次数",
                     color_discrete_map={"标准库": "#3498db", "新发现": "#f1c40f"})
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # 模块 D: 差评原声透视
    # =========================
    st.markdown("---")
    st.header("3. 差评原声透视")
    st.caption("点击下方选择一个问题，查看用户具体在说什么")
    
    neg_df = df[df["情感分类"] == "差评"]
    
    if not neg_df.empty:
        neg_issues = neg_df["标签"].value_counts().index.tolist()
        selected_issue = st.selectbox("选择差评标签:", neg_issues)
        
        reviews = neg_df[neg_df["标签"] == selected_issue]["text"].head(5)
        
        st.markdown(f"**关于【{selected_issue}】的用户原声:**")
        for i, txt in enumerate(reviews):
            st.info(f"{i+1}. {txt}")
    else:
        st.success("暂无差评数据！")

    # =========================
    # 下载
    # =========================
    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
    
    st.download_button(
        label="⬇️ 下载 Excel 分析报告",
        data=buffer.getvalue(),
        file_name="analysis_report.xlsx",
        mime="application/vnd.ms-excel"
    )
