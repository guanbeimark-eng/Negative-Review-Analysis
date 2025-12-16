import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import torch
import re
import io

# =========================
# 0. 页面配置与安全验证
# =========================
st.set_page_config(
    page_title="智能评论标签挖掘系统",
    page_icon="⛏️",
    layout="wide"
)

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
    st.text_input("访问密码", type="password", key="password_input", on_change=check_password)
    st.stop() 

# =========================
# 1. 您的标准标签库
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

# =========================
# 2. AI 模型加载
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 3. 核心功能：动态标签提取
# =========================
def extract_dynamic_label(text, model, ngram_range=(2, 3)):
    """
    当评论不匹配标准库时，从原文中提取最核心的短语作为新标签
    原理：KeyBERT 算法简化版
    """
    try:
        # 1. 使用 CountVectorizer 提取候选短语 (2-3个词的组合)
        # stop_words='english' 会自动过滤掉 the, is, at 等无意义词
        count = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit([text])
        candidates = count.get_feature_names_out()
        
        if len(candidates) == 0:
            return "其他未分类"

        # 2. 编码原文和所有候选短语
        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)

        # 3. 计算原文与候选短语的相似度
        distances = util.cos_sim(doc_embedding, candidate_embeddings)
        
        # 4. 取最相似的那个短语作为标签
        keywords = [candidates[index] for index in distances.argsort()[0][-1:]]
        
        # 将英文短语首字母大写，看起来更像标签
        return keywords[0].title()
        
    except Exception:
        # 如果文本太短或报错，返回默认
        return "其他(文本过短)"

def semantic_classify_and_discover(df, model, match_threshold=0.45):
    """
    双层逻辑：
    1. 优先匹配标准库 (相似度 > threshold)
    2. 匹配不到 -> 判断情感 -> 提取原文短语作为新标签
    """
    reviews = df['text'].tolist()
    
    # 批量编码，速度快
    review_embeddings = model.encode(reviews, convert_to_tensor=True)
    pos_embeddings = model.encode(POS_LABELS, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS, convert_to_tensor=True)
    
    # 计算相似度矩阵
    pos_sims = util.cos_sim(review_embeddings, pos_embeddings)
    neg_sims = util.cos_sim(review_embeddings, neg_embeddings)
    
    final_labels = []
    is_new_label = [] # 标记是否是新发现的标签
    sentiment_types = []

    progress_bar = st.progress(0)
    total = len(df)
    
    for i in range(total):
        if i % 10 == 0: progress_bar.progress(i / total)

        rating = df.iloc[i]['rating']
        text = df.iloc[i]['text']
        
        # 获取与标准库的最佳匹配
        best_pos_idx = torch.argmax(pos_sims[i]).item()
        best_pos_score = pos_sims[i][best_pos_idx].item()
        
        best_neg_idx = torch.argmax(neg_sims[i]).item()
        best_neg_score = neg_sims[i][best_neg_idx].item()
        
        label = None
        s_type = "Unknown"
        is_new = False

        # --- 逻辑 A: 判定情感方向 ---
        # 1-3星：差评；5星：好评；4星：看相似度
        is_negative = False
        if rating <= 3:
            is_negative = True
        elif rating == 4:
            if best_neg_score > best_pos_score: is_negative = True
            else: is_negative = False
        else:
            is_negative = False

        # --- 逻辑 B: 匹配或发现 ---
        
        if is_negative:
            s_type = "Negative"
            # 1. 尝试匹配标准差评库
            if best_neg_score > match_threshold:
                label = NEG_LABELS[best_neg_idx]
            else:
                # 2. 匹配失败，执行“新标签挖掘”
                label = extract_dynamic_label(text, model)
                is_new = True
        else:
            s_type = "Positive"
            # 1. 尝试匹配标准好评库
            if best_pos_score > match_threshold:
                label = POS_LABELS[best_pos_idx]
            else:
                # 2. 匹配失败，执行“新标签挖掘”
                label = extract_dynamic_label(text, model)
                is_new = True
        
        final_labels.append(label)
        sentiment_types.append(s_type)
        is_new_label.append(is_new)

    progress_bar.empty()
    return final_labels, sentiment_types, is_new_label

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
st.title("⛏️ 智能评论标签挖掘系统 (标准库 + 新词发现)")
st.markdown("""
**核心逻辑更新：**
1. **严格匹配**：首先检查评论是否符合您设定的 `POS_LABELS` 和 `NEG_LABELS`。
2. **新词发现**：如果不符合，AI 会自动分析是好评还是差评，并**从评论中提取核心短语**作为新标签。
""")

with st.spinner("正在加载 AI 引擎..."):
    model = load_model()

uploaded = st.file_uploader("上传评论文件", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('AI 正在逐行分析：匹配标准库 或 挖掘新标签...'):
        df = load_file(uploaded)
        
        # 字段识别
        all_cols = df.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in c or "rating" in c.lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in c or "review" in c.lower()), all_cols[1])
        
        # 清洗
        df["rating"] = df[rating_col].apply(parse_rating).round().astype(int)
        df = df[df["rating"].between(1, 5)]
        df["text"] = df[text_col].astype(str).fillna("")
        
        # === 核心运算 ===
        labels, sentiments, is_new = semantic_classify_and_discover(df, model)
        df["Tag_Label"] = labels
        df["Sentiment_Type"] = sentiments
        df["Is_New_Tag"] = is_new # 标记是否是新发现的标签
        
    st.success(f"✅ 处理完成！发现 {sum(is_new)} 条评论产生了新标签。")

    # =========================
    # 模块 A: 标签分布概览
    # =========================
    st.markdown("---")
    st.header("1. 标签分布概览")
    
    # 统计 Top 标签
    top_labels = df["Tag_Label"].value_counts().head(20).reset_index()
    top_labels.columns = ["Label", "Count"]
    
    # 标记哪些是新标签以便在图中区分
    std_set = set(POS_LABELS + NEG_LABELS)
    top_labels["Type"] = top_labels["Label"].apply(lambda x: "标准库" if x in std_set else "✨新发现")
    
    fig_bar = px.bar(top_labels, x="Count", y="Label", orientation='h', color="Type",
                     title="热门标签排行 (区分标准库与新发现)",
                     color_discrete_map={"标准库": "#1f77b4", "✨新发现": "#ff7f0e"})
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # 模块 B: 新发现的痛点 (差评挖掘)
    # =========================
    st.markdown("---")
    st.header("2. 🔍 新发现的潜在痛点 (不在标准库中)")
    st.caption("AI 识别出这些差评不属于您的标准库，并提取了以下核心短语：")
    
    new_neg_df = df[(df["Is_New_Tag"] == True) & (df["Sentiment_Type"] == "Negative")]
    
    if not new_neg_df.empty:
        # 统计新发现的差评标签
        new_neg_counts = new_neg_df["Tag_Label"].value_counts().reset_index()
        new_neg_counts.columns = ["New Issue", "Count"]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(new_neg_counts.head(10), hide_index=True)
        with c2:
            sel_new_issue = st.selectbox("选择一个新发现的问题查看原声:", new_neg_counts["New Issue"].unique())
            
            st.markdown(f"**用户关于 '{sel_new_issue}' 的原话:**")
            reviews = new_neg_df[new_neg_df["Tag_Label"] == sel_new_issue]["text"].head(5)
            for r in reviews:
                st.info(r)
    else:
        st.success("您的标准差评库覆盖了所有差评，未发现新问题！")

    # =========================
    # 模块 C: 详细数据表
    # =========================
    st.markdown("---")
    st.header("3. 详细分类数据")
    
    # 增加筛选器
    filter_type = st.radio("筛选查看:", ["全部", "仅查看新发现的标签", "仅查看标准库匹配"])
    
    view_df = df
    if filter_type == "仅查看新发现的标签":
        view_df = df[df["Is_New_Tag"] == True]
    elif filter_type == "仅查看标准库匹配":
        view_df = df[df["Is_New_Tag"] == False]
        
    st.dataframe(view_df[["rating", "Sentiment_Type", "Tag_Label", "Is_New_Tag", "text"]], height=400)

    # 下载
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
    
    st.download_button(
        label="⬇️ 下载完整 Excel 报表",
        data=buffer.getvalue(),
        file_name="smart_tag_discovery_report.xlsx",
        mime="application/vnd.ms-excel"
    )
