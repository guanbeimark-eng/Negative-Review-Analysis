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
st.set_page_config(page_title="AI 评论分析 (修复版)", page_icon="🔧", layout="wide")

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
# 1. 标签库与关键词规则 (核心修复)
# =========================
# 定义标签的同时，定义“强制关注词”。如果评论包含这些词，AI 会加倍关注对应标签。
POS_LABELS_MAP = {
    "面料舒适/柔软": ["soft", "comfortable", "fabric", "material", "舒适", "软", "面料"],
    "做工质量好": ["quality", "well made", "sturdy", "质量", "做工"],
    "缓解疼痛/医疗效果": ["pain", "relief", "arthritis", "ache", "疼痛", "缓解", "关节炎"],
    "尺码合身/舒适贴合": ["fit", "size", "snug", "perfect", "合身", "尺码"],
    "增加抓握力/防滑": ["grip", "slip", "traction", "抓握", "滑"],
    "耐用性强": ["durable", "last", "tear", "耐用", "破"],
}

NEG_LABELS_MAP = {
    "尺码太小/太紧/伸不进去": ["small", "tight", "fit", "cut off", "circulation", "cuff", "hand in", "紧", "小", "勒", "伸不进", "窄"],
    "尺码太大/太松": ["big", "loose", "huge", "large", "松", "大", "长"],
    "无效/没有作用": ["work", "effect", "useless", "help", "无效", "没用"],
    "缝线开裂/破损": ["seam", "rip", "tear", "hole", "split", "缝线", "破", "洞", "开裂"],
    "面料质量差/廉价": ["material", "thin", "cheap", "rough", "scratchy", "面料", "薄", "粗糙"],
    "太滑/没有抓握力": ["slippery", "slide", "no grip", "smooth", "滑", "抓不住"],
    "过敏/皮疹/发痒": ["rash", "itch", "allergy", "skin", "痒", "过敏", "红肿"]
}

# 提取纯标签列表供模型编码
POS_LABELS = list(POS_LABELS_MAP.keys())
NEG_LABELS = list(NEG_LABELS_MAP.keys())

# =========================
# 2. AI 模型加载
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# =========================
# 3. 核心功能：混合打标 (关键词 + 语义)
# =========================
def extract_dynamic_label(text, model, ngram_range=(2, 3)):
    """提取新标签"""
    try:
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        analyzer_type = 'char' if is_chinese else 'word'
        count = CountVectorizer(ngram_range=ngram_range, analyzer=analyzer_type, stop_words='english').fit([text])
        candidates = count.get_feature_names_out()
        if len(candidates) == 0: return "其他未分类"
        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)
        distances = util.cos_sim(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-1:]]
        tag = keywords[0]
        return tag.replace(" ", "") if is_chinese else tag.title()
    except:
        return "其他(文本过短)"

def hybrid_classify(df, model, match_threshold=0.35):
    """
    混合打标逻辑：
    1. 关键词增强：如果评论含有 "tight", "fit"，会给 "尺码" 类标签加分。
    2. 语义匹配：使用 AI 计算向量相似度。
    """
    reviews = df['text'].tolist()
    
    # 1. 向量编码
    review_embeddings = model.encode(reviews, convert_to_tensor=True)
    pos_embeddings = model.encode(POS_LABELS, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS, convert_to_tensor=True)
    
    # 2. 计算原始相似度
    pos_sims = util.cos_sim(review_embeddings, pos_embeddings)
    neg_sims = util.cos_sim(review_embeddings, neg_embeddings)
    
    final_labels = []
    sentiment_display = []
    is_new_label = []

    progress_bar = st.progress(0)
    total = len(df)
    
    for i in range(total):
        if i % 10 == 0: progress_bar.progress(i / total)

        rating = df.iloc[i]['rating']
        text = str(df.iloc[i]['text']).lower()
        
        # --- 关键词加权 (Booster) ---
        # 如果评论里有 "tight"，则 "尺码太小" 的相似度分数 +0.3
        
        # 处理差评权重
        current_neg_scores = neg_sims[i].clone()
        for idx, label in enumerate(NEG_LABELS):
            keywords = NEG_LABELS_MAP[label]
            if any(k in text for k in keywords):
                current_neg_scores[idx] += 0.35  # 显著提升包含关键词的标签分数

        # 处理好评权重
        current_pos_scores = pos_sims[i].clone()
        for idx, label in enumerate(POS_LABELS):
            keywords = POS_LABELS_MAP[label]
            if any(k in text for k in keywords):
                current_pos_scores[idx] += 0.35

        # 获取加权后的最佳匹配
        best_pos_idx = torch.argmax(current_pos_scores).item()
        best_pos_score = current_pos_scores[best_pos_idx].item()
        
        best_neg_idx = torch.argmax(current_neg_scores).item()
        best_neg_score = current_neg_scores[best_neg_idx].item()
        
        label = None
        s_display = "未知"
        is_new = False
        
        # --- 严格的情感判定 ---
        # 3星绝对是差评
        if rating <= 3:
            is_negative = True
        elif rating == 4:
            is_negative = best_neg_score > best_pos_score
        else:
            is_negative = False

        # --- 最终决策 ---
        if is_negative:
            s_display = "差评"
            if best_neg_score > match_threshold:
                label = NEG_LABELS[best_neg_idx]
            else:
                label = extract_dynamic_label(df.iloc[i]['text'], model)
                is_new = True
        else:
            s_display = "好评"
            if best_pos_score > match_threshold:
                label = POS_LABELS[best_pos_idx]
            else:
                label = extract_dynamic_label(df.iloc[i]['text'], model)
                is_new = True
        
        final_labels.append(label)
        sentiment_display.append(s_display)
        is_new_label.append(is_new)

    progress_bar.empty()
    return final_labels, sentiment_display, is_new_label

# =========================
# 4. 辅助工具 (评分修复)
# =========================
def load_file(f):
    if f.name.lower().endswith(".csv"):
        try: return pd.read_csv(f, encoding="utf-8")
        except: return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating_strict(x):
    """严格解析评分，强制转为 1-5 的整数"""
    if pd.isna(x): return np.nan
    s = str(x)
    # 提取数字
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        val = float(m.group(1))
        # 四舍五入并取整
        val_int = int(round(val))
        # 边界保护
        if val_int < 1: val_int = 1
        if val_int > 5: val_int = 5
        return val_int
    return np.nan

# =========================
# 5. 主程序 UI
# =========================
st.title("📊 AI 评论分析 (Hybrid 增强版)")
st.markdown("""
**本次更新修复：**
1. **评分修正**：强制将所有评分（如 3.0, 4.0）转为整数，准确统计 3 星差评。
2. **打标修正**：引入“关键词规则”，当评论提到“袖口”、“伸不进”时，强制判定为【尺码问题】，不再误判为滑。
""")

with st.spinner("AI 引擎启动中..."):
    model = load_model()

uploaded = st.file_uploader("上传评论文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('正在清洗数据并进行混合分析...'):
        df = load_file(uploaded)
        
        # 1. 字段识别
        all_cols = df.columns.tolist()
        # 尝试找 rating 列，如果没有包含 "rating" 或 "星" 的列，默认用第几列
        rating_col_candidates = [c for c in all_cols if "星" in str(c) or "rating" in str(c).lower() or "score" in str(c).lower()]
        text_col_candidates = [c for c in all_cols if "内容" in str(c) or "review" in str(c).lower() or "text" in str(c).lower() or "body" in str(c).lower()]
        
        rating_col = rating_col_candidates[0] if rating_col_candidates else all_cols[0]
        text_col = text_col_candidates[0] if text_col_candidates else all_cols[1]

        # 2. 严格清洗数据 (修复图1的问题)
        # 强制转换为整数
        df["rating_clean"] = df[rating_col].apply(parse_rating_strict)
        # 去除无效评分
        df = df.dropna(subset=["rating_clean"])
        df["rating_clean"] = df["rating_clean"].astype(int)
        
        df["text"] = df[text_col].astype(str).fillna("")
        
        # 为了后续代码兼容，将 rating_clean 映射回 rating
        df["rating"] = df["rating_clean"]
        
        # 3. 核心运算
        labels, sentiments, is_new = hybrid_classify(df, model)
        df["标签"] = labels
        df["情感分类"] = sentiments
        df["是否新标签"] = is_new
        
    st.success("✅ 分析完成！")

    # =========================
    # A: 宏观概览 (修复版)
    # =========================
    st.markdown("---")
    st.header("1. 宏观数据概览 (已修复)")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("评论总数", len(df))
    
    avg_score = df['rating'].mean()
    k2.metric("平均评分", f"{avg_score:.2f} ⭐")
    
    # 严格计算 <=3 星
    neg_df = df[df['rating'] <= 3]
    neg_count = len(neg_df)
    neg_rate = (neg_count / len(df) * 100) if len(df) > 0 else 0
    
    k3.metric("差评占比 (<=3星)", f"{neg_rate:.1f}%", delta_color="inverse")
    k4.metric("新标签挖掘数", sum(is_new))
    
    # 星级分布图 (修复为离散柱状图)
    st.subheader("评分等级分布")
    # 强制统计 1-5 的每一个数量，即使是 0 也要显示
    counts = df['rating'].value_counts().reindex([1,2,3,4,5], fill_value=0).reset_index()
    counts.columns = ["星级", "数量"]
    # 强制星级为字符串，防止 Plotly 把它当连续数字画
    counts["星级"] = counts["星级"].astype(str) + "星"
    
    fig_bar = px.bar(counts, x="星级", y="数量", text="数量", color="数量", color_continuous_scale="Blues")
    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # B: 深度可视化
    # =========================
    st.markdown("---")
    st.header("2. 标签深度分析")
    
    c1, c2 = st.columns(2)
    with c1:
        st.caption("情感分布环形图")
        s_counts = df["情感分类"].value_counts().reset_index()
        s_counts.columns = ["情感", "数量"]
        fig_pie = px.pie(s_counts, values="数量", names="情感", hole=0.4, 
                         color="情感", color_discrete_map={"好评":"#2ecc71", "差评":"#e74c3c"})
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.caption("问题层级旭日图")
        # 过滤低频
        viz_df = df.copy()
        tc = viz_df["标签"].value_counts()
        viz_df["标签展示"] = viz_df["标签"].apply(lambda x: x if tc[x] > 0 else "其他")
        
        sun_df = viz_df.groupby(["情感分类", "标签展示"]).size().reset_index(name="数量")
        fig_sun = px.sunburst(sun_df, path=['情感分类', '标签展示'], values='数量',
                              color='情感分类', color_discrete_map={"好评":"#2ecc71", "差评":"#e74c3c"})
        st.plotly_chart(fig_sun, use_container_width=True)

    # =========================
    # C: 差评原声 (验证修复结果)
    # =========================
    st.markdown("---")
    st.header("3. 差评原声透视")
    st.caption("请检查：'尺码问题' 是否包含了抱怨袖口紧的评论")
    
    if not neg_df.empty:
        # 只看差评
        neg_issues = neg_df["标签"].value_counts().index.tolist()
        selected_issue = st.selectbox("选择差评标签查看:", neg_issues)
        
        reviews = neg_df[neg_df["标签"] == selected_issue][["rating", "text"]]
        
        st.markdown(f"**标签【{selected_issue}】下的评论:**")
        for idx, row in reviews.iterrows():
            st.warning(f"[{row['rating']}星] {row['text']}")
    else:
        st.info("恭喜，当前数据中没有 <=3 星的差评。")

    # =========================
    # 下载
    # =========================
    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Result')
    
    st.download_button("⬇️ 下载 Excel 结果", buffer.getvalue(), "fixed_analysis.xlsx", "application/vnd.ms-excel")
