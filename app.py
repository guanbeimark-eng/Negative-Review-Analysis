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
st.set_page_config(page_title="AI 评论分析 (语义修正版)", page_icon="🎯", layout="wide")

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
# 1. 标签库与关键词规则 (深度优化)
# =========================
# 逻辑说明：如果评论中包含列表里的词，该标签的分数会获得巨大加成。

POS_LABELS_MAP = {
    # 提高 "功能性" 标签的优先级，防止被 "舒适" 掩盖
    "提供压缩感/支撑力": ["compression", "pressure", "support", "tightness", "squeeze", "压力", "压缩", "支撑", "紧实", "包裹"],
    "缓解疼痛/医疗效果": ["pain", "relief", "arthritis", "ache", "soothing", "hurts", "疼痛", "缓解", "关节炎", "止痛", "疗效"],
    "增加抓握力/防滑": ["grip", "traction", "slip", "rubber", "抓握", "防滑", "摩擦", "稳"],
    "保暖性能好": ["warm", "heat", "cold", "winter", "保暖", "热", "冷", "温"],
    
    # 通用标签放在后面
    "面料舒适/柔软": ["soft", "comfortable", "fabric", "cotton", "smooth", "cozy", "舒适", "软", "棉", "舒服"],
    "做工质量好": ["quality", "well made", "sturdy", "stitch", "质量", "做工", "缝线", "耐用"],
    "尺码合身/舒适贴合": ["fit", "size", "snug", "perfect", "true to size", "合身", "合适", "贴合"],
    "耐用性强": ["durable", "last", "wash", "wear", "耐用", "洗", "磨损"],
    "灵活性好": ["dexterity", "flexible", "type", "write", "灵活", "打字", "活动"],
}

NEG_LABELS_MAP = {
    # 针对您的案例1：增加 "袖口", "伸不进" 等具体场景词
    "尺码太小/太紧/伸不进去": [
        "small", "tight", "cut off", "circulation", "cuff", "hand in", "wrist", "opening", 
        "restrict", "squeeze", "tiny", "child",
        "紧", "小", "勒", "伸不进", "窄", "袖口", "穿不", "进不去", "卡住", "血液循环"
    ],
    "尺码太大/太松": ["big", "loose", "huge", "large", "baggy", "fall off", "long", "松", "大", "长", "掉"],
    "太滑/没有抓握力": ["slippery", "slide", "no grip", "smooth", "plastic", "drop", "滑", "抓不住", "溜"],
    "缝线开裂/破损": ["seam", "rip", "tear", "hole", "split", "fray", "thread", "unravel", "缝线", "破", "洞", "开裂", "线头", "裂"],
    "无效/没有作用": ["work", "effect", "useless", "help", "difference", "waste", "无效", "没用", "智商税", "不值"],
    "过敏/皮疹/发痒": ["rash", "itch", "allergy", "skin", "red", "bump", "痒", "过敏", "红肿", "刺挠"],
    "面料质量差/廉价": ["material", "thin", "cheap", "rough", "scratchy", "junk", "paper", "面料", "薄", "粗糙", "廉价", "烂"],
    "数量不符/发错货": ["count", "missing", "wrong", "received", "order", "数量", "少", "发错", "缺"],
}

# 提取标签列表
POS_LABELS = list(POS_LABELS_MAP.keys())
NEG_LABELS = list(NEG_LABELS_MAP.keys())

# =========================
# 2. AI 模型加载
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# =========================
# 3. 核心功能：混合打标 (关键词 > 语义)
# =========================
def extract_dynamic_label(text, model, ngram_range=(2, 3)):
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
    逻辑升级：
    1. 关键词命中时，给予巨大加分 (Bonus +1.5)，确保覆盖语义相似度。
    2. 如果有特定功能词（如“压力”），优先于通用词（如“舒适”）。
    """
    reviews = df['text'].tolist()
    
    review_embeddings = model.encode(reviews, convert_to_tensor=True)
    pos_embeddings = model.encode(POS_LABELS, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS, convert_to_tensor=True)
    
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
        
        # --- 关键词强力加权 ---
        
        # 1. 处理差评
        current_neg_scores = neg_sims[i].clone()
        for idx, label in enumerate(NEG_LABELS):
            keywords = NEG_LABELS_MAP[label]
            # 检查是否包含关键词
            if any(k in text for k in keywords):
                # +1.5 是一个巨大的权重，基本能保证只要有关键词，就选这个标签
                current_neg_scores[idx] += 1.5 

        # 2. 处理好评
        current_pos_scores = pos_sims[i].clone()
        for idx, label in enumerate(POS_LABELS):
            keywords = POS_LABELS_MAP[label]
            if any(k in text for k in keywords):
                # 针对案例2：如果是"压缩/压力"类词，加分更高，压过"舒适"
                if "压缩" in label or "compression" in label.lower():
                     current_pos_scores[idx] += 2.0 
                else:
                     current_pos_scores[idx] += 1.5

        # 获取最佳匹配
        best_pos_idx = torch.argmax(current_pos_scores).item()
        best_pos_score = current_pos_scores[best_pos_idx].item()
        
        best_neg_idx = torch.argmax(current_neg_scores).item()
        best_neg_score = current_neg_scores[best_neg_idx].item()
        
        label = None
        s_display = "未知"
        is_new = False
        
        # --- 严格的情感判定 (修复评分逻辑) ---
        if rating <= 3:
            is_negative = True
        elif rating == 4:
            is_negative = best_neg_score > best_pos_score
        else:
            is_negative = False

        # --- 最终决策 ---
        if is_negative:
            s_display = "差评"
            # 阈值判断：如果有关键词加成，分数肯定 > 1.0，直接通过
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
# 4. 辅助工具 (严格评分解析)
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
        val = float(m.group(1))
        val_int = int(round(val)) # 四舍五入
        if val_int < 1: val_int = 1
        if val_int > 5: val_int = 5
        return val_int
    return np.nan

# =========================
# 5. 主程序 UI
# =========================
st.title("🎯 AI 评论分析 (精准语义修正版)")
st.markdown("""
**本次修正重点：**
1. **解决“袖口伸不进”问题**：增加了 `袖口`, `伸不进`, `cuff` 等强规则词，强制识别为【尺码太小/太紧】。
2. **解决“压力被泛化”问题**：提高了功能性词汇（如 `压力`, `compression`）的权重，优先于通用的“舒适”。
3. **评分统计修复**：强制将所有评分（如 3.0）转为整数，准确统计差评。
""")

with st.spinner("AI 引擎加载中..."):
    model = load_model()

uploaded = st.file_uploader("上传评论文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('正在进行关键词增强分析...'):
        df = load_file(uploaded)
        
        all_cols = df.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in str(c) or "rating" in str(c).lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in str(c) or "review" in str(c).lower() or "text" in str(c).lower()), all_cols[1])

        # 严格清洗
        df["rating_clean"] = df[rating_col].apply(parse_rating_strict)
        df = df.dropna(subset=["rating_clean"])
        df["rating_clean"] = df["rating_clean"].astype(int)
        df["text"] = df[text_col].astype(str).fillna("")
        df["rating"] = df["rating_clean"]
        
        # 核心运算
        labels, sentiments, is_new = hybrid_classify(df, model)
        df["标签"] = labels
        df["情感分类"] = sentiments
        df["是否新标签"] = is_new
        
    st.success("✅ 分析完成！")

    # =========================
    # A: 宏观概览
    # =========================
    st.markdown("---")
    st.header("1. 宏观概览")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("评论总数", len(df))
    avg_score = df['rating'].mean()
    k2.metric("平均评分", f"{avg_score:.2f} ⭐")
    
    neg_count = len(df[df['rating'] <= 3])
    neg_rate = (neg_count / len(df) * 100) if len(df) > 0 else 0
    k3.metric("差评占比 (<=3星)", f"{neg_rate:.1f}%", delta_color="inverse")
    k4.metric("新标签挖掘", sum(is_new))
    
    # 评分分布
    counts = df['rating'].value_counts().reindex([1,2,3,4,5], fill_value=0).reset_index()
    counts.columns = ["星级", "数量"]
    counts["星级"] = counts["星级"].astype(str) + "星"
    fig_bar = px.bar(counts, x="星级", y="数量", text="数量", color="数量", color_continuous_scale="Blues")
    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # B: 深度分析
    # =========================
    st.markdown("---")
    st.header("2. 标签深度分析")
    c1, c2 = st.columns(2)
    with c1:
        s_counts = df["情感分类"].value_counts().reset_index()
        s_counts.columns = ["情感", "数量"]
        fig_pie = px.pie(s_counts, values="数量", names="情感", hole=0.4, 
                         color="情感", color_discrete_map={"好评":"#2ecc71", "差评":"#e74c3c"})
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        viz_df = df.copy()
        tc = viz_df["标签"].value_counts()
        viz_df["标签展示"] = viz_df["标签"].apply(lambda x: x if tc[x] > 0 else "其他")
        sun_df = viz_df.groupby(["情感分类", "标签展示"]).size().reset_index(name="数量")
        fig_sun = px.sunburst(sun_df, path=['情感分类', '标签展示'], values='数量',
                              color='情感分类', color_discrete_map={"好评":"#2ecc71", "差评":"#e74c3c"})
        st.plotly_chart(fig_sun, use_container_width=True)

    # =========================
    # C: 验证区 (查找特定评论)
    # =========================
    st.markdown("---")
    st.header("3. 结果验证")
    st.caption("检查特定标签下的评论是否准确")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        # 差评验证
        neg_issues = df[df["情感分类"] == "差评"]["标签"].unique().tolist()
        if neg_issues:
            sel_neg = st.selectbox("查看差评标签:", neg_issues)
            reviews_n = df[df["标签"] == sel_neg]["text"].head(3)
            for r in reviews_n: st.error(r)
        else:
            st.info("无差评")
            
    with col_v2:
        # 好评验证
        pos_issues = df[df["情感分类"] == "好评"]["标签"].unique().tolist()
        if pos_issues:
            sel_pos = st.selectbox("查看好评标签:", pos_issues)
            reviews_p = df[df["标签"] == sel_pos]["text"].head(3)
            for r in reviews_p: st.success(r)
        else:
            st.info("无好评")

    # =========================
    # 下载
    # =========================
    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Result')
    st.download_button("⬇️ 下载 Excel 结果", buffer.getvalue(), "fixed_analysis.xlsx", "application/vnd.ms-excel")
