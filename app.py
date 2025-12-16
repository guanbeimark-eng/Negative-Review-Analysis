import streamlit as st
import pandas as pd
import numpy as np
import altair as alt  # 核心变动：使用 Altair 替代 Matplotlib
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io

# =========================
# 0. 页面配置
# =========================
st.set_page_config(
    page_title="AI 语义分析看板 (Altair版)",
    page_icon="📊",
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
    st.text_input("请输入访问密码", type="password", key="password_input", on_change=check_password)
    st.stop() 

# =========================
# 1. 标签库定义 (固定集合)
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
# 3. 核心 NLP 引擎 (保持原逻辑不变)
# =========================
def split_into_sentences(text):
    """拆句逻辑"""
    if not isinstance(text, str): return []
    sentences = re.split(r'[.!?;。！？；\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 1]

def analyze_single_review(row_idx, rating, date_val, full_text, model, threshold=0.40):
    """单条评论深度拆解"""
    sentences = split_into_sentences(full_text)
    analyzed_results = []
    
    pos_embeddings = model.encode(POS_LABELS_LIST, convert_to_tensor=True)
    neg_embeddings = model.encode(NEG_LABELS_LIST, convert_to_tensor=True)
    
    # 基于星级的基准情感
    review_polarity_base = "negative" if rating <= 3 else "positive"

    # 无法拆句或空评论处理
    if not sentences:
        fallback_label = "差评其他" if review_polarity_base == "negative" else "好评其他"
        return [{
            "review_id": row_idx,
            "date": date_val,
            "rating": rating,
            "original_review": full_text,
            "sentence": full_text[:50], 
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

        # 胜者通吃逻辑
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
                "date": date_val,
                "rating": rating,
                "original_review": full_text,
                "sentence": sent,
                "polarity": matched_polarity,
                "label": matched_label,
                "evidence": sent,
                "confidence": round(confidence, 4)
            })

    # 兜底
    if not analyzed_results:
        fallback_label = "差评其他" if review_polarity_base == "negative" else "好评其他"
        analyzed_results.append({
            "review_id": row_idx,
            "date": date_val,
            "rating": rating,
            "original_review": full_text,
            "sentence": "(无明确特征)",
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
st.title("📊 AI 全维评论分析看板 (无乱码版)")
st.markdown("""
**本次更新：**
1. **可视化重构**：弃用 Matplotlib，改用 **Altair**。图表文字由浏览器渲染，**彻底解决中文乱码/方框问题**。
2. **交互增强**：所有图表支持鼠标悬停查看详细数据。
3. **蝴蝶图**：好评向右，差评向左，对比更直观。
""")

with st.spinner("AI 神经模型加载中..."):
    model = load_model()

uploaded = st.file_uploader("上传文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('正在进行深度语义拆解...'):
        df_raw = load_file(uploaded)
        
        # 字段智能识别
        all_cols = df_raw.columns.tolist()
        rating_col = next((c for c in all_cols if "星" in str(c) or "rating" in str(c).lower()), all_cols[0])
        text_col = next((c for c in all_cols if "内容" in str(c) or "review" in str(c).lower() or "text" in str(c).lower()), all_cols[1])
        date_col = next((c for c in all_cols if "时间" in str(c) or "date" in str(c).lower() or "time" in str(c).lower()), None)
        
        # 清洗
        df_raw["rating_clean"] = df_raw[rating_col].apply(parse_rating_strict)
        df_raw = df_raw.dropna(subset=["rating_clean"])
        df_raw["text_clean"] = df_raw[text_col].astype(str).fillna("")
        
        # 处理日期
        has_date = False
        if date_col:
            try:
                df_raw["date_clean"] = pd.to_datetime(df_raw[date_col], errors='coerce')
                if df_raw["date_clean"].notna().sum() > 0:
                    has_date = True
            except:
                pass
        if not has_date:
            df_raw["date_clean"] = None

        # 核心分析
        all_results = []
        progress_bar = st.progress(0)
        total = len(df_raw)
        
        for idx, row in df_raw.iterrows():
            if idx % 10 == 0: progress_bar.progress(idx / total)
            res = analyze_single_review(idx, row["rating_clean"], row["date_clean"], row["text_clean"], model)
            all_results.extend(res)
        
        progress_bar.empty()
        detailed_df = pd.DataFrame(all_results)

    st.success(f"✅ 分析完成！拆解出 {len(detailed_df)} 个语义单元。")

    # ==========================================
    # 维度 1: 宏观概览
    # ==========================================
    st.markdown("---")
    st.header("1. 宏观数据概览")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("总评论数", len(df_raw))
    avg_score = df_raw['rating_clean'].mean()
    k2.metric("平均评分", f"{avg_score:.2f} ⭐")
    
    neg_reviews = len(df_raw[df_raw['rating_clean']<=3])
    k3.metric("差评率", f"{(neg_reviews/len(df_raw)*100):.1f}%", delta_color="inverse")
    
    # Altair: 星级分布柱状图
    st.subheader("评分分布")
    star_counts = df_raw['rating_clean'].value_counts().reset_index()
    star_counts.columns = ['Rating', 'Count']
    
    chart_stars = alt.Chart(star_counts).mark_bar().encode(
        x=alt.X('Rating:O', title='星级'), # O for Ordinal
        y=alt.Y('Count:Q', title='评论数'),
        color=alt.Color('Rating:O', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Rating', 'Count']
    ).properties(height=300)
    
    st.altair_chart(chart_stars, use_container_width=True)

    # ==========================================
    # 维度 2: 蝴蝶图 (好评 vs 差评)
    # ==========================================
    st.markdown("---")
    st.header("2. 市场口碑对比 (蝴蝶图)")
    st.caption("👈 左侧红色为差评痛点 | 右侧绿色为好评卖点 👉")
    
    # 数据准备
    label_counts = detailed_df.groupby(['label', 'polarity']).size().reset_index(name='count')
    # 让差评数量变成负数，以便在图中向左延伸
    label_counts['display_count'] = label_counts.apply(lambda x: -x['count'] if x['polarity'] == 'negative' else x['count'], axis=1)
    # 排序：按绝对值数量排序
    label_counts['abs_count'] = label_counts['count'].abs()
    
    # 过滤掉数量太少的标签，保持图表整洁
    top_labels = label_counts.sort_values('abs_count', ascending=False).head(20)

    # Altair: 蝴蝶图
    butterfly_chart = alt.Chart(top_labels).mark_bar().encode(
        x=alt.X('display_count:Q', title='提及次数 (负数代表差评)', axis=alt.Axis(format='d')),
        y=alt.Y('label:N', title=None, sort=alt.EncodingSortField(field="abs_count", order="descending")),
        color=alt.Color('polarity:N', scale=alt.Scale(domain=['negative', 'positive'], range=['#e74c3c', '#2ecc71']), legend=alt.Legend(title="情感倾向")),
        tooltip=[alt.Tooltip('label', title='标签'), alt.Tooltip('count', title='提及次数'), alt.Tooltip('polarity', title='情感')]
    ).properties(height=500)

    # 添加中间的文字标签 (可选，简单起见直接展示图)
    st.altair_chart(butterfly_chart, use_container_width=True)

    # ==========================================
    # 维度 3: 交叉分析 (星级堆叠图)
    # ==========================================
    st.markdown("---")
    st.header("3. 星级与语义成分分析")
    st.caption("查看每个星级中，包含了多少好评语义和差评语义")
    
    # 数据聚合
    stack_data = detailed_df.groupby(['rating', 'polarity']).size().reset_index(name='count')
    
    stack_chart = alt.Chart(stack_data).mark_bar().encode(
        x=alt.X('rating:O', title='星级'),
        y=alt.Y('count:Q', title='语义单元数量'),
        color=alt.Color('polarity:N', scale=alt.Scale(domain=['negative', 'positive'], range=['#e74c3c', '#2ecc71'])),
        tooltip=['rating', 'polarity', 'count']
    ).properties(height=400)
    
    st.altair_chart(stack_chart, use_container_width=True)

    # ==========================================
    # 维度 4: 证据回溯
    # ==========================================
    st.markdown("---")
    st.header("4. 证据回溯 (Traceability)")
    
    search_label = st.selectbox("🔍 选择标签查看原文证据:", detailed_df['label'].unique())
    
    subset = detailed_df[detailed_df['label'] == search_label]
    st.write(f"标签 **【{search_label}】** 共出现 {len(subset)} 次：")
    
    for i, row in subset.head(5).iterrows():
        with st.expander(f"{row['rating']}星 | 语义匹配度: {row['confidence']}"):
            # 高亮证据
            st.markdown(f"**拆解语义:** :red[{row['evidence']}]")
            st.caption(f"**完整原文:** {row['original_review']}")

    # ==========================================
    # 下载数据
    # ==========================================
    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        detailed_df.to_excel(writer, index=False, sheet_name='Detailed_Labels')
        df_raw.to_excel(writer, index=False, sheet_name='Raw_Data')
        
    st.download_button(
        label="⬇️ 下载完整分析报表 (Excel)",
        data=buffer.getvalue(),
        file_name="altair_analysis_report.xlsx",
        mime="application/vnd.ms-excel"
    )
