import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer, util
import torch
import re
import io
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# =========================
# 0. 页面配置与字体修复
# =========================
st.set_page_config(
    page_title="AI 全维评论分析看板 (Pro Ver.)",
    page_icon="📊",
    layout="wide"
)

# --- 字体自动配置逻辑 (防止云端中文乱码) ---
def configure_matplotlib_font():
    """
    尝试找到系统可用的中文字体，如果找不到则回退到默认
    """
    # 常见中文字体列表 (Windows, Mac, Linux)
    font_candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    system_fonts = set(f.name for f in fm.fontManager.ttflist)
    found_font = None
    
    for f in font_candidates:
        if f in system_fonts:
            found_font = f
            break
            
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
    else:
        # 如果实在没找到，尝试设置为 sans-serif，至少显示英文
        plt.rcParams['font.sans-serif'] = ['sans-serif']
        
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

configure_matplotlib_font()

# --- 访问密码 ---
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
# 3. 核心 NLP 引擎
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
            "sentence": full_text[:50], # 截取部分作为展示
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

    # 兜底：如果整条评论没有任何句子匹配到标签
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
st.title("📊 AI 全维评论分析看板 (可视化增强版)")
st.markdown("""
**核心功能：**
1. **语义拆解**：自动拆分长难句，精准归类好评与差评点。
2. **多维可视化**：包含情感分布、标签对比、星级交叉分析及时间趋势（若有日期）。
3. **强证据链**：所有分析结果均关联原文句子。
""")

with st.spinner("正在加载 AI 神经模型..."):
    model = load_model()

uploaded = st.file_uploader("上传文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded:
    with st.spinner('正在进行深度语义拆解...'):
        df_raw = load_file(uploaded)
        
        # 字段智能识别
        all_cols = df_raw.columns.tolist()
        # 1. 星级列
        rating_col = next((c for c in all_cols if "星" in str(c) or "rating" in str(c).lower()), all_cols[0])
        # 2. 内容列
        text_col = next((c for c in all_cols if "内容" in str(c) or "review" in str(c).lower() or "text" in str(c).lower()), all_cols[1])
        # 3. 日期列 (可选)
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

        # 核心分析循环
        all_results = []
        progress_bar = st.progress(0)
        total = len(df_raw)
        
        for idx, row in df_raw.iterrows():
            if idx % 10 == 0: progress_bar.progress(idx / total)
            res = analyze_single_review(
                idx, 
                row["rating_clean"], 
                row["date_clean"], 
                row["text_clean"], 
                model
            )
            all_results.extend(res)
        
        progress_bar.empty()
        
        # 转换为打标层级的 DataFrame
        detailed_df = pd.DataFrame(all_results)

    st.success(f"✅ 分析完成！从 {len(df_raw)} 条评论中拆解出 {len(detailed_df)} 个语义单元。")

    # ==========================================
    # 维度 1: 宏观概览 (KPI & 基础分布)
    # ==========================================
    st.markdown("---")
    st.header("1. 宏观数据概览")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("总评论数", len(df_raw))
    k1.metric("语义单元数", len(detailed_df), help="一条评论可能拆分成多个语义点")
    
    avg_score = df_raw['rating_clean'].mean()
    k2.metric("平均评分", f"{avg_score:.2f} ⭐")
    
    # 差评率
    neg_reviews = len(df_raw[df_raw['rating_clean']<=3])
    k3.metric("差评率 (Review Level)", f"{(neg_reviews/len(df_raw)*100):.1f}%", delta_color="inverse")

    # 绘制星级分布 (Bar Chart)
    st.subheader("评分星级分布")
    star_counts = df_raw['rating_clean'].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
    
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
    bars = ax1.bar(star_counts.index, star_counts.values, color=colors, alpha=0.9)
    ax1.set_xticks([1,2,3,4,5])
    ax1.set_ylabel("数量")
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')
    
    st.pyplot(fig1)

    # ==========================================
    # 维度 2: 标签深度分析 (好评 vs 差评)
    # ==========================================
    st.markdown("---")
    st.header("2. 标签深度透视")
    
    c1, c2 = st.columns(2)
    
    # --- 左侧：情感占比饼图 ---
    with c1:
        st.subheader("语义情感占比")
        pol_counts = detailed_df['polarity'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(pol_counts.values, labels=pol_counts.index, autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'], startangle=140, explode=(0.05, 0))
        ax2.set_title("Sentiment Distribution")
        st.pyplot(fig2)
        
    # --- 右侧：标签 Top 榜单 (对比图) ---
    with c2:
        st.subheader("Top 标签对比")
        # 分别提取好评和差评的前5名
        top_pos = detailed_df[detailed_df['polarity']=='positive']['label'].value_counts().head(5)
        top_neg = detailed_df[detailed_df['polarity']=='negative']['label'].value_counts().head(5)
        
        # 合并绘图数据
        labels = list(top_pos.index) + list(top_neg.index)
        counts = list(top_pos.values) + list(top_neg.values)
        colors = ['#2ecc71']*len(top_pos) + ['#e74c3c']*len(top_neg)
        
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        y_pos = np.arange(len(labels))
        ax3.barh(y_pos, counts, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        ax3.invert_yaxis() # 最大的在上面
        ax3.set_xlabel("提及次数")
        ax3.set_title("Top Positive vs Top Negative Labels")
        st.pyplot(fig3)

    # ==========================================
    # 维度 3: 交叉分析 (星级 x 情感)
    # ==========================================
    st.markdown("---")
    st.header("3. 交叉分析：星级背后的真实声音")
    st.caption("检查：高分评论里是否藏着差评标签？低分评论里是否有好评点？")
    
    # 交叉表：星级 vs 情感
    cross_tab = pd.crosstab(detailed_df['rating'], detailed_df['polarity'])
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    cross_tab.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=ax4)
    ax4.set_xlabel("星级")
    ax4.set_ylabel("语义单元数量")
    ax4.set_title("星级与情感分布堆叠图")
    ax4.legend(["Negative (差评点)", "Positive (好评点)"], loc='upper left')
    plt.xticks(rotation=0)
    st.pyplot(fig4)
    
    #
