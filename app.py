import streamlit as st
import pandas as pd
import numpy as np
import re
import uuid

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="评论自动打标（规则引擎版）",
    page_icon="🏷️",
    layout="wide"
)

# =========================
# 内置评价库 + 关键词规则
# =========================
TAG_LIBRARY = {
    "positive": {
        "佩戴舒适": ["comfortable", "soft", "fits well", "no pain"],
        "尺寸合适": ["true to size", "perfect fit", "fits perfectly"],
        "质量好": ["good quality", "well made", "durable"],
        "性价比高": ["worth", "value", "price is good"],
    },
    "negative": {
        "尺码偏小": ["too small", "runs small", "tight"],
        "尺码偏大": ["too big", "runs large", "loose"],
        "穿戴困难": ["hard to put on", "difficult to wear"],
        "不舒适": ["uncomfortable", "pain", "hurts"],
        "质量差": ["poor quality", "broke", "cheap"],
    }
}

# =========================
# 工具函数
# =========================
def load_file(f):
    if f.name.lower().endswith(".csv"):
        return pd.read_csv(f, encoding="utf-8", errors="ignore")
    return pd.read_excel(f)

def parse_rating(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+(\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan

def auto_detect_column(cols, keywords):
    for k in keywords:
        for c in cols:
            if k.lower() in c.lower():
                return c
    return None

def keyword_score(text, keywords):
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)

def rule_based_label(row):
    rating = row["rating"]
    text = row["text"].lower()

    if rating <= 3:
        candidate_tags = TAG_LIBRARY["negative"]
    elif rating == 4:
        candidate_tags = {**TAG_LIBRARY["negative"], **TAG_LIBRARY["positive"]}
    else:
        candidate_tags = TAG_LIBRARY["positive"]

    scores = {
        tag: keyword_score(text, kws)
        for tag, kws in candidate_tags.items()
    }

    max_score = max(scores.values())
    if max_score == 0:
        return ""

    best_tags = [t for t, s in scores.items() if s == max_score]

    # 4 星平票时优先差评
    if rating == 4:
        for t in best_tags:
            if t in TAG_LIBRARY["negative"]:
                return t

    return best_tags[0]

# =========================
# 主界面
# =========================
st.title("🏷️ 评论自动打标系统（无模型 / 规则引擎版）")

uploaded = st.file_uploader("上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

if uploaded:
    df = load_file(uploaded)

    # 自动列识别
    cols = df.columns.tolist()
    col_rating = auto_detect_column(cols, ["rating", "星级"])
    col_text = auto_detect_column(cols, ["content", "review", "内容", "翻译"])

    if not col_rating or not col_text:
        st.error("无法识别星级列或内容列，请检查文件")
        st.stop()

    df["rating"] = df[col_rating].apply(parse_rating).round().astype("Int64")
    df = df[df["rating"].between(1, 5)]

    df["text"] = df[col_text].astype(str)
    df["id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]

    # 差评统计
    neg_rate = (df["rating"] <= 3).mean() * 100

    st.subheader("📊 数据概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("有效评论数", len(df))
    c2.metric("平均星级", f"{df['rating'].mean():.2f}")
    c3.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")

    st.bar_chart(df["rating"].value_counts().sort_index())

    # 自动打标（规则）
    df["AI_Label"] = df.apply(rule_based_label, axis=1)

    st.subheader("🏷️ 自动打标结果预览")
    st.dataframe(df[["rating", "AI_Label", "text"]].head(20))

    # 导出
    out = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载打标结果 CSV",
        out,
        "tagged_reviews_rule_based.csv",
        "text/csv"
    )
