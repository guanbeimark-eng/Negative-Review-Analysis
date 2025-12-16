import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter, defaultdict

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="评论自动打标系统（数据驱动权重版）",
    page_icon="🏷️",
    layout="wide"
)

# =========================
# 标签主题定义（只定义“语义桶”，不写关键词）
# =========================
NEGATIVE_TOPICS = {
    "尺码偏小": ["small", "tight"],
    "尺码偏大": ["big", "large", "loose"],
    "不舒适 / 勒手": ["uncomfortable", "pain", "hurt"],
    "穿戴困难": ["hard", "difficult"],
    "支撑不足": ["support"],
    "质量差 / 易损": ["broke", "cheap", "poor"],
    "与描述不符": ["describe", "different"],
    "差评-其他问题": []
}

POSITIVE_TOPICS = {
    "佩戴舒适": ["comfortable", "soft"],
    "尺寸合适": ["perfect", "true"],
    "支撑性好": ["support"],
    "缓解疼痛": ["relief", "pain"],
    "质量好": ["well", "quality"],
    "性价比高": ["worth", "value"],
    "好评-整体满意": []
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

def tokenize(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = text.split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

# =========================
# 关键词权重学习
# =========================
def learn_keyword_weights(texts, ratings):
    neg_counter = Counter()
    pos_counter = Counter()

    for text, r in zip(texts, ratings):
        tokens = tokenize(text)
        if r <= 3:
            neg_counter.update(tokens)
        elif r == 5:
            pos_counter.update(tokens)

    weights = {}
    vocab = set(neg_counter) | set(pos_counter)
    for w in vocab:
        fn = neg_counter[w]
        fp = pos_counter[w]
        if fn + fp < 3:
            continue
        weight_neg = math.log((fn + 1) / (fp + 1))
        weight_pos = math.log((fp + 1) / (fn + 1))
        weights[w] = (weight_neg, weight_pos)
    return weights

def score_text(text, topic_keywords, weights, mode):
    score = 0.0
    tokens = tokenize(text)
    for t in tokens:
        if t in weights:
            w_neg, w_pos = weights[t]
            if mode == "neg":
                score += w_neg
            else:
                score += w_pos
    return score

def choose_label(text, rating, weights):
    if rating <= 3:
        scores = {
            tag: score_text(text, kws, weights, "neg")
            for tag, kws in NEGATIVE_TOPICS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "差评-其他问题"

    if rating == 5:
        scores = {
            tag: score_text(text, kws, weights, "pos")
            for tag, kws in POSITIVE_TOPICS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "好评-整体满意"

    # 4 星：优先差评
    neg_scores = {
        tag: score_text(text, kws, weights, "neg")
        for tag, kws in NEGATIVE_TOPICS.items()
    }
    best_neg = max(neg_scores, key=neg_scores.get)
    if neg_scores[best_neg] > 0:
        return best_neg

    pos_scores = {
        tag: score_text(text, kws, weights, "pos")
        for tag, kws in POSITIVE_TOPICS.items()
    }
    best_pos = max(pos_scores, key=pos_scores.get)
    return best_pos if pos_scores[best_pos] > 0 else "好评-整体满意"

# =========================
# 主界面
# =========================
st.title("🏷️ 评论自动打标系统（关键词权重学习版）")

uploaded = st.file_uploader("上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

if uploaded:
    df = load_file(uploaded)

    cols = df.columns.tolist()
    col_rating = next((c for c in cols if "rating" in c.lower() or "星" in c), None)
    col_text = next((c for c in cols if "content" in c.lower() or "review" in c.lower() or "翻译" in c), None)

    if not col_rating or not col_text:
        st.error("无法识别星级或评论内容列")
        st.stop()

    df["rating"] = df[col_rating].apply(parse_rating).round().astype("Int64")
    df = df[df["rating"].between(1, 5)]
    df["text"] = df[col_text].astype(str)

    # 学习权重
    weights = learn_keyword_weights(df["text"], df["rating"])

    # 打标
    df["AI_Label"] = df.apply(lambda r: choose_label(r["text"], r["rating"], weights), axis=1)

    # 指标
    neg_rate = (df["rating"] <= 3).mean() * 100

    st.subheader("📊 数据概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("有效评论数", len(df))
    c2.metric("平均星级", f"{df['rating'].mean():.2f}")
    c3.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")

    st.bar_chart(df["rating"].value_counts().sort_index())

    st.subheader("🏷️ 打标结果预览")
    st.dataframe(df[["rating", "AI_Label", "text"]].head(20))

    out = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载数据驱动打标结果 CSV",
        out,
        "tagged_reviews_weighted.csv",
        "text/csv"
    )
