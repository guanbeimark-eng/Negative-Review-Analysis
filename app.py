import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter, defaultdict

st.set_page_config(
    page_title="评论自动打标系统（稳定版）",
    page_icon="🏷️",
    layout="wide"
)

# =========================
# 1. 标签库
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
# 2. Seed 词（弱监督）
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
# 3. 工具函数
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
    text = text.lower()
    eng = re.findall(r"[a-z]+", text)
    bigram = [f"{eng[i]} {eng[i+1]}" for i in range(len(eng)-1)]
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return eng + bigram + zh

# =========================
# 4. 学习极性权重
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
        weights[tok] = math.log((fn + 1) / (fp + 1))
    return weights

# =========================
# 5. 学习标签关键词
# =========================
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
                if (lb in NEG_LABELS and pol > 0) or (lb in POS_LABELS and pol < 0):
                    scores[tok] = abs(pol) * f
        label_kw[lb] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk])

    for lb in POS_LABELS + NEG_LABELS:
        label_kw.setdefault(lb, {})
    return label_kw

# =========================
# 6. 打标逻辑
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
# 7. UI
# =========================
st.title("🏷️ 评论自动打标系统（好评其他 / 差评其他）")

uploaded = st.file_uploader("上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

if uploaded:
    df = load_file(uploaded)

    rating_col = next(c for c in df.columns if "星" in c or "rating" in c.lower())
    text_col = next(c for c in df.columns if "内容" in c or "review" in c.lower())

    df["rating"] = df[rating_col].apply(parse_rating).round().astype(int)
    df = df[df["rating"].between(1, 5)]
    df["text"] = df[text_col].astype(str)

    polarity_weights = learn_polarity_weights(df["text"], df["rating"])
    label_kw = learn_label_kw(df, polarity_weights)

    df["Tag_Label"] = df.apply(lambda r: choose_tag(r["text"], r["rating"], label_kw), axis=1)

    # =========================
    # 可视化（原生）
    # =========================
    st.subheader("📊 评分分布")
    st.bar_chart(df["rating"].value_counts().sort_index())

    st.subheader("📊 标签分布")
    st.bar_chart(df["Tag_Label"].value_counts())

    st.subheader("预览（前 30 条）")
    st.dataframe(df[[rating_col, "Tag_Label", "text"]].head(30))

    st.download_button(
        "⬇️ 下载打标结果 CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "tagged_reviews_final.csv",
        "text/csv"
    )
