import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter, defaultdict

st.set_page_config(page_title="亚马逊评论自动打标（权重学习版/无模型）", page_icon="🏷️", layout="wide")

# =========================
# 1) 你的正式标签库（只输出这些标签）
# =========================
POS_LABELS = [
    "面料舒适","质量很好","有助于锻炼","有助于缓解疼痛","保暖","舒适贴合","有压缩感","抓握式有效","合身",
    "有助于关节炎/扳机指","增加手指灵活","促进血液循环","耐用","缓解不适","轻盈","覆盖整个手指","有助于防止肿胀"
]
NEG_LABELS = [
    "没有作用/没有效果","缝线裂开","二手商品","质量问题","不适合","尺码太小","尺码不对","接缝处不舒适","不耐用",
    "尺码太大","过敏","光滑/没有抓握","实物与购买数量不一致"
]

# 兜底（必须在库里）
POS_FALLBACK = "舒适贴合"
NEG_FALLBACK = "不适合"

# =========================
# 2) 标签“种子触发词”（用于弱监督分桶）
#    这些不是最终关键词库，程序会用数据学习并扩展权重
#    你可以后续继续加/改（越贴近你品类越准）
# =========================
SEEDS_POS = {
    "面料舒适": ["comfortable", "soft", "舒服", "柔软"],
    "质量很好": ["well made", "good quality", "质量", "做工好"],
    "有助于锻炼": ["workout", "exercise", "gym", "锻炼"],
    "有助于缓解疼痛": ["pain relief", "relief pain", "疼痛", "缓解"],
    "保暖": ["warm", "keep warm", "保暖"],
    "舒适贴合": ["fits well", "snug", "贴合", "合适"],
    "有压缩感": ["compression", "compressive", "压缩"],
    "抓握式有效": ["grip", "grippy", "抓握", "防滑"],
    "合身": ["perfect fit", "true to size", "合身", "刚好"],
    "有助于关节炎/扳机指": ["arthritis", "trigger finger", "关节炎", "扳机指"],
    "增加手指灵活": ["flexible", "dexterity", "灵活"],
    "促进血液循环": ["circulation", "blood flow", "血液循环"],
    "耐用": ["durable", "last long", "耐用"],
    "缓解不适": ["relieve", "help", "不适", "缓解"],
    "轻盈": ["lightweight", "light", "轻", "轻盈"],
    "覆盖整个手指": ["full finger", "full fingers", "覆盖", "全指"],
    "有助于防止肿胀": ["swelling", "prevent swelling", "肿胀", "防止肿胀"],
}

SEEDS_NEG = {
    "没有作用/没有效果": ["no effect", "doesn't work", "no help", "没用", "没有效果"],
    "缝线裂开": ["seam", "stitch", "split", "缝线", "开线"],
    "二手商品": ["used", "second hand", "二手", "用过"],
    "质量问题": ["poor quality", "cheap", "quality issue", "质量问题", "差"],
    "不适合": ["not fit", "not suitable", "不适合"],
    "尺码太小": ["too small", "runs small", "tight", "太小", "偏小"],
    "尺码不对": ["wrong size", "size not right", "尺码不对", "买错尺码"],
    "接缝处不舒适": ["seam hurts", "seam uncomfortable", "接缝", "磨", "硌"],
    "不耐用": ["not durable", "broke", "tear", "易破", "不耐用"],
    "尺码太大": ["too big", "runs large", "loose", "太大", "偏大"],
    "过敏": ["allergy", "rash", "red", "过敏", "红肿"],
    "光滑/没有抓握": ["slippery", "no grip", "滑", "没有抓握"],
    "实物与购买数量不一致": ["missing", "quantity", "not enough", "数量", "少了", "缺"],
}

# =========================
# 3) 文件读取 & 列自动识别
# =========================
def load_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan

def auto_match_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    for cand in candidates:
        cl = cand.lower()
        for col in cols:
            if cl in col.lower():
                return col
    return None

COLUMN_CANDIDATES = {
    "rating": ["星级","rating","Rating","score","Score","评分"],
    "title": ["标题","title","Title","headline","summary"],
    "content": ["内容(翻译)","内容（翻译）","翻译","translation","Translated","内容","content","Content","review","Review","评论内容","text","body"],
    "date": ["评论时间","date","Date","review_date","time","时间","评论日期"],
}

def build_text(row, col_title, col_text):
    t = str(row.get(col_text, "") or "")
    if col_title:
        h = str(row.get(col_title, "") or "")
        if h.strip():
            return f"{h.strip()} | {t.strip()}"
    return t.strip()

# =========================
# 4) Tokenize：英文词 + bigram + 中文2-gram（无需 jieba）
# =========================
def tokenize_mixed(text: str):
    if not text:
        return []
    s = text.lower()

    # 英文词
    eng = re.findall(r"[a-z]+", s)
    eng_bi = [f"{eng[i]} {eng[i+1]}" for i in range(len(eng)-1)]

    # 中文：提取连续中文并做2-gram
    zh_chunks = re.findall(r"[\u4e00-\u9fff]+", s)
    zh_tokens = []
    for chunk in zh_chunks:
        if len(chunk) == 1:
            zh_tokens.append(chunk)
        else:
            zh_tokens.extend([chunk[i:i+2] for i in range(len(chunk)-1)])

    return eng + eng_bi + zh_tokens

# =========================
# 5) 从数据学习：token 极性权重（1–3 vs 5）
# =========================
def learn_polarity_weights(texts, ratings, min_df=3):
    neg = Counter()
    pos = Counter()
    for txt, r in zip(texts, ratings):
        toks = tokenize_mixed(txt)
        if r <= 3:
            neg.update(toks)
        elif r == 5:
            pos.update(toks)

    vocab = set(neg) | set(pos)
    weights = {}
    for t in vocab:
        fn, fp = neg[t], pos[t]
        if fn + fp < min_df:
            continue
        # log-odds：>0 更像差评，<0 更像好评
        w = math.log((fn + 1) / (fp + 1))
        weights[t] = w
    return weights, neg, pos

# =========================
# 6) 弱监督分桶：用 seed 触发把部分评论分到各标签桶
# =========================
def weak_assign_bucket(text, seeds_map):
    s = text.lower()
    hit_labels = []
    for label, seeds in seeds_map.items():
        for kw in seeds:
            if kw.lower() in s:
                hit_labels.append(label)
                break
    return hit_labels

def learn_label_keyword_weights(df, polarity_weights, seeds_pos, seeds_neg, min_df_label=2, topk=40):
    """
    输出：label_kw[label][token] = weight
    weight = (token 在 label 桶的相对强度) * (token 的极性强度)
    """
    label_docs = defaultdict(list)  # label -> list of token lists

    for _, row in df.iterrows():
        r = int(row["rating"])
        text = row["text"]
        toks = tokenize_mixed(text)

        if r <= 3:
            hits = weak_assign_bucket(text, seeds_neg)
            for lb in hits:
                label_docs[lb].append(toks)
        elif r == 5:
            hits = weak_assign_bucket(text, seeds_pos)
            for lb in hits:
                label_docs[lb].append(toks)
        else:
            # 4星不参与学习，避免混杂（只用于推理）
            pass

    # 统计每个 label 桶内 token freq
    label_kw = {}
    for label, docs in label_docs.items():
        c = Counter()
        for toks in docs:
            c.update(toks)

        # 计算 token 权重（桶内相对 + 极性）
        total = sum(c.values()) + 1e-9
        token_scores = {}
        for t, f in c.items():
            if f < min_df_label:
                continue
            pol = polarity_weights.get(t, 0.0)
            # 桶内相对频率（避免全是高频虚词）
            rel = f / total
            # 最终权重：相对频率 * |polarity|，并对方向做一致性约束
            # 差评标签希望 pol>0，好评标签希望 pol<0
            if label in NEG_LABELS and pol <= 0:
                continue
            if label in POS_LABELS and pol >= 0:
                continue
            token_scores[t] = rel * abs(pol)

        # 只保留 TopK
        top = dict(sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:topk])
        label_kw[label] = top

    # 对于没有学到的 label（样本太少），给空字典（推理时靠 fallback）
    for lb in POS_LABELS:
        label_kw.setdefault(lb, {})
    for lb in NEG_LABELS:
        label_kw.setdefault(lb, {})

    return label_kw

# =========================
# 7) 推理：对每条评论按星级选 label（100%覆盖）
# =========================
def score_with_label_kw(toks, label_kw):
    s = 0.0
    for t in toks:
        if t in label_kw:
            s += label_kw[t]
    return s

def choose_label(row, label_kw, mode):
    """
    mode: 'neg_only' / 'pos_only' / 'four_star'
    """
    text = row["text"]
    toks = tokenize_mixed(text)

    if mode == "neg_only":
        best_lb, best_sc = None, -1e18
        for lb in NEG_LABELS:
            sc = score_with_label_kw(toks, label_kw.get(lb, {}))
            if sc > best_sc:
                best_lb, best_sc = lb, sc
        return best_lb if best_sc > 0 else NEG_FALLBACK

    if mode == "pos_only":
        best_lb, best_sc = None, -1e18
        for lb in POS_LABELS:
            sc = score_with_label_kw(toks, label_kw.get(lb, {}))
            if sc > best_sc:
                best_lb, best_sc = lb, sc
        return best_lb if best_sc > 0 else POS_FALLBACK

    # 4星：优先差评
    best_neg, sc_neg = None, -1e18
    for lb in NEG_LABELS:
        sc = score_with_label_kw(toks, label_kw.get(lb, {}))
        if sc > sc_neg:
            best_neg, sc_neg = lb, sc
    if sc_neg > 0:
        return best_neg

    best_pos, sc_pos = None, -1e18
    for lb in POS_LABELS:
        sc = score_with_label_kw(toks, label_kw.get(lb, {}))
        if sc > sc_pos:
            best_pos, sc_pos = lb, sc
    return best_pos if sc_pos > 0 else POS_FALLBACK

# =========================
# 8) UI
# =========================
st.title("🏷️ 评论自动打标（从Excel学习关键词权重 / 无模型 / 100%覆盖）")
st.caption("上传 → 系统自动学习「关键词→标签权重」→ 全量打标 → 下载结果（不需要复制粘贴任何东西）")

uploaded = st.file_uploader("上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

with st.expander("标签库（只读展示：输出只会使用这些标签）", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.write("✅ 好评标签库")
        st.write(POS_LABELS)
    with c2:
        st.write("❌ 差评标签库")
        st.write(NEG_LABELS)

if uploaded:
    df_raw = load_file(uploaded)

    cols = df_raw.columns.tolist()
    col_rating = auto_match_column(cols, COLUMN_CANDIDATES["rating"])
    col_title = auto_match_column(cols, COLUMN_CANDIDATES["title"])
    col_text = auto_match_column(cols, COLUMN_CANDIDATES["content"])
    col_date = auto_match_column(cols, COLUMN_CANDIDATES["date"])

    if not col_rating or not col_text:
        st.error("❌ 无法自动识别【星级】或【内容/翻译】列。请检查表头命名。")
        st.write({"rating": col_rating, "title": col_title, "text": col_text, "date": col_date})
        st.stop()

    df = df_raw.copy()
    df["rating_raw"] = df[col_rating]
    df["rating"] = df["rating_raw"].apply(parse_rating)
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].round().astype(int)
    df = df[df["rating"].between(1, 5)]

    df["text"] = df.apply(lambda r: build_text(r, col_title, col_text), axis=1)

    raw_total = len(df_raw)
    valid_total = len(df)
    invalid_total = raw_total - valid_total

    neg_rate = (df["rating"] <= 3).mean() * 100 if valid_total else 0.0
    severe_rate = (df["rating"] <= 2).mean() * 100 if valid_total else 0.0

    st.subheader("📊 自动看板")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("原始行数", raw_total)
    k2.metric("有效评分行数", valid_total)
    k3.metric("解析失败/无效行", invalid_total)
    k4.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")
    k5.metric("严重差评(≤2⭐)", f"{severe_rate:.1f}%")

    dist = df["rating"].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
    st.bar_chart(dist)

    st.markdown("---")
    st.subheader("🧠 Step A：从数据学习 token 极性权重（≤3 vs 5）")
    min_df = st.slider("极性学习：token最小出现次数(min_df)", 1, 10, 3, 1)
    polarity_weights, neg_counter, pos_counter = learn_polarity_weights(df["text"].tolist(), df["rating"].tolist(), min_df=min_df)
    st.success(f"已学习极性权重：{len(polarity_weights)} 个 token")

    st.subheader("🧩 Step B：弱监督分桶 + 学习「关键词→标签」权重")
    topk = st.slider("每个标签保留 TopK 关键词", 10, 120, 40, 5)
    min_df_label = st.slider("标签桶内：token最小出现次数", 1, 8, 2, 1)

    label_kw = learn_label_keyword_weights(
        df,
        polarity_weights=polarity_weights,
        seeds_pos=SEEDS_POS,
        seeds_neg=SEEDS_NEG,
        min_df_label=min_df_label,
        topk=topk
    )
    st.success("已学习标签关键词权重（用于全量打标）")

    with st.expander("查看：每个标签学到的 Top 关键词（可用于你迭代标签库/写PPT）", expanded=False):
        show_lb = st.selectbox("选择一个标签查看关键词权重", POS_LABELS + NEG_LABELS, index=0)
        kv = label_kw.get(show_lb, {})
        if not kv:
            st.info("该标签在数据中触发样本较少，目前学到的关键词较少；仍可用兜底/其它标签覆盖。")
        else:
            tmp = pd.DataFrame({"token": list(kv.keys()), "weight": list(kv.values())}).sort_values("weight", ascending=False)
            st.dataframe(tmp, use_container_width=True)

    st.markdown("---")
    st.subheader("🏷️ Step C：全量打标（100%覆盖，4星优先差评）")
    df["AI_Label"] = df.apply(
        lambda r: choose_label(
            r,
            label_kw=label_kw,
            mode="neg_only" if r["rating"] <= 3 else ("pos_only" if r["rating"] == 5 else "four_star")
        ),
        axis=1
    )

    # 校验：确保100%都在库里
    allowed = set(POS_LABELS + NEG_LABELS)
    bad = df[~df["AI_Label"].isin(allowed)]
    if len(bad) > 0:
        st.warning(f"发现 {len(bad)} 条标签不在库内（已自动回退兜底）")
        df.loc[~df["AI_Label"].isin(allowed), "AI_Label"] = np.where(df["rating"] <= 3, NEG_FALLBACK, POS_FALLBACK)

    st.subheader("预览（前 30 条）")
    st.dataframe(df[[col_rating, "rating", "AI_Label", "text"]].head(30), use_container_width=True)

    # 标签分布
    st.subheader("标签占比（Top 20）")
    lab_dist = df["AI_Label"].value_counts().head(20)
    st.bar_chart(lab_dist)

    st.markdown("---")
    st.subheader("⬇️ 导出")
    out_full = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载：全量打标结果 CSV（含AI_Label）", out_full, "tagged_reviews_weighted_labels.csv", "text/csv")

    # 额外导出：学习到的“关键词→标签权重表”
    rows = []
    for lb, kv in label_kw.items():
        for t, w in kv.items():
            rows.append({"label": lb, "token": t, "weight": w})
    kw_df = pd.DataFrame(rows).sort_values(["label", "weight"], ascending=[True, False])
    out_kw = kw_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载：学习到的关键词权重表（label-token-weight）", out_kw, "label_keyword_weights.csv", "text/csv")
