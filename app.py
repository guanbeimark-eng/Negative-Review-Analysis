import streamlit as st
import pandas as pd
import json
import uuid
import re
import numpy as np

# =========================
# 0) App Config + Login
# =========================
st.set_page_config(page_title="LLM 评论打标（傻瓜式）", page_icon="🏷️", layout="wide")

ACCESS_PASSWORD = "admin123"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    if st.session_state.get("password_input") == ACCESS_PASSWORD:
        st.session_state.logged_in = True
    else:
        st.error("密码错误")

if not st.session_state.logged_in:
    st.markdown("## 🔒 系统锁定")
    st.text_input("访问密码", type="password", key="password_input", on_change=check_password)
    st.stop()

# =========================
# 1) 内置评价库（默认）
#    你后面把这块替换成你们“文件评价库”的正式标签即可
# =========================
TAG_LIBRARY = {
    "positive": [
        "佩戴舒适", "支撑性好", "缓解关节不适", "尺寸合适", "质量好", "性价比高", "效果明显", "物流/发货快", "外观好看"
    ],
    "negative": [
        "尺码偏小", "尺码偏大", "尺码不一致", "不适合男士", "穿戴困难", "质量差", "与描述不符", "不舒适/勒手", "气味/异味", "耐用性差/易破", "压力/压缩感不足"
    ]
}

# =========================
# 2) Session State
# =========================
defaults = {
    "raw_df": None,
    "main_df": None,          # 清洗后主表（保留原字段 + rating_int + sys_id）
    "norm_df": None,          # id/rating/text （给模型用）
    "full_df": None,          # 主表+AI_Label 合并导出

    "col_map": None,
    "tag_config": {"pos": TAG_LIBRARY["positive"], "neg": TAG_LIBRARY["negative"], "all": TAG_LIBRARY["positive"] + TAG_LIBRARY["negative"]},
    "prompts": [],

    "step": 1,                # 导航步进：1-4
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# 3) Utils
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
    """兼容 rating: '4.0 out of 5 stars' / 'Rated 3' / '5' / 4.0"""
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except:
        return np.nan

COLUMN_CANDIDATES = {
    "rating": ["星级", "rating", "Rating", "评分", "Score"],
    "title": ["标题", "title", "Title", "headline", "summary"],
    "content": ["内容", "content", "Content", "review", "Review", "评论内容", "body", "text"],
    "translation": ["内容(翻译)", "翻译", "translation", "Translated", "内容（翻译）"],
    "date": ["评论时间", "date", "Date", "review_date", "time", "时间", "评论日期"],
    "id": ["review_id", "id", "ID", "评论ID", "uuid", "唯一ID"],
}

def auto_match_column(cols, candidates):
    # 精确
    for c in candidates:
        if c in cols:
            return c
    # 模糊包含
    for cand in candidates:
        cl = cand.lower()
        for col in cols:
            if cl in col.lower():
                return col
    return None

def auto_build_mapping(df):
    cols = df.columns.tolist()
    col_rating = auto_match_column(cols, COLUMN_CANDIDATES["rating"])
    col_title = auto_match_column(cols, COLUMN_CANDIDATES["title"])
    col_content = auto_match_column(cols, COLUMN_CANDIDATES["content"])
    col_trans = auto_match_column(cols, COLUMN_CANDIDATES["translation"])
    col_date = auto_match_column(cols, COLUMN_CANDIDATES["date"])
    col_id = auto_match_column(cols, COLUMN_CANDIDATES["id"])

    text_primary = col_trans or col_content
    return {
        "rating": col_rating,
        "title": col_title,
        "text": text_primary,
        "date": col_date,
        "id": col_id,
        "content_raw": col_content,
        "translation": col_trans
    }

def build_cleaned_frames(df_raw, m):
    tmp = df_raw.copy()

    # rating
    tmp["rating_numeric"] = tmp[m["rating"]].apply(parse_rating) if m.get("rating") else np.nan
    invalid_rating_cnt = int(tmp["rating_numeric"].isna().sum())

    valid = tmp.dropna(subset=["rating_numeric"]).copy()
    valid["rating_int"] = valid["rating_numeric"].round().astype(int)
    valid = valid[valid["rating_int"].between(1, 5)]

    # date
    time_ok = False
    if m.get("date") and m["date"] in valid.columns:
        valid["date_parsed"] = pd.to_datetime(valid[m["date"]], errors="coerce")
        time_ok = valid["date_parsed"].notna().sum() > 0

    # sys_id（优先用识别到的id列，否则生成）
    if m.get("id") and m["id"] in valid.columns:
        valid["sys_id"] = valid[m["id"]].astype(str)
    else:
        valid["sys_id"] = [str(uuid.uuid4())[:8] for _ in range(len(valid))]

    # text（title可选拼接）
    if m.get("text") is None:
        valid["__text__"] = ""
    else:
        if m.get("title") and m["title"] in valid.columns:
            valid["__text__"] = (
                valid[m["title"]].fillna("").astype(str).str.strip()
                + " | "
                + valid[m["text"]].fillna("").astype(str).str.strip()
            ).str.strip(" |")
        else:
            valid["__text__"] = valid[m["text"]].fillna("").astype(str)

    norm = valid[["sys_id", "rating_int", "__text__"]].rename(columns={
        "sys_id": "id",
        "rating_int": "rating",
        "__text__": "text"
    }).copy()

    return valid, norm, invalid_rating_cnt, time_ok

def safe_parse_json(text):
    """支持带 ```json```、以及多段 JSON list 粘贴"""
    if not text:
        return None
    clean = text.replace("```json", "").replace("```", "").strip()
    if not clean:
        return None
    try:
        return json.loads(clean)
    except:
        pass
    parts = [p.strip() for p in clean.split("\n\n") if p.strip()]
    merged = []
    ok = False
    for p in parts:
        try:
            obj = json.loads(p)
            if isinstance(obj, list):
                merged.extend(obj)
                ok = True
        except:
            continue
    return merged if ok else None

def extract_id_label_list(obj):
    """
    容错提取：
    - 标准：[{id,label}]
    - 变体：[{id,AI_Label}] / [{id,tag}] / [{id,Label}]
    """
    if not isinstance(obj, list):
        return None, "不是 list"
    if len(obj) == 0:
        return None, "空 list"

    # 找可能的label字段
    label_keys = ["label", "AI_Label", "ai_label", "tag", "Label", "标签", "分类"]
    out = []
    miss = 0

    for item in obj:
        if not isinstance(item, dict):
            miss += 1
            continue
        _id = item.get("id")
        if _id is None:
            miss += 1
            continue
        found = None
        for k in label_keys:
            if k in item:
                found = item.get(k)
                break
        if found is None:
            # 这里说明你粘贴的是 {id,rating,text} 这种，不含label
            out.append({"id": str(_id), "label": None})
        else:
            out.append({"id": str(_id), "label": "" if found is None else str(found).strip()})

    # 如果全部都没有label值（全是None），就判定“粘贴错了/模型没按格式输出”
    if all(x["label"] is None for x in out):
        return out, "缺少 label 字段（你粘贴的可能是评论数据而不是模型打标结果）"

    # 将 None 变成空串
    for x in out:
        if x["label"] is None:
            x["label"] = ""

    return out, None

def build_fix_prompt_from_bad_output(bad_json_text):
    """
    给用户一段“纠错提示词”：
    把模型输出转成正确格式
    """
    return f"""请把下面这段内容转换为严格 JSON list，仅保留每条的 id 和 label 两个字段：
- 输出格式必须是：[{{"id":"...","label":"..."}}]
- 不要输出任何解释文字，不要输出 ``` 包裹
- label 必须从我提供的标签库中选择；不匹配就输出空字符串 ""

原始内容如下：
{bad_json_text}
"""

# =========================
# 4) 顶部傻瓜式导航（不用点很多按钮）
# =========================
st.caption("流程：① 上传评论&自动映射  → ② 评价库（可选编辑） → ③ 生成 Prompt（4星优先差评） → ④ 粘贴JSON回填导出")
step = st.session_state.step

nav = st.columns(4)
if nav[0].button("1 上传&自动映射", use_container_width=True):
    st.session_state.step = 1
if nav[1].button("2 评价库", use_container_width=True):
    st.session_state.step = 2
if nav[2].button("3 生成Prompt", use_container_width=True):
    st.session_state.step = 3
if nav[3].button("4 回填&导出", use_container_width=True):
    st.session_state.step = 4

st.markdown("---")

# =========================
# Step 1：上传&自动映射（自动完成：清洗+看板+锁定）
# =========================
if st.session_state.step == 1:
    st.header("Step 1：上传评论文件（系统自动完成映射/清洗/看板）")

    uploaded = st.file_uploader("上传评论数据（CSV / Excel）", type=["csv", "xlsx"])
    if uploaded:
        df_raw = load_file(uploaded)
        st.session_state.raw_df = df_raw

        # 自动映射
        m = auto_build_mapping(df_raw)
        st.session_state.col_map = m

        # 必要列检查
        if not m.get("rating") or not m.get("text"):
            st.error("❌ 自动识别失败：缺少 rating 或 text 列。请换一个文件或把列名改成常见命名（如 星级 / 内容 / 内容(翻译)）。")
            st.json(m)
            st.stop()

        # 自动清洗并锁定（关键：不需要用户点按钮）
        valid, norm, invalid_cnt, time_ok = build_cleaned_frames(df_raw, m)
        st.session_state.main_df = valid
        st.session_state.norm_df = norm
        st.session_state.full_df = None  # 回填后才生成

        # 看板（自动展示）
        raw_total = len(df_raw)
        valid_total = len(valid)
        neg_cnt = int((valid["rating_int"] <= 3).sum())
        neg_rate = (neg_cnt / valid_total * 100) if valid_total else 0
        severe_cnt = int((valid["rating_int"] <= 2).sum())
        severe_rate = (severe_cnt / valid_total * 100) if valid_total else 0

        st.success(f"✅ 数据已准备就绪：原始 {raw_total} 行 / 有效评分 {valid_total} 行 / 解析失败 {invalid_cnt} 行")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("原始行数", raw_total)
        k2.metric("有效评分", valid_total)
        k3.metric("解析失败", invalid_cnt)
        k4.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")
        k5.metric("严重差评(≤2⭐)", f"{severe_rate:.1f}%")

        st.subheader("⭐ 星级分布")
        dist = valid["rating_int"].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
        st.bar_chart(dist)

        st.subheader("📝 LLM 输入预览（前5条）")
        st.dataframe(norm.head(5))

        with st.expander("查看系统自动识别的列映射"):
            st.json(m)

        st.info("下一步：点顶部『2 评价库』或『3 生成Prompt』继续。")

# =========================
# Step 2：评价库（可选编辑）
# =========================
if st.session_state.step == 2:
    st.header("Step 2：评价库（默认已内置，可选编辑）")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ 好评标签")
        pos_text = st.text_area("一行一个标签", value="\n".join(st.session_state.tag_config["pos"]), height=260)
    with c2:
        st.subheader("❌ 差评标签")
        neg_text = st.text_area("一行一个标签", value="\n".join(st.session_state.tag_config["neg"]), height=260)

    if st.button("保存评价库"):
        pos = [x.strip() for x in pos_text.splitlines() if x.strip()]
        neg = [x.strip() for x in neg_text.splitlines() if x.strip()]
        st.session_state.tag_config = {"pos": pos, "neg": neg, "all": pos + neg}
        st.success(f"✅ 已保存：好评 {len(pos)} 个 / 差评 {len(neg)} 个")

    st.info("下一步：点顶部『3 生成Prompt』继续。")

# =========================
# Step 3：生成 Prompt（只保留一个“复制”动作）
# =========================
if st.session_state.step == 3:
    st.header("Step 3：生成 Prompt（4星优先差评点）")

    if st.session_state.norm_df is None:
        st.warning("请先去 Step 1 上传评论数据。")
        st.stop()

    df = st.session_state.norm_df
    pos_tags = st.session_state.tag_config["pos"]
    neg_tags = st.session_state.tag_config["neg"]

    batch_size = st.slider("每批条数（越大越省事，但模型上下文要够）", 20, 120, 60, 10)

    def build_prompt(chunk, mode):
        pos_str = ", ".join([f'"{t}"' for t in pos_tags])
        neg_str = ", ".join([f'"{t}"' for t in neg_tags])

        # 关键：强制只输出 id,label
        system = (
            "You are an expert customer review tagger.\n"
            "You MUST choose labels ONLY from the provided tag libraries.\n"
            "Return STRICT JSON ONLY. No explanations. No extra text.\n"
            "Output schema MUST be: [{\"id\":\"...\",\"label\":\"...\"}]\n"
            "If no suitable tag, label must be empty string \"\".\n"
        )

        if mode == "1-3":
            task = f"""
These are 1-3 star reviews.
Choose label ONLY from NEGATIVE TAG LIBRARY.
NEGATIVE TAG LIBRARY: [{neg_str}]
"""
        elif mode == "5":
            task = f"""
These are 5 star reviews.
Choose label ONLY from POSITIVE TAG LIBRARY.
POSITIVE TAG LIBRARY: [{pos_str}]
"""
        else:
            task = f"""
These are 4 star reviews. PRIORITIZE complaints.
Rule:
1) If ANY complaint/negative point exists, choose from NEGATIVE TAG LIBRARY.
2) Otherwise choose from POSITIVE TAG LIBRARY.
3) If no suitable tag, output "".
NEGATIVE TAG LIBRARY: [{neg_str}]
POSITIVE TAG LIBRARY: [{pos_str}]
"""

        data = "DATA (JSON):\n" + json.dumps(chunk, ensure_ascii=False, indent=2)
        return f"{system}\n{task}\n{data}"

    # 自动生成批次（无需额外按钮；改变 batch_size 就会重算）
    prompts = []
    groups = {
        "1-3": df[df["rating"] <= 3],
        "4": df[df["rating"] == 4],
        "5": df[df["rating"] == 5],
    }
    for mode, gdf in groups.items():
        records = gdf.to_dict("records")
        for i in range(0, len(records), int(batch_size)):
            chunk = records[i:i+int(batch_size)]
            prompts.append({
                "title": f"[{mode}星] 批次 {i//int(batch_size)+1}（{len(chunk)}条）",
                "prompt": build_prompt(chunk, mode)
            })

    st.session_state.prompts = prompts
    st.success(f"✅ 已生成 {len(prompts)} 个 Prompt 批次（无需再点生成按钮）")

    for b in prompts[:6]:
        with st.expander(b["title"]):
            st.text_area("复制给模型（只需复制一次）", b["prompt"], height=280)

    if len(prompts) > 6:
        st.info(f"还有 {len(prompts)-6} 个批次未展开（为避免页面太长）。你可以在代码里改成全展开。")

    st.info("下一步：把模型返回的 JSON 粘贴到『4 回填&导出』。")

# =========================
# Step 4：回填（粘贴后自动判断、自动提示纠错）
# =========================
if st.session_state.step == 4:
    st.header("Step 4：粘贴模型 JSON → 自动回填 → 一键导出")

    if st.session_state.norm_df is None:
        st.warning("请先完成 Step 1。")
        st.stop()

    allowed_set = set(st.session_state.tag_config["all"])

    st.caption("你应该粘贴模型的返回结果：格式必须是 JSON list，例如："
               "[{\"id\":\"1bc3a5ae\",\"label\":\"尺码偏小\"}, ...]")

    json_text = st.text_area("粘贴 LLM 返回 JSON（可一次粘贴多批次）", height=240)

    # 这里保持一个按钮即可（不再让客户点很多按钮）
    if st.button("✅ 回填并更新导出文件", type="primary"):
        parsed = safe_parse_json(json_text)
        if parsed is None:
            st.error("JSON 解析失败：请确认粘贴的是合法 JSON（不要夹带解释文字）。")
            st.stop()

        extracted, err = extract_id_label_list(parsed)
        if err and "缺少 label" in err:
            st.error("❌ 你粘贴的不是『模型打标结果』，里面没有 label 字段。")
            st.info("你粘贴的看起来像『评论数据（id/rating/text）』而不是『打标结果（id/label）』。")

            st.subheader("✅ 复制下面这段纠错提示词，发给模型，让它把输出改成正确格式")
            fix_prompt = build_fix_prompt_from_bad_output(json_text)
            st.code(fix_prompt, language="text")
            st.stop()

        if extracted is None:
            st.error(f"无法提取 id/label：{err}")
            st.stop()

        # 校验 label 必须来自库
        for x in extracted:
            x["label"] = validate_label(x.get("label", ""), allowed_set)

        id_map = {x["id"]: x["label"] for x in extracted}

        # 回填到 normalized
        df = st.session_state.norm_df.copy()
        if "AI_Label" not in df.columns:
            df["AI_Label"] = ""
        df["AI_Label"] = df["id"].map(id_map).fillna(df["AI_Label"]).astype(str)
        st.session_state.norm_df = df

        st.success(f"✅ 回填完成：本次处理 {len(extracted)} 条（库外标签已自动置空）")
        st.dataframe(df.head(20))

        # 合并回 full_df
        if st.session_state.main_df is not None:
            main = st.session_state.main_df.copy()
            lab = df[["id", "AI_Label"]].copy()
            main["sys_id"] = main["sys_id"].astype(str)
            lab["id"] = lab["id"].astype(str)
            merged = main.merge(lab, left_on="sys_id", right_on="id", how="left")
            merged.drop(columns=["id"], inplace=True, errors="ignore")
            st.session_state.full_df = merged

    st.markdown("---")
    st.subheader("导出")

    out_norm = st.session_state.norm_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载：normalized（id/rating/text/AI_Label）", out_norm, "tagged_reviews_normalized.csv", "text/csv")

    if st.session_state.full_df is not None:
        out_full = st.session_state.full_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载：full（原始字段+AI_Label）", out_full, "tagged_reviews_full.csv", "text/csv")
