import streamlit as st
import pandas as pd
import json
import uuid
import re
import numpy as np

# ======================================================
# 0) 基础配置 & 登录
# ======================================================
st.set_page_config(
    page_title="Amazon 评论打标系统（自动映射+内置评价库）",
    page_icon="🏷️",
    layout="wide"
)

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

# ======================================================
# 1) 内置评价库（默认：你不需要上传）
#    你后续可以把这里替换成你“文件评价库”的正式标签
# ======================================================
TAG_LIBRARY = {
    "positive": [
        "佩戴舒适",
        "支撑性好",
        "缓解关节不适",
        "尺寸合适",
        "质量好",
        "性价比高",
        "效果明显",
        "物流/发货快",
        "外观好看"
    ],
    "negative": [
        "尺码偏小",
        "尺码偏大",
        "尺码不一致",
        "不适合男士",
        "穿戴困难",
        "质量差",
        "与描述不符",
        "不舒适/勒手",
        "气味/异味",
        "耐用性差/易破"
    ]
}

# ======================================================
# 2) Session State 初始化
# ======================================================
defaults = {
    "raw_df": None,
    "main_df": None,          # 清洗后主表（含rating_int、sys_uuid等）
    "norm_df": None,          # 归一化表（id/rating/text）
    "mapping_locked": False,

    "col_map": None,          # 自动识别到的列映射
    "tag_config": {
        "pos": TAG_LIBRARY["positive"],
        "neg": TAG_LIBRARY["negative"],
        "all": TAG_LIBRARY["positive"] + TAG_LIBRARY["negative"]
    },

    "generated_batches": [],
    "merged_full_df": None,   # 原始字段+AI_Label 合并后的导出表
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# 3) 工具函数
# ======================================================
def load_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def parse_rating(x):
    """兼容亚马逊常见rating格式：'4.0 out of 5 stars' / 'Rated 3' / '5' / 4.0"""
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

def safe_parse_json(text):
    """支持一次粘贴多段JSON（用空行分隔）"""
    if not text:
        return None
    clean = text.replace("```json", "").replace("```", "").strip()
    if not clean:
        return None

    # 先尝试整体解析
    try:
        return json.loads(clean)
    except:
        pass

    # 再尝试按空行拆分
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

def validate_label(label, allowed_set: set):
    if label is None:
        return ""
    lab = str(label).strip()
    return lab if lab in allowed_set else ""

# -------- 自动列映射：预设组合（不让用户点） --------
COLUMN_CANDIDATES = {
    "rating": ["星级", "rating", "Rating", "评分", "Score"],
    "title": ["标题", "title", "Title", "headline", "summary"],
    "content": ["内容", "content", "Content", "review", "Review", "评论内容", "body", "text"],
    "translation": ["内容(翻译)", "翻译", "translation", "Translated", "内容（翻译）"],
    "date": ["评论时间", "date", "Date", "review_date", "time", "时间", "评论日期"],
    "id": ["review_id", "id", "ID", "评论ID", "uuid", "唯一ID"],
}

def auto_match_column(cols, candidates):
    # 1) 先精确匹配
    for c in candidates:
        if c in cols:
            return c
    # 2) 再模糊包含匹配（列名里包含关键词）
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower()
        for col in cols:
            if cand_l in col.lower():
                return col
    return None

def auto_build_mapping(df: pd.DataFrame):
    cols = df.columns.tolist()
    col_rating = auto_match_column(cols, COLUMN_CANDIDATES["rating"])
    col_title = auto_match_column(cols, COLUMN_CANDIDATES["title"])
    col_content = auto_match_column(cols, COLUMN_CANDIDATES["content"])
    col_trans = auto_match_column(cols, COLUMN_CANDIDATES["translation"])
    col_date = auto_match_column(cols, COLUMN_CANDIDATES["date"])
    col_id = auto_match_column(cols, COLUMN_CANDIDATES["id"])

    # 内容优先级：翻译列 > 内容列
    text_primary = col_trans or col_content

    return {
        "rating": col_rating,
        "title": col_title,         # 可空
        "text": text_primary,       # 必须
        "content_raw": col_content, # 可空（用于排查）
        "translation": col_trans,   # 可空
        "date": col_date,           # 可空
        "id": col_id                # 可空（可自动生成）
    }

# ======================================================
# 4) 页面
# ======================================================
st.title("🏷️ Amazon 评论打标系统（自动列映射 + 内置评价库）")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ 上传评论 & 自动映射",
    "2️⃣ 内置评价库（可选编辑）",
    "3️⃣ 生成 Prompt（4星优先差评）",
    "4️⃣ 回填 & 导出"
])

# ======================================================
# Tab 1：上传评论 & 自动映射（核心改动：不让用户点）
# ======================================================
with tab1:
    st.header("Step 1：上传评论文件 → 系统自动识别列组合 → 一键锁定")

    uploaded = st.file_uploader("上传评论数据（CSV / Excel）", type=["csv", "xlsx"])

    if uploaded:
        df_raw = load_file(uploaded)
        st.session_state.raw_df = df_raw

        st.success(f"✅ 评论数据加载成功：原始行数 = {len(df_raw)}")
        st.dataframe(df_raw.head(8))

        # 自动映射
        auto_map = auto_build_mapping(df_raw)
        st.session_state.col_map = auto_map

        st.markdown("### 🔍 系统自动识别到的列映射（默认不需要手动点）")
        st.json(auto_map)

        # 检查是否满足最低要求
        missing_critical = []
        if not auto_map["rating"]:
            missing_critical.append("rating（星级列）")
        if not auto_map["text"]:
            missing_critical.append("text（内容/翻译列）")

        if missing_critical:
            st.error("❌ 自动识别失败：缺少关键列：" + "、".join(missing_critical))
            st.info("请在下方【高级设置】里手动指定（仅在识别失败时需要）。")
            with st.expander("高级设置：手动修正列映射（仅识别失败时用）", expanded=True):
                cols = df_raw.columns.tolist()
                col_rating = st.selectbox("手动选择星级列", cols, index=0)
                col_text = st.selectbox("手动选择内容列（建议选 内容(翻译) 优先）", cols, index=0)
                col_title = st.selectbox("手动选择标题列（可选）", ["--不使用--"] + cols, index=0)
                col_date = st.selectbox("手动选择时间列（可选）", ["--不使用--"] + cols, index=0)
                col_id = st.selectbox("手动选择ID列（可选）", ["--自动生成UUID--"] + cols, index=0)

                # 覆盖auto_map
                auto_map["rating"] = col_rating
                auto_map["text"] = col_text
                auto_map["title"] = None if col_title == "--不使用--" else col_title
                auto_map["date"] = None if col_date == "--不使用--" else col_date
                auto_map["id"] = None if col_id == "--自动生成UUID--" else col_id
                st.session_state.col_map = auto_map
                st.warning("已用手动设置覆盖自动识别。请继续预览/锁定。")

        st.markdown("---")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            preview = st.button("🔍 预览清洗效果", disabled=st.session_state.mapping_locked)
        with c2:
            lock = st.button("✅ 锁定映射并生成看板", type="primary",
                             disabled=(st.session_state.mapping_locked))
        with c3:
            unlock = st.button("♻️ 解除锁定", disabled=(not st.session_state.mapping_locked))

        if unlock:
            st.session_state.mapping_locked = False
            st.session_state.main_df = None
            st.session_state.norm_df = None
            st.session_state.merged_full_df = None
            st.success("已解除锁定，可重新预览/锁定。")

        def build_cleaned_frames(df_in: pd.DataFrame, m: dict):
            tmp = df_in.copy()

            # rating 解析
            tmp["rating_numeric"] = tmp[m["rating"]].apply(parse_rating)
            invalid_rating_cnt = int(tmp["rating_numeric"].isna().sum())

            valid = tmp.dropna(subset=["rating_numeric"]).copy()
            valid["rating_int"] = valid["rating_numeric"].round().astype(int)
            valid = valid[valid["rating_int"].between(1, 5)]

            # date（可选）
            time_parse_success = False
            if m.get("date"):
                valid["date_parsed"] = pd.to_datetime(valid[m["date"]], errors="coerce")
                time_parse_success = valid["date_parsed"].notna().sum() > 0

            # id：优先用识别到的id列，否则自动uuid
            if m.get("id") and m["id"] in valid.columns:
                valid["sys_id"] = valid[m["id"]].astype(str)
                id_col = "sys_id"
            else:
                valid["sys_id"] = [str(uuid.uuid4())[:8] for _ in range(len(valid))]
                id_col = "sys_id"

            # text：title可选拼接
            title_col = m.get("title")
            text_col = m.get("text")
            if title_col and title_col in valid.columns:
                valid["__text__"] = (
                    valid[title_col].fillna("").astype(str).str.strip()
                    + " | "
                    + valid[text_col].fillna("").astype(str).str.strip()
                ).str.strip(" |")
            else:
                valid["__text__"] = valid[text_col].fillna("").astype(str)

            norm = valid[[id_col, "rating_int", "__text__"]].rename(columns={
                id_col: "id",
                "rating_int": "rating",
                "__text__": "text"
            }).copy()

            return valid, norm, invalid_rating_cnt, time_parse_success

        if preview:
            m = st.session_state.col_map
            valid, norm, invalid_cnt, time_ok = build_cleaned_frames(df_raw, m)

            # 指标
            raw_total = len(df_raw)
            valid_total = len(valid)
            neg_cnt = int((valid["rating_int"] <= 3).sum())
            neg_rate = (neg_cnt / valid_total * 100) if valid_total else 0
            severe_cnt = int((valid["rating_int"] <= 2).sum())
            severe_rate = (severe_cnt / valid_total * 100) if valid_total else 0

            st.subheader("📊 预览看板（未锁定）")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("原始行数", raw_total)
            k2.metric("有效评分行数", valid_total)
            k3.metric("评分解析失败", invalid_cnt)
            k4.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")
            k5.metric("严重差评(≤2⭐)", f"{severe_rate:.1f}%")

            st.markdown("### ⭐ 星级分布（1–5）")
            dist = valid["rating_int"].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
            st.bar_chart(dist)

            st.markdown("### 📝 LLM 输入文本预览（前 5 条）")
            st.dataframe(norm.head(5))

            if time_ok:
                st.markdown("### 📈 评论时间趋势（月度）")
                ts = valid.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M").size()
                st.line_chart(ts)
            else:
                st.info("未识别到时间列或时间解析失败：已跳过趋势分析。")

        if lock:
            m = st.session_state.col_map
            valid, norm, invalid_cnt, _ = build_cleaned_frames(df_raw, m)
            st.session_state.main_df = valid
            st.session_state.norm_df = norm
            st.session_state.mapping_locked = True
            st.success("✅ 已锁定映射并生成标准数据，可进入 Step 2/3。")

# ======================================================
# Tab 2：内置评价库（可选编辑）
# ======================================================
with tab2:
    st.header("Step 2：内置评价库（默认已加载，不需要上传）")
    st.info("你要用“文件评价库”的正式标签时，把它们替换到这里，或者在下方直接编辑。")

    pos = st.session_state.tag_config["pos"]
    neg = st.session_state.tag_config["neg"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ 好评标签（Positive）")
        pos_text = st.text_area("一行一个标签", value="\n".join(pos), height=260)
    with c2:
        st.subheader("❌ 差评标签（Negative）")
        neg_text = st.text_area("一行一个标签", value="\n".join(neg), height=260)

    if st.button("保存评价库修改"):
        pos_new = [x.strip() for x in pos_text.splitlines() if x.strip()]
        neg_new = [x.strip() for x in neg_text.splitlines() if x.strip()]
        st.session_state.tag_config = {
            "pos": pos_new,
            "neg": neg_new,
            "all": pos_new + neg_new
        }
        st.success(f"✅ 已保存：好评 {len(pos_new)} 个，差评 {len(neg_new)} 个")

# ======================================================
# Tab 3：Prompt 生成（4星优先差评）
# ======================================================
with tab3:
    st.header("Step 3：生成 Prompt（4星优先差评点）")

    if not st.session_state.mapping_locked or st.session_state.norm_df is None:
        st.warning("请先在 Step 1 锁定映射并生成标准数据。")
        st.stop()

    if not st.session_state.tag_config["pos"] or not st.session_state.tag_config["neg"]:
        st.warning("评价库为空：请在 Step 2 先配置好评/差评标签。")
        st.stop()

    batch_size = st.number_input("每批条数", value=40, min_value=10, max_value=200, step=10)

    def build_prompt(chunk, mode, pos_tags, neg_tags):
        pos_str = ", ".join([f'"{t}"' for t in pos_tags])
        neg_str = ", ".join([f'"{t}"' for t in neg_tags])

        system = (
            "You are an expert customer review tagger.\n"
            "You MUST choose labels ONLY from the provided tag libraries.\n"
            "Return STRICT JSON only. No explanations. No extra text.\n"
            "Output schema: [{\"id\":\"...\",\"label\":\"\"}].\n"
            "If no suitable tag, label must be empty string \"\".\n"
        )

        if mode == "1-3":
            task = f"""
TASK:
These are 1-3 star reviews.
You MUST choose from NEGATIVE TAG LIBRARY only.

NEGATIVE TAG LIBRARY:
[{neg_str}]
"""
        elif mode == "5":
            task = f"""
TASK:
These are 5 star reviews.
You MUST choose from POSITIVE TAG LIBRARY only.

POSITIVE TAG LIBRARY:
[{pos_str}]
"""
        else:  # 4-star
            task = f"""
TASK:
These are 4 star reviews. PRIORITIZE complaints.
Rule:
1) If review contains ANY complaint/negative point, choose from NEGATIVE TAG LIBRARY.
2) Otherwise choose from POSITIVE TAG LIBRARY.
3) If still no suitable tag, output "".

NEGATIVE TAG LIBRARY:
[{neg_str}]

POSITIVE TAG LIBRARY:
[{pos_str}]
"""
        data = "DATA (JSON):\n" + json.dumps(chunk, ensure_ascii=False, indent=2)
        return f"{system}\n{task}\n{data}"

    if st.button("生成 Prompt"):
        df = st.session_state.norm_df
        pos_tags = st.session_state.tag_config["pos"]
        neg_tags = st.session_state.tag_config["neg"]

        groups = {
            "1-3": df[df["rating"] <= 3],
            "4": df[df["rating"] == 4],
            "5": df[df["rating"] == 5],
        }

        batches = []
        for mode, gdf in groups.items():
            if gdf.empty:
                continue
            records = gdf.to_dict("records")
            for i in range(0, len(records), int(batch_size)):
                chunk = records[i:i+int(batch_size)]
                batches.append({
                    "title": f"[{mode}星] 批次 {i//int(batch_size)+1}（{len(chunk)}条）",
                    "prompt": build_prompt(chunk, mode, pos_tags, neg_tags)
                })

        st.session_state.generated_batches = batches
        st.success(f"✅ 已生成 {len(batches)} 个任务包")

    for b in st.session_state.generated_batches:
        with st.expander(b["title"]):
            st.text_area("Prompt（复制给模型）", b["prompt"], height=290)
            st.caption("务必让模型只输出 JSON（不带解释），否则回填会解析失败。")

# ======================================================
# Tab 4：回填 & 导出
# ======================================================
with tab4:
    st.header("Step 4：粘贴模型 JSON → 回填 → 导出")

    if st.session_state.norm_df is None:
        st.warning("请先完成 Step 1 / Step 3。")
        st.stop()

    allowed_set = set(st.session_state.tag_config["all"])

    json_text = st.text_area("粘贴 LLM 返回 JSON（可一次粘贴多批次）", height=220)

    if st.button("合并结果"):
        data = safe_parse_json(json_text)
        if not data or not isinstance(data, list):
            st.error("JSON 解析失败：请确保返回格式为 list，例如：[{'id':'xxx','label':'...'}]")
        else:
            res_df = pd.DataFrame(data)
            if "id" not in res_df.columns or "label" not in res_df.columns:
                st.error("返回 JSON 必须包含 id 和 label 字段。")
            else:
                res_df["id"] = res_df["id"].astype(str)
                res_df["label"] = res_df["label"].apply(lambda x: validate_label(x, allowed_set))

                id_map = dict(zip(res_df["id"], res_df["label"]))

                df = st.session_state.norm_df.copy()
                if "AI_Label" not in df.columns:
                    df["AI_Label"] = ""

                df["AI_Label"] = df["id"].map(id_map).fillna(df["AI_Label"]).astype(str)
                st.session_state.norm_df = df

                st.success(f"✅ 合并完成：本次合并 {len(res_df)} 条（库外标签已自动置空）")
                st.dataframe(df.head(20))

                # 合并回主表（原字段+AI_Label）
                if st.session_state.main_df is not None:
                    main = st.session_state.main_df.copy()
                    lab = df[["id", "AI_Label"]].copy()
                    # main里sys_id对应norm_df的id
                    if "sys_id" in main.columns:
                        main["sys_id"] = main["sys_id"].astype(str)
                        lab["id"] = lab["id"].astype(str)
                        merged = main.merge(lab, left_on="sys_id", right_on="id", how="left")
                        merged.drop(columns=["id"], inplace=True, errors="ignore")
                        st.session_state.merged_full_df = merged

    st.markdown("---")
    st.subheader("导出")

    out1 = st.session_state.norm_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载：normalized（id/rating/text/AI_Label）", out1, "tagged_reviews_normalized.csv", "text/csv")

    if st.session_state.merged_full_df is not None:
        out2 = st.session_state.merged_full_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载：full（原始字段 + AI_Label）", out2, "tagged_reviews_full.csv", "text/csv")
