import streamlit as st
import pandas as pd
import json
import uuid

# ======================================================
# 0. 页面配置 & 登录
# ======================================================
st.set_page_config(
    page_title="LLM 评论打标系统（可视化列映射版）",
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
# 1. Session State
# ======================================================
for k, v in {
    "raw_df": None,
    "preview_df": None,
    "main_df": None,
    "normalized_df": None,
    "id_col_in_main": None,
    "tag_config": {"pos": [], "neg": [], "all": []},
    "generated_batches": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# 2. 工具函数
# ======================================================
def load_file(f):
    if f.name.endswith(".csv"):
        try:
            return pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(f, encoding="gbk")
    return pd.read_excel(f)

def normalize_polarity(x):
    s = str(x).lower()
    if any(k in s for k in ["pos", "good", "好", "正"]):
        return "positive"
    if any(k in s for k in ["neg", "bad", "差", "负"]):
        return "negative"
    return ""

def safe_parse_json(text):
    if not text:
        return None
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except:
        return None

# ======================================================
# 3. 页面结构
# ======================================================
st.title("🏷️ 评论打标系统（列映射可视化）")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ 数据列映射（可视化）",
    "2️⃣ 评价库配置",
    "3️⃣ Prompt 生成",
    "4️⃣ 结果回填 & 导出"
])

# ======================================================
# Tab 1：数据列映射（重点）
# ======================================================
with tab1:
    st.header("Step 1：数据列映射（选择 → 预览 → 确认）")

    uploaded = st.file_uploader("上传评论数据（CSV / Excel）", type=["csv", "xlsx"])
    if uploaded:
        df = load_file(uploaded)
        st.session_state.raw_df = df
        st.dataframe(df.head())

        cols = df.columns.tolist()
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1: col_rating = st.selectbox("⭐ 星级列", cols)
        with c2: col_title = st.selectbox("📝 标题列（可选）", ["--不使用--"] + cols)
        with c3: col_content = st.selectbox("📄 内容列", cols)
        with c4: col_id = st.selectbox("🆔 ID 列", ["-- 自动生成 --"] + cols)
        with c5: col_date = st.selectbox("📅 时间列（可选）", ["--不使用--"] + cols)

        # ---------- 预览 ----------
        if st.button("🔍 预览列映射效果"):
            tmp = df.copy()

            # rating
            tmp["rating"] = pd.to_numeric(tmp[col_rating], errors="coerce").round()
            tmp = tmp[tmp["rating"].between(1, 5)]

            # id
            if col_id == "-- 自动生成 --":
                tmp["id"] = [str(uuid.uuid4())[:8] for _ in range(len(tmp))]
            else:
                tmp["id"] = tmp[col_id].astype(str)

            # text
            if col_title != "--不使用--":
                tmp["text"] = (
                    tmp[col_title].fillna("").astype(str) + " | " +
                    tmp[col_content].fillna("").astype(str)
                )
            else:
                tmp["text"] = tmp[col_content].astype(str)

            preview = tmp[["id", "rating", "text"]].copy()
            st.session_state.preview_df = preview

            st.subheader("⭐ 星级解析预览")
            st.dataframe(preview[["rating"]].head())

            st.subheader("🆔 ID 安全性检查")
            c1, c2, c3 = st.columns(3)
            c1.metric("是否唯一", "✅" if preview["id"].is_unique else "❌")
            c2.metric("空值数", int(preview["id"].isna().sum()))
            c3.metric("数据量", len(preview))

            st.subheader("📝 LLM 输入文本预览")
            st.dataframe(preview.head(5))

        # ---------- 确认 ----------
        if st.session_state.preview_df is not None:
            st.markdown("---")
            if st.button("✅ 确认映射并锁定", type="primary"):
                st.session_state.main_df = df.copy()
                st.session_state.normalized_df = st.session_state.preview_df.copy()
                st.session_state.id_col_in_main = "id"
                st.success("映射已锁定，可以进入下一步")

# ======================================================
# Tab 2：评价库
# ======================================================
with tab2:
    st.header("Step 2：评价库配置")

    tag_file = st.file_uploader("上传评价库（需 label / polarity）", type=["csv", "xlsx"])
    if tag_file:
        tag_df = load_file(tag_file)
        st.dataframe(tag_df.head())

        lbl = st.selectbox("标签列", tag_df.columns)
        pol = st.selectbox("极性列", tag_df.columns)

        if st.button("加载评价库"):
            tag_df["pol"] = tag_df[pol].apply(normalize_polarity)
            pos = tag_df[tag_df["pol"] == "positive"][lbl].astype(str).tolist()
            neg = tag_df[tag_df["pol"] == "negative"][lbl].astype(str).tolist()
            st.session_state.tag_config = {
                "pos": pos,
                "neg": neg,
                "all": list(set(pos + neg))
            }
            st.success(f"已加载：好评 {len(pos)} / 差评 {len(neg)}")

# ======================================================
# Tab 3：Prompt 生成
# ======================================================
with tab3:
    st.header("Step 3：生成 Prompt（4 星优先差评）")

    if st.session_state.normalized_df is None:
        st.warning("请先完成列映射")
        st.stop()

    if not st.session_state.tag_config["all"]:
        st.warning("请先加载评价库")
        st.stop()

    batch = st.number_input("每批条数", 10, 200, 30)

    def build_prompt(chunk, mode):
        pos = ", ".join(f'"{x}"' for x in st.session_state.tag_config["pos"])
        neg = ", ".join(f'"{x}"' for x in st.session_state.tag_config["neg"])

        rule = {
            "1-3": f"只允许从差评标签中选择：[{neg}]",
            "5": f"只允许从好评标签中选择：[{pos}]",
            "4": f"若有任何抱怨 → 差评标签 [{neg}]，否则好评标签 [{pos}]"
        }[mode]

        return f"""
You are a review tagger.
Rules:
- Output ONLY JSON: [{{"id":"...","label":""}}]
- 标签必须来自给定标签库
- 不匹配输出 ""

{rule}

DATA:
{json.dumps(chunk, ensure_ascii=False, indent=2)}
"""

    if st.button("生成 Prompt"):
        df = st.session_state.normalized_df
        batches = []

        for mode, g in {
            "1-3": df[df["rating"] <= 3],
            "4": df[df["rating"] == 4],
            "5": df[df["rating"] == 5],
        }.items():
            rows = g.to_dict("records")
            for i in range(0, len(rows), batch):
                batches.append(build_prompt(rows[i:i+batch], mode))

        st.session_state.generated_batches = batches

    for i, p in enumerate(st.session_state.generated_batches):
        with st.expander(f"Prompt 批次 {i+1}"):
            st.text_area("复制给模型", p, height=260)

# ======================================================
# Tab 4：结果回填
# ======================================================
with tab4:
    st.header("Step 4：回填 & 导出")

    result = st.text_area("粘贴 LLM 返回 JSON", height=200)

    if st.button("合并结果"):
        data = safe_parse_json(result)
        if not data:
            st.error("JSON 解析失败")
        else:
            res = pd.DataFrame(data)
            allowed = set(st.session_state.tag_config["all"])
            res["label"] = res["label"].apply(lambda x: x if x in allowed else "")

            df = st.session_state.normalized_df.copy()
            df["AI_Label"] = df["id"].map(dict(zip(res["id"], res["label"])))
            st.session_state.normalized_df = df
            st.success("合并完成")
            st.dataframe(df.head())

    if st.session_state.normalized_df is not None:
        csv = st.session_state.normalized_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载 CSV", csv, "tagged_reviews.csv")
