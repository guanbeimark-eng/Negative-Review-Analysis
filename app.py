import streamlit as st
import pandas as pd
import json
import uuid
import re
import numpy as np

# ======================================================
# 0. 页面配置 & 登录
# ======================================================
st.set_page_config(
    page_title="LLM 评论打标系统（可视化列映射 + 差评占比修复版）",
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
defaults = {
    "raw_df": None,
    "preview_main_df": None,     # 预览态（主表清洗后，含rating_int等）
    "preview_norm_df": None,     # 预览态（归一化后 id/rating/text）
    "main_df": None,             # 确认态主表
    "normalized_df": None,       # 确认态归一化表
    "id_col_in_main": None,      # 主表里真实ID列名（sys_uuid or 用户选列）
    "mapping_locked": False,

    "tag_config": {"pos": [], "neg": [], "all": []},
    "generated_batches": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# 2. 工具函数
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
    """
    兼容：
    - '4.0 out of 5 stars'
    - 'Rated 3 out of 5'
    - '5'
    - 4.0
    """
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

def normalize_polarity(x):
    s = str(x).strip().lower()
    if any(k in s for k in ["positive", "pos", "good", "好", "正"]):
        return "positive"
    if any(k in s for k in ["negative", "neg", "bad", "差", "负"]):
        return "negative"
    return ""

def safe_parse_json(text):
    if not text:
        return None
    clean = text.replace("```json", "").replace("```", "").strip()
    if not clean:
        return None
    # 先整体解析
    try:
        return json.loads(clean)
    except:
        pass
    # 多段粘贴：按空行切开尝试合并
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

# ======================================================
# 3. 页面结构
# ======================================================
st.title("🏷️ 评论数据打标系统（列映射可视化 + 差评占比修复版）")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ 数据列映射（可视化）",
    "2️⃣ 评价库配置",
    "3️⃣ Prompt 生成（4星优先差评）",
    "4️⃣ 回填 & 导出"
])

# ======================================================
# Tab 1：数据列映射（选择 → 预览 → 确认锁定）
# ======================================================
with tab1:
    st.header("Step 1：数据导入 & 列映射（按钮化 + 可视化）")

    uploaded = st.file_uploader("上传评论数据（CSV / Excel）", type=["csv", "xlsx"])

    if uploaded:
        df_raw = load_file(uploaded)
        st.session_state.raw_df = df_raw

        st.success(f"数据加载成功：共包含 {len(df_raw)} 行（原始行数）")
        st.dataframe(df_raw.head(5))

        cols = df_raw.columns.tolist()

        st.markdown("---")
        st.subheader("🔧 选择关键字段（先选，后预览，再确认）")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            col_rating = st.selectbox("⭐ 星级列 (rating)", cols, disabled=st.session_state.mapping_locked)
        with c2:
            col_title = st.selectbox("📝 标题列（可选）", ["--不使用--"] + cols, disabled=st.session_state.mapping_locked)
        with c3:
            col_content = st.selectbox("📄 内容列 (content)", cols, disabled=st.session_state.mapping_locked)
        with c4:
            col_id = st.selectbox("🆔 唯一ID列", ["-- 自动生成 UUID --"] + cols, disabled=st.session_state.mapping_locked)
        with c5:
            col_date = st.selectbox("📅 时间列（可选）", ["--不使用--"] + cols, disabled=st.session_state.mapping_locked)

        st.markdown("---")
        c_btn1, c_btn2, c_btn3 = st.columns([1,1,2])

        with c_btn1:
            preview_clicked = st.button("🔍 预览映射效果", disabled=st.session_state.mapping_locked)
        with c_btn2:
            confirm_clicked = st.button("✅ 确认并锁定映射", type="primary", disabled=(st.session_state.preview_norm_df is None or st.session_state.mapping_locked))
        with c_btn3:
            reset_clicked = st.button("♻️ 解除锁定/重新映射", disabled=not st.session_state.mapping_locked)

        if reset_clicked:
            st.session_state.mapping_locked = False
            st.session_state.preview_main_df = None
            st.session_state.preview_norm_df = None
            st.session_state.main_df = None
            st.session_state.normalized_df = None
            st.session_state.id_col_in_main = None
            st.success("已解除锁定，可以重新选择列并预览。")

        if preview_clicked:
            tmp = df_raw.copy()

            # ---------- 1) rating 解析（关键修复） ----------
            tmp["rating_numeric"] = tmp[col_rating].apply(parse_rating)
            invalid_rating_cnt = int(tmp["rating_numeric"].isna().sum())

            # 先保留统计信息，再过滤
            valid = tmp.dropna(subset=["rating_numeric"]).copy()
            valid["rating_int"] = valid["rating_numeric"].round().astype(int)
            valid = valid[valid["rating_int"].between(1, 5)]

            # ---------- 2) 时间解析（可选） ----------
            time_parse_success = False
            if col_date != "--不使用--":
                valid["date_parsed"] = pd.to_datetime(valid[col_date], errors="coerce")
                time_parse_success = valid["date_parsed"].notna().sum() > 0

            # ---------- 3) ID 处理 ----------
            if col_id.startswith("--"):
                valid["sys_uuid"] = [str(uuid.uuid4())[:8] for _ in range(len(valid))]
                id_col_in_main = "sys_uuid"
            else:
                valid[col_id] = valid[col_id].astype(str)
                id_col_in_main = col_id

            # ---------- 4) text 拼接（标题可选） ----------
            if col_title != "--不使用--":
                valid["__text_joined__"] = (
                    valid[col_title].fillna("").astype(str).str.strip()
                    + " | "
                    + valid[col_content].fillna("").astype(str).str.strip()
                ).str.strip(" |")
                text_col = "__text_joined__"
            else:
                text_col = col_content

            # 预览归一化表（给 LLM 的输入）
            norm = valid[[id_col_in_main, "rating_int", text_col]].rename(columns={
                id_col_in_main: "id",
                "rating_int": "rating",
                text_col: "text"
            }).copy()

            # 保存到 session（预览态）
            st.session_state.preview_main_df = valid
            st.session_state.preview_norm_df = norm
            st.session_state.id_col_in_main = id_col_in_main

            # ========== 可视化预览区 ==========
            st.subheader("✅ 预览结果（确认无误再点“锁定映射”）")

            raw_total = len(df_raw)
            valid_total = len(valid)

            neg_cnt = int((valid["rating_int"] <= 3).sum())
            neg_rate = (neg_cnt / valid_total * 100) if valid_total else 0

            severe_neg_cnt = int((valid["rating_int"] <= 2).sum())
            severe_neg_rate = (severe_neg_cnt / valid_total * 100) if valid_total else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("原始行数", raw_total)
            m2.metric("有效评分行数", valid_total)
            m3.metric("评分解析失败行数", invalid_rating_cnt)
            m4.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")
            m5.metric("严重差评(≤2⭐)", f"{severe_neg_rate:.1f}%")

            st.markdown("### ⭐ 星级分布（1–5）")
            dist = valid["rating_int"].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
            st.bar_chart(dist)

            st.markdown("### 🆔 ID 安全性检查")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("ID是否唯一", "✅" if norm["id"].is_unique else "❌")
            cc2.metric("ID空值数", int(norm["id"].isna().sum()))
            cc3.metric("示例ID", str(norm["id"].iloc[0]) if len(norm) else "N/A")

            if not norm["id"].is_unique:
                st.error("⚠️ 你选择的ID列不唯一，会导致回填错乱。请换一个唯一列或使用自动UUID。")

            st.markdown("### 📝 送入 LLM 的文本预览（前 5 条）")
            st.dataframe(norm.head(5))

            if time_parse_success:
                st.markdown("### 📈 评论时间趋势（月度）")
                try:
                    ts = valid.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M").size()
                    st.line_chart(ts)
                except Exception:
                    st.info("时间趋势绘制失败（但不影响其它功能）")
            else:
                st.info("未选择时间列或时间解析失败：已跳过趋势分析。")

        if confirm_clicked:
            # 锁定：将预览态写入正式态
            st.session_state.main_df = st.session_state.preview_main_df.copy()
            st.session_state.normalized_df = st.session_state.preview_norm_df.copy()
            st.session_state.mapping_locked = True
            st.success("✅ 已锁定列映射。现在可以进入 Step 2 配置评价库。")

# ======================================================
# Tab 2：评价库配置
# ======================================================
with tab2:
    st.header("Step 2：导入评价库标签（label + polarity）")

    st.info("建议评价库表包含：label（标签名）、polarity（positive/negative 或 好评/差评）")

    tag_file = st.file_uploader("上传评价库（CSV/Excel）", type=["csv", "xlsx"])
    if tag_file:
        tag_df = load_file(tag_file)
        st.dataframe(tag_df.head(10))

        c1, c2 = st.columns(2)
        with c1:
            lbl_col = st.selectbox("标签列(label)", tag_df.columns)
        with c2:
            pol_col = st.selectbox("极性列(polarity)", tag_df.columns)

        if st.button("加载评价库"):
            tmp = tag_df.copy()
            tmp["pol_norm"] = tmp[pol_col].apply(normalize_polarity)

            pos = tmp[tmp["pol_norm"] == "positive"][lbl_col].dropna().astype(str).unique().tolist()
            neg = tmp[tmp["pol_norm"] == "negative"][lbl_col].dropna().astype(str).unique().tolist()

            st.session_state.tag_config = {
                "pos": pos,
                "neg": neg,
                "all": list(dict.fromkeys(pos + neg))
            }
            st.success(f"✅ 评价库加载成功：好评 {len(pos)} 个，差评 {len(neg)} 个")

    st.markdown("---")
    st.subheader("当前已加载标签预览")
    st.write({
        "好评标签数": len(st.session_state.tag_config["pos"]),
        "差评标签数": len(st.session_state.tag_config["neg"])
    })
    with st.expander("查看好评标签"):
        st.write(st.session_state.tag_config["pos"])
    with st.expander("查看差评标签"):
        st.write(st.session_state.tag_config["neg"])

# ======================================================
# Tab 3：Prompt 生成（4星优先差评点）
# ======================================================
with tab3:
    st.header("Step 3：生成 Prompt（4 星优先差评点）")

    if st.session_state.normalized_df is None or not st.session_state.mapping_locked:
        st.warning("请先在 Step 1 完成并锁定列映射。")
        st.stop()

    if (not st.session_state.tag_config["pos"]) or (not st.session_state.tag_config["neg"]):
        st.warning("请先在 Step 2 加载评价库（需要同时有好评/差评标签）。")
        st.stop()

    batch_size = st.number_input("每批条数", value=30, min_value=10, max_value=200, step=10)

    def build_prompt(data_chunk, mode):
        pos = ", ".join([f'"{t}"' for t in st.session_state.tag_config["pos"]])
        neg = ", ".join([f'"{t}"' for t in st.session_state.tag_config["neg"]])

        system = (
            "You are an expert review tagger.\n"
            "You MUST choose labels ONLY from the provided tag libraries.\n"
            "Return STRICT JSON only (no explanations, no extra text).\n"
            "Output schema: [{\"id\":\"...\",\"label\":\"\"}].\n"
            "If no suitable tag, label must be empty string \"\".\n"
        )

        if mode == "1-3":
            task = f"""
TASK:
These are 1-3 star reviews.
You MUST choose from NEGATIVE TAG LIBRARY only.

NEGATIVE TAG LIBRARY:
[{neg}]
"""
        elif mode == "5":
            task = f"""
TASK:
These are 5 star reviews.
You MUST choose from POSITIVE TAG LIBRARY only.

POSITIVE TAG LIBRARY:
[{pos}]
"""
        else:  # 4
            task = f"""
TASK:
These are 4 star reviews. PRIORITIZE complaints.
Rule:
1) If review contains ANY complaint/negative point, choose from NEGATIVE TAG LIBRARY.
2) Otherwise choose from POSITIVE TAG LIBRARY.
3) If still no suitable tag, output "".

NEGATIVE TAG LIBRARY:
[{neg}]

POSITIVE TAG LIBRARY:
[{pos}]
"""

        data = "DATA (JSON):\n" + json.dumps(data_chunk, ensure_ascii=False, indent=2)
        return f"{system}\n{task}\n{data}"

    if st.button("生成 Prompt"):
        df = st.session_state.normalized_df
        batches = []

        groups = {
            "1-3": df[df["rating"] <= 3],
            "4": df[df["rating"] == 4],
            "5": df[df["rating"] == 5],
        }

        for mode, gdf in groups.items():
            if gdf.empty:
                continue
            records = gdf.to_dict("records")
            for i in range(0, len(records), int(batch_size)):
                chunk = records[i:i+int(batch_size)]
                batches.append({
                    "title": f"[{mode}星] 批次 {i//int(batch_size)+1}（{len(chunk)}条）",
                    "prompt": build_prompt(chunk, mode)
                })

        st.session_state.generated_batches = batches
        st.success(f"✅ 已生成 {len(batches)} 个任务包")

    for b in st.session_state.generated_batches:
        with st.expander(b["title"]):
            st.text_area("Prompt（复制给模型）", b["prompt"], height=280)
            st.caption("提示：让模型只返回 JSON，避免夹带解释导致解析失败。")

# ======================================================
# Tab 4：结果回填 & 导出
# ======================================================
with tab4:
    st.header("Step 4：回填结果（严格校验库内标签）& 导出")

    if st.session_state.normalized_df is None:
        st.warning("请先完成 Step 1 & Step 3。")
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
                # 严格校验：库外标签置空
                res_df["label"] = res_df["label"].apply(lambda x: validate_label(x, allowed_set))

                id_map = dict(zip(res_df["id"], res_df["label"]))

                df = st.session_state.normalized_df.copy()
                if "AI_Label" not in df.columns:
                    df["AI_Label"] = ""

                df["AI_Label"] = df["id"].map(id_map).fillna(df["AI_Label"]).astype(str)
                st.session_state.normalized_df = df

                st.success(f"✅ 合并完成：本次合并 {len(res_df)} 条（库外标签已自动置空）")
                st.dataframe(df.head(20))

    st.markdown("---")
    st.subheader("导出结果")

    if st.session_state.normalized_df is not None:
        out_csv = st.session_state.normalized_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载打标结果 CSV（normalized）", out_csv, "tagged_reviews_normalized.csv", "text/csv")

    # 如果你也想导出主表（带原字段 + AI_Label），可以做一次 merge
    if st.session_state.main_df is not None and st.session_state.normalized_df is not None:
        id_col = st.session_state.id_col_in_main  # 主表里的ID列名
        main = st.session_state.main_df.copy()
        main[id_col] = main[id_col].astype(str)

        lab = st.sessio
