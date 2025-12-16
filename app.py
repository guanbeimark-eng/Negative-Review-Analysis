import streamlit as st
import pandas as pd
import json
import uuid

# ==========================================
# 0. 基础配置与安全登录
# ==========================================
st.set_page_config(
    page_title="LLM 评论智能打标",
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

# ==========================================
# 1. Session State 初始化
# ==========================================
if "main_df" not in st.session_state: st.session_state.main_df = None
if "normalized_df" not in st.session_state: st.session_state.normalized_df = None
if "tag_config" not in st.session_state: st.session_state.tag_config = {"pos": [], "neg": [], "all": []}
if "generated_batches" not in st.session_state: st.session_state.generated_batches = []
if "id_col_in_main" not in st.session_state: st.session_state.id_col_in_main = None  # ✅关键：保存主表里的ID列名

# ==========================================
# 2. 工具函数
# ==========================================
def load_file(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding="gbk")
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def safe_json_parse_maybe_multi(json_str: str):
    """支持粘贴多段 JSON：可以是单个list，也可以是多段list拼一起（用换行分隔）。"""
    if not json_str:
        return None
    clean = json_str.replace("```json", "").replace("```", "").strip()
    if not clean:
        return None

    # 先尝试整体解析
    try:
        obj = json.loads(clean)
        return obj
    except Exception:
        pass

    # 尝试按段落拆分解析并合并
    parts = [p.strip() for p in clean.split("\n\n") if p.strip()]
    merged = []
    ok_any = False
    for p in parts:
        try:
            obj = json.loads(p)
            if isinstance(obj, list):
                merged.extend(obj)
                ok_any = True
        except Exception:
            continue
    return merged if ok_any else None

def normalize_polarity(x: str) -> str:
    s = str(x).strip().lower()
    # 常见写法容错
    if s in ["positive", "pos", "good", "好评", "正向", "正"]:
        return "positive"
    if s in ["negative", "neg", "bad", "差评", "负向", "负"]:
        return "negative"
    # 模糊匹配
    if any(k in s for k in ["pos", "good", "好", "正"]): return "positive"
    if any(k in s for k in ["neg", "bad", "差", "负"]): return "negative"
    return ""

def validate_label(label: str, allowed_set: set) -> str:
    """严格校验：只允许库内标签；否则返回空字符串"""
    if label is None:
        return ""
    lab = str(label).strip()
    return lab if lab in allowed_set else ""

# ==========================================
# 3. 页面主体
# ==========================================
st.title("🏷️ 评论数据打标系统（按评价库标签输出）")
tab1, tab2, tab3, tab4 = st.tabs(["1. 数据看板 & 清洗", "2. 评价库配置", "3. 生成 Prompt", "4. 结果回填"])

# ------------------------------------------
# Tab 1: 数据导入 & 可视化看板
# ------------------------------------------
with tab1:
    st.header("Step 1: 数据导入与概览")
    uploaded_file = st.file_uploader("上传 Excel/CSV 文件", type=["csv", "xlsx"])

    if uploaded_file:
        df_raw = load_file(uploaded_file)
        if df_raw is not None:
            st.info(f"📄 文件读取成功！检测到 **{len(df_raw)}** 行数据。")
            st.dataframe(df_raw.head(5))

            st.markdown("---")
            st.subheader("🔧 关键字段设置")

            all_cols = df_raw.columns.tolist()
            c1, c2, c3, c4, c5 = st.columns(5)

            idx_rating = all_cols.index("rating") if "rating" in all_cols else 0
            idx_title  = all_cols.index("title") if "title" in all_cols else 0
            idx_content = all_cols.index("content") if "content" in all_cols else 0
            idx_date = all_cols.index("date") if "date" in all_cols else None

            with c1: col_rating = st.selectbox("Rating (星级)", all_cols, index=idx_rating)
            with c2: col_title = st.selectbox("Title (标题-可选)", ["--不使用--"] + all_cols, index=(idx_title + 1) if "title" in all_cols else 0)
            with c3: col_content = st.selectbox("Content (内容)", all_cols, index=idx_content)
            with c4:
                date_options = ["--不分析--"] + all_cols
                col_date = st.selectbox("Date (时间-可选)", date_options, index=(idx_date + 1) if idx_date is not None else 0)
            with c5: col_id_opt = st.selectbox("ID (唯一标识)", ["-- 自动生成 UUID --"] + all_cols)

            if st.button("生成看板并标准化", type="primary"):
                clean_df = df_raw.copy()

                # 星级清洗
                clean_df["rating_numeric"] = pd.to_numeric(clean_df[col_rating], errors="coerce")
                clean_df = clean_df.dropna(subset=["rating_numeric"])
                clean_df["rating_int"] = clean_df["rating_numeric"].round().astype(int)
                clean_df = clean_df[clean_df["rating_int"].between(1, 5)]

                # 时间清洗（可选）
                time_parse_success = False
                if col_date != "--不分析--":
                    clean_df["date_parsed"] = pd.to_datetime(clean_df[col_date], errors="coerce")
                    time_parse_success = clean_df["date_parsed"].notna().sum() > 0

                # ID处理（✅保存主表ID列名）
                if col_id_opt.startswith("--"):
                    clean_df["sys_uuid"] = [str(uuid.uuid4())[:8] for _ in range(len(clean_df))]
                    st.session_state.id_col_in_main = "sys_uuid"
                else:
                    clean_df[col_id_opt] = clean_df[col_id_opt].astype(str)
                    st.session_state.id_col_in_main = col_id_opt

                # 文本拼接（title可选）
                if col_title != "--不使用--":
                    clean_df["__text_joined__"] = (
                        clean_df[col_title].fillna("").astype(str).str.strip()
                        + " | "
                        + clean_df[col_content].fillna("").astype(str).str.strip()
                    ).str.strip(" |")
                    text_col = "__text_joined__"
                else:
                    text_col = col_content

                st.session_state.main_df = clean_df

                # 规范化表（供prompt）
                st.session_state.normalized_df = clean_df[
                    [st.session_state.id_col_in_main, "rating_int", text_col]
                ].rename(columns={
                    st.session_state.id_col_in_main: "id",
                    "rating_int": "rating",
                    text_col: "text"
                })

                # 看板
                st.markdown("---")
                total = len(clean_df)
                neg_rate = (len(clean_df[clean_df["rating_int"] <= 3]) / total * 100) if total else 0
                k1, k2, k3 = st.columns(3)
                k1.metric("有效评论数", total)
                k2.metric("平均分", f"{clean_df['rating_int'].mean():.2f}" if total else "N/A")
                k3.metric("差评率(<=3星)", f"{neg_rate:.1f}%")

                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    counts = clean_df["rating_int"].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
                    st.bar_chart(counts)
                with c_chart2:
                    if time_parse_success:
                        # 月度趋势
                        tmp = clean_df.dropna(subset=["date_parsed"]).set_index("date_parsed")
                        st.line_chart(tmp.resample("M").size())
                    else:
                        st.info("暂无时间趋势数据或时间列解析失败")

                st.success("✅ 数据准备就绪")

# ------------------------------------------
# Tab 2: 评价库配置
# ------------------------------------------
with tab2:
    st.header("Step 2: 导入标签库")
    st.info("建议表头包含: `label`, `polarity`（positive/negative 或 好评/差评）")

    tag_file = st.file_uploader("上传标签库", type=["csv", "xlsx"], key="tag_uploader")

    if tag_file:
        tag_df = load_file(tag_file)
        if tag_df is not None:
            c1, c2 = st.columns(2)
            lbl_col = c1.selectbox("标签列(label)", tag_df.columns)
            pol_col = c2.selectbox("极性列(polarity)", tag_df.columns)

            if st.button("加载标签"):
                tmp = tag_df.copy()
                tmp["pol_norm"] = tmp[pol_col].apply(normalize_polarity)

                pos = tmp[tmp["pol_norm"] == "positive"][lbl_col].dropna().astype(str).unique().tolist()
                neg = tmp[tmp["pol_norm"] == "negative"][lbl_col].dropna().astype(str).unique().tolist()

                st.session_state.tag_config = {"pos": pos, "neg": neg, "all": list(dict.fromkeys(pos + neg))}
                st.success(f"✅ 已加载: 好评 {len(pos)} 个, 差评 {len(neg)} 个")

    # 展示当前库
    st.markdown("---")
    st.subheader("当前已加载标签预览")
    st.write({"好评标签数": len(st.session_state.tag_config["pos"]), "差评标签数": len(st.session_state.tag_config["neg"])})
    with st.expander("查看好评标签"):
        st.write(st.session_state.tag_config["pos"])
    with st.expander("查看差评标签"):
        st.write(st.session_state.tag_config["neg"])

# ------------------------------------------
# Tab 3: Prompt 生成
# ------------------------------------------
with tab3:
    st.header("Step 3: 生成 Prompt（4星优先差评点）")

    if st.session_state.normalized_df is None:
        st.warning("请先完成 Step 1：数据导入与标准化")
        st.stop()

    if (not st.session_state.tag_config["pos"]) or (not st.session_state.tag_config["neg"]):
        st.warning("请先完成 Step 2：加载标签库（需要同时有好评与差评标签）")
        st.stop()

    batch_size = st.number_input("每批条数", value=30, min_value=10, max_value=200, step=10)

    def build_prompt(data_chunk, rating_mode):
        pos_tags_str = ", ".join([f'"{t}"' for t in st.session_state.tag_config["pos"]])
        neg_tags_str = ", ".join([f'"{t}"' for t in st.session_state.tag_config["neg"]])

        system_part = (
            "You are an expert customer review tagger.\n"
            "You MUST select labels ONLY from the provided tag library.\n"
            "Return STRICT JSON only, no explanations, no extra text.\n"
            'Output schema: [{"id": "...", "label": ""}] where label is either a library tag or empty string.\n'
        )

        if rating_mode == "1-3":
            task_part = f"""
TASK:
These are 1-3 star reviews. You MUST choose from NEGATIVE tag library only.
If no suitable tag, output "".

NEGATIVE TAG LIBRARY:
[{neg_tags_str}]
"""
        elif rating_mode == "5":
            task_part = f"""
TASK:
These are 5 star reviews. You MUST choose from POSITIVE tag library only.
If no suitable tag, output "".

POSITIVE TAG LIBRARY:
[{pos_tags_str}]
"""
        else:  # 4-star
            task_part = f"""
TASK:
These are 4 star reviews. PRIORITIZE complaints.
Rule:
1) If the review contains ANY complaint/negative point, choose from NEGATIVE tag library.
2) Otherwise choose from POSITIVE tag library.
3) If still no suitable tag, output "".

POSITIVE TAG LIBRARY:
[{pos_tags_str}]

NEGATIVE TAG LIBRARY:
[{neg_tags_str}]
"""

        data_part = "DATA (JSON):\n" + json.dumps(data_chunk, ensure_ascii=False, indent=2)
        return f"{system_part}\n{task_part}\n{data_part}"

    if st.button("生成 Prompt"):
        df = st.session_state.normalized_df
        batches = []

        groups = {
            "1-3": df[df["rating"] <= 3],
            "4":   df[df["rating"] == 4],
            "5":   df[df["rating"] == 5],
        }

        for r_mode, g_df in groups.items():
            if g_df.empty:
                continue
            records = g_df.to_dict(orient="records")
            for i in range(0, len(records), int(batch_size)):
                chunk = records[i:i+int(batch_size)]
                prompt_text = build_prompt(chunk, r_mode)
                batches.append({
                    "title": f"[{r_mode}星] 批次 {i//int(batch_size)+1}（{len(chunk)}条）",
                    "prompt": prompt_text
                })

        st.session_state.generated_batches = batches
        st.success(f"✅ 已生成 {len(batches)} 个任务包")

    for b in st.session_state.generated_batches:
        with st.expander(b["title"]):
            st.text_area("Prompt（复制给模型）", b["prompt"], height=260)

# ------------------------------------------
# Tab 4: 结果回填
# ------------------------------------------
with tab4:
    st.header("Step 4: 结果回填（严格校验库内标签）")

    if st.session_state.main_df is None or st.session_state.normalized_df is None:
        st.warning("请先完成 Step 1")
        st.stop()

    allowed_set = set(st.session_state.tag_config["all"])
    json_input = st.text_area("粘贴 LLM 返回的 JSON（可一次粘贴多批次）", height=220)

    if st.button("合并结果"):
        data = safe_json_parse_maybe_multi(json_input)
        if not data or not isinstance(data, list):
            st.error("JSON 解析失败：请确保返回的是 JSON list，例如：[{'id':'xxx','label':'...'}]")
        else:
            res_df = pd.DataFrame(data)
            if "id" not in res_df.columns or "label" not in res_df.columns:
                st.error("JSON 格式错误：必须包含 id 与 label 字段")
            else:
                # ✅严格校验label
                res_df["id"] = res_df["id"].astype(str)
                res_df["label"] = res_df["label"].apply(lambda x: validate_label(x, allowed_set))

                # 统计库外标签（被置空的数量）
                invalid_cnt = (pd.Series([x.get("label") for x in data]).astype(str).apply(lambda s: s.strip()).apply(lambda s: s != "" and s not in allowed_set)).sum()

                id_map = dict(zip(res_df["id"], res_df["label"]))

                main = st.session_state.main_df
                id_col = st.session_state.id_col_in_main  # ✅准确使用主表ID列
                main[id_col] = main[id_col].astype(str)

                if "AI_Label" not in main.columns:
                    main["AI_Label"] = ""

                main["AI_Label"] = main[id_col].map(id_map).fillna(main["AI_Label"]).astype(str)

                st.session_state.main_df = main
                st.success(f"✅ 合并成功！本次合并 {len(res_df)} 条；库外标签已自动置空（估算 {invalid_cnt} 条）。")
                st.dataframe(main[[id_col, "rating_int", "AI_Label"]].head(20))

    if st.session_state.main_df is not None:
        csv = st.session_state.main_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载结果 CSV", csv, "tagged_result.csv", "text/csv")
