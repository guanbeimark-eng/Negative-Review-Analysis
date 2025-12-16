import streamlit as st
import pandas as pd
import json
import uuid
import re
import numpy as np
from typing import List, Dict, Any, Optional

# OpenAI SDK (pip install openai)
from openai import OpenAI

# =========================
# 0) App Config + Login
# =========================
st.set_page_config(page_title="评论自动打标（傻瓜式一键版）", page_icon="🏷️", layout="wide")

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
# 你后续把这里替换成你们“文件评价库”的正式标签即可
# =========================
DEFAULT_TAG_LIBRARY = {
    "positive": [
        "佩戴舒适", "支撑性好", "缓解关节不适", "尺寸合适", "质量好", "性价比高", "效果明显", "物流/发货快", "外观好看"
    ],
    "negative": [
        "尺码偏小", "尺码偏大", "尺码不一致", "不适合男士", "穿戴困难", "质量差", "与描述不符",
        "不舒适/勒手", "气味/异味", "耐用性差/易破", "压力/压缩感不足"
    ]
}

# =========================
# 2) Session State
# =========================
defaults = {
    "raw_df": None,
    "main_df": None,          # 清洗后主表（原字段 + rating_int + sys_id + __text__）
    "norm_df": None,          # id/rating/text
    "full_df": None,          # 主表+AI_Label 合并后的导出表
    "col_map": None,
    "tag_config": {"pos": DEFAULT_TAG_LIBRARY["positive"],
                   "neg": DEFAULT_TAG_LIBRARY["negative"],
                   "all": DEFAULT_TAG_LIBRARY["positive"] + DEFAULT_TAG_LIBRARY["negative"]},
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
    """兼容：'4.0 out of 5 stars' / 'Rated 3' / '5' / 4.0"""
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
    for c in candidates:
        if c in cols:
            return c
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
        "text": text_primary,   # 优先翻译列
        "date": col_date,
        "id": col_id,
        "content_raw": col_content,
        "translation": col_trans
    }

def build_cleaned_frames(df_raw, m):
    tmp = df_raw.copy()

    tmp["rating_numeric"] = tmp[m["rating"]].apply(parse_rating) if m.get("rating") else np.nan
    invalid_rating_cnt = int(tmp["rating_numeric"].isna().sum())

    valid = tmp.dropna(subset=["rating_numeric"]).copy()
    valid["rating_int"] = valid["rating_numeric"].round().astype(int)
    valid = valid[valid["rating_int"].between(1, 5)]

    time_ok = False
    if m.get("date") and m["date"] in valid.columns:
        valid["date_parsed"] = pd.to_datetime(valid[m["date"]], errors="coerce")
        time_ok = valid["date_parsed"].notna().sum() > 0

    if m.get("id") and m["id"] in valid.columns:
        valid["sys_id"] = valid[m["id"]].astype(str)
    else:
        valid["sys_id"] = [str(uuid.uuid4())[:8] for _ in range(len(valid))]

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

def strict_json_load(s: str) -> Optional[Any]:
    """尽量从模型输出里抠出 JSON（支持包裹在```里、前后有杂字的情况）"""
    if not s:
        return None
    s = s.strip().replace("```json", "").replace("```", "").strip()

    # 直接尝试
    try:
        return json.loads(s)
    except:
        pass

    # 尝试提取第一个 [...] 段
    m = re.search(r"(\[\s*\{.*\}\s*\])", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            return None
    return None

def validate_label(label: str, allowed_set: set) -> str:
    lab = (label or "").strip()
    return lab if lab in allowed_set else ""

# =========================
# 4) OpenAI 调用：一键自动打标
# =========================
def build_api_prompt(records: List[Dict[str, Any]],
                     mode: str,
                     pos_tags: List[str],
                     neg_tags: List[str]) -> str:
    pos_str = ", ".join([f'"{t}"' for t in pos_tags])
    neg_str = ", ".join([f'"{t}"' for t in neg_tags])

    header = (
        "你是电商客户评论的标签归类专家。\n"
        "你必须严格只输出 JSON list，格式如下：\n"
        "[{\"id\":\"...\",\"label\":\"...\"}, ...]\n"
        "不要输出任何解释文字，不要输出 markdown 代码块。\n"
        "label 必须从给定标签库中选择；不匹配则输出空字符串 \"\"。\n"
    )

    if mode == "1-3":
        rules = f"\n这些是 1-3 星评论：只能从【差评标签库】选择。\n差评标签库：[{neg_str}]\n"
    elif mode == "5":
        rules = f"\n这些是 5 星评论：只能从【好评标签库】选择。\n好评标签库：[{pos_str}]\n"
    else:  # 4-star
        rules = (
            f"\n这些是 4 星评论：优先找差评点。\n"
            "规则：\n"
            "1) 只要有任何抱怨/不满意/缺点，就优先从【差评标签库】选择。\n"
            "2) 如果完全是夸赞，再从【好评标签库】选择。\n"
            "3) 不匹配输出空字符串。\n"
            f"差评标签库：[{neg_str}]\n"
            f"好评标签库：[{pos_str}]\n"
        )

    payload = "数据如下（JSON）：\n" + json.dumps(records, ensure_ascii=False)
    return header + rules + "\n" + payload

def call_openai_tagging(client: OpenAI,
                        model: str,
                        prompt: str,
                        max_retries: int = 2) -> List[Dict[str, str]]:
    """
    返回：[{id,label}, ...]
    失败会重试（让模型只输出 JSON）
    """
    last_text = ""
    for attempt in range(max_retries + 1):
        resp = client.responses.create(
            model=model,
            input=prompt
        )
        # SDK Quickstart: response.output_text 取文本输出 :contentReference[oaicite:2]{index=2}
        text = getattr(resp, "output_text", "") or ""
        last_text = text

        obj = strict_json_load(text)
        if isinstance(obj, list) and all(isinstance(x, dict) and "id" in x and "label" in x for x in obj):
            # 正常
            return [{"id": str(x["id"]), "label": str(x.get("label", "")).strip()} for x in obj]

        # 重试：给更强约束
        prompt = (
            "再次强调：你只能输出 JSON list，且每个元素只允许包含 id 和 label 两个键。\n"
            "不要输出任何解释，不要输出 ```。\n\n"
            + prompt
        )

    raise ValueError(f"模型输出无法解析为 [{'{id,label}'}] JSON。原始输出片段：{last_text[:500]}")

# =========================
# 5) UI：真正傻瓜式（一个主按钮）
# =========================
st.title("🏷️ 评论自动打标（傻瓜式：上传 → 一键打标 → 导出）")

with st.expander("① 配置 OpenAI API（只需一次）", expanded=True):
    st.caption("建议把 API Key 配在服务器环境变量 OPENAI_API_KEY；也可在此临时输入（仅服务端使用）。")
    api_key = st.text_input("OpenAI API Key", type="password", value="")
    model_name = st.text_input("模型（默认 gpt-5.2）", value="gpt-5.2")

    st.caption("OpenAI 推荐使用 Responses API。:contentReference[oaicite:3]{index=3}")

uploaded = st.file_uploader("② 上传评论文件（CSV / Excel）", type=["csv", "xlsx"])

# 评价库可选编辑（但不强迫）
with st.expander("③ 评价库（可选编辑：默认已内置）", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        pos_text = st.text_area("好评标签（一行一个）", value="\n".join(st.session_state.tag_config["pos"]), height=220)
    with c2:
        neg_text = st.text_area("差评标签（一行一个）", value="\n".join(st.session_state.tag_config["neg"]), height=220)
    if st.button("保存评价库"):
        pos = [x.strip() for x in pos_text.splitlines() if x.strip()]
        neg = [x.strip() for x in neg_text.splitlines() if x.strip()]
        st.session_state.tag_config = {"pos": pos, "neg": neg, "all": pos + neg}
        st.success(f"已保存：好评 {len(pos)} 个 / 差评 {len(neg)} 个")

if uploaded:
    df_raw = load_file(uploaded)
    st.session_state.raw_df = df_raw

    m = auto_build_mapping(df_raw)
    st.session_state.col_map = m

    # 必要列检查
    if not m.get("rating") or not m.get("text"):
        st.error("❌ 自动识别失败：缺少星级列或内容列。请检查列名（建议：星级/内容/内容(翻译)）。")
        st.json(m)
        st.stop()

    valid, norm, invalid_cnt, time_ok = build_cleaned_frames(df_raw, m)
    st.session_state.main_df = valid
    st.session_state.norm_df = norm
    st.session_state.full_df = None

    # 看板（自动）
    raw_total = len(df_raw)
    valid_total = len(valid)
    neg_cnt = int((valid["rating_int"] <= 3).sum())
    neg_rate = (neg_cnt / valid_total * 100) if valid_total else 0
    severe_cnt = int((valid["rating_int"] <= 2).sum())
    severe_rate = (severe_cnt / valid_total * 100) if valid_total else 0

    st.subheader("📊 自动看板")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("原始行数", raw_total)
    k2.metric("有效评分行数", valid_total)
    k3.metric("评分解析失败", invalid_cnt)
    k4.metric("差评占比(≤3⭐)", f"{neg_rate:.1f}%")
    k5.metric("严重差评(≤2⭐)", f"{severe_rate:.1f}%")

    st.bar_chart(valid["rating_int"].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index())

    st.dataframe(norm.head(8))

    # 一键打标按钮（唯一主按钮）
    st.markdown("---")
    st.subheader("④ 一键自动打标（不需要复制粘贴任何东西）")

    batch_size = st.slider("每批条数（越大越快，但更吃上下文）", 20, 120, 60, 10)

    if st.button("🚀 一键自动打标并生成导出文件", type="primary"):
        if not api_key:
            st.error("请先填写 OpenAI API Key（或在服务器设置 OPENAI_API_KEY）。")
            st.stop()

        client = OpenAI(api_key=api_key)

        df = st.session_state.norm_df.copy()
        pos_tags = st.session_state.tag_config["pos"]
        neg_tags = st.session_state.tag_config["neg"]
        allowed_set = set(st.session_state.tag_config["all"])

        # 分组：1-3 / 4 / 5
        groups = {
            "1-3": df[df["rating"] <= 3],
            "4": df[df["rating"] == 4],
            "5": df[df["rating"] == 5],
        }

        # 准备输出列
        if "AI_Label" not in df.columns:
            df["AI_Label"] = ""

        total_jobs = 0
        for _, g in groups.items():
            total_jobs += int(np.ceil(len(g) / batch_size)) if len(g) else 0

        progress = st.progress(0)
        job_done = 0

        # 执行
        for mode, gdf in groups.items():
            if gdf.empty:
                continue
            records = gdf.to_dict("records")

            for i in range(0, len(records), int(batch_size)):
                chunk = records[i:i+int(batch_size)]
                prompt = build_api_prompt(chunk, mode, pos_tags, neg_tags)

                # 调用 API
                try:
                    results = call_openai_tagging(client, model_name, prompt, max_retries=2)
                except Exception as e:
                    st.error(f"❌ 模型调用失败（{mode}星 批次 {i//batch_size+1}）：{e}")
                    st.stop()

                # 回填 + 严格校验标签在库内
                id_map = {r["id"]: validate_label(r["label"], allowed_set) for r in results}
                mask = df["id"].isin(id_map.keys())
                df.loc[mask, "AI_Label"] = df.loc[mask, "id"].map(id_map).fillna(df.loc[mask, "AI_Label"])

                job_done += 1
                progress.progress(min(1.0, job_done / max(1, total_jobs)))

        st.session_state.norm_df = df

        # 合并回主表
        main = st.session_state.main_df.copy()
        lab = df[["id", "AI_Label"]].copy()
        main["sys_id"] = main["sys_id"].astype(str)
        lab["id"] = lab["id"].astype(str)
        merged = main.merge(lab, left_on="sys_id", right_on="id", how="left")
        merged.drop(columns=["id"], inplace=True, errors="ignore")
        st.session_state.full_df = merged

        st.success("✅ 自动打标完成！你可以直接下载结果。")
        st.dataframe(df.head(20))

# 导出区（随时可下载，永远不用复制粘贴）
st.markdown("---")
st.subheader("⑤ 导出")

if st.session_state.norm_df is not None:
    out_norm = st.session_state.norm_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载：normalized（id/rating/text/AI_Label）",
                       out_norm, "tagged_reviews_normalized.csv", "text/csv")

if st.session_state.full_df is not None:
    out_full = st.session_state.full_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载：full（原始字段 + AI_Label）",
                       out_full, "tagged_reviews_full.csv", "text/csv")
