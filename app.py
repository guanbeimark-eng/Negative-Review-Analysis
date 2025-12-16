import streamlit as st
import pandas as pd
import json
import uuid
import io

# ==========================================
# 0. 基础配置与安全登录
# ==========================================
st.set_page_config(
    page_title="LLM 评论智能打标 (新逻辑版)", 
    page_icon="🏷️", 
    layout="wide"
)

# --- 简单的密码保护 ---
ACCESS_PASSWORD = "admin123" 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    if st.session_state["password_input"] == ACCESS_PASSWORD:
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
if 'main_df' not in st.session_state: st.session_state.main_df = None
if 'normalized_df' not in st.session_state: st.session_state.normalized_df = None
if 'tag_config' not in st.session_state: st.session_state.tag_config = {"pos": [], "neg": [], "all": []}
if 'generated_batches' not in st.session_state: st.session_state.generated_batches = []
if 'temp_result_df' not in st.session_state: st.session_state.temp_result_df = None

# ==========================================
# 2. 工具函数
# ==========================================
def load_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def safe_json_parse(json_str):
    if not json_str: return None
    try:
        clean_str = json_str.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_str)
    except json.JSONDecodeError:
        return None

# ==========================================
# 3. 页面主体
# ==========================================
st.title("🚀 评论自动打标工具 (Updated Logic)")
st.markdown("### 逻辑：1-3星(差评库) | 5星(好评库) | 4星(综合分析)")

tab1, tab2, tab3, tab4 = st.tabs(["📂 1.数据导入", "🏷️ 2.评价库", "🤖 3.生成Prompt", "📥 4.结果回填"])

# ------------------------------------------
# Tab 1: 数据导入
# ------------------------------------------
with tab1:
    st.header("Step 1: 上传原始评论")
    uploaded_file = st.file_uploader("上传 Excel/CSV", type=['csv', 'xlsx'])

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            st.session_state.main_df = df
            st.dataframe(df.head(3))

            st.subheader("🔧 字段映射")
            all_cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            
            # 智能预选
            idx_rating = all_cols.index('rating') if 'rating' in all_cols else 0
            idx_content = all_cols.index('content') if 'content' in all_cols else 0
            
            with c1: col_rating = st.selectbox("Rating (星级)", all_cols, index=idx_rating)
            with c2: col_content = st.selectbox("Content (内容)", all_cols, index=idx_content)
            with c3: col_id_opt = st.selectbox("ID 列", ["-- 自动生成 UUID --"] + all_cols)

            if st.button("标准化数据", type="primary"):
                norm_df = df.copy()
                
                # ID处理
                if col_id_opt.startswith("--"):
                    norm_df['sys_uuid'] = [str(uuid.uuid4())[:8] for _ in range(len(norm_df))]
                    st.session_state.main_df['sys_uuid'] = norm_df['sys_uuid'] 
                    target_id_col = 'sys_uuid'
                else:
                    norm_df[col_id_opt] = norm_df[col_id_opt].astype(str)
                    target_id_col = col_id_opt

                # 星级处理
                norm_df['rating_std'] = pd.to_numeric(norm_df[col_rating], errors='coerce').fillna(0).astype(int)
                
                # 保存标准表
                st.session_state.normalized_df = norm_df[[target_id_col, 'rating_std', col_content]].rename(
                    columns={target_id_col: 'id', 'rating_std': 'rating', col_content: 'text'}
                )
                st.success("✅ 数据已准备就绪")

# ------------------------------------------
# Tab 2: 评价库
# ------------------------------------------
with tab2:
    st.header("Step 2: 导入标签库")
    st.info("表头需包含: `label`, `polarity` (positive/negative)")
    tag_file = st.file_uploader("上传标签库", type=['csv', 'xlsx'])
    
    if tag_file:
        tag_df = load_file(tag_file)
        if tag_df is not None:
            c1, c2 = st.columns(2)
            lbl_col = c1.selectbox("标签列", tag_df.columns)
            pol_col = c2.selectbox("极性列", tag_df.columns)
            
            if st.button("加载标签"):
                tag_df['p_lower'] = tag_df[pol_col].astype(str).str.lower()
                pos = tag_df[tag_df['p_lower'].str.contains('pos|good|好')][lbl_col].dropna().unique().tolist()
                neg = tag_df[tag_df['p_lower'].str.contains('neg|bad|差')][lbl_col].dropna().unique().tolist()
                
                st.session_state.tag_config = {"pos": pos, "neg": neg, "all": list(set(pos + neg))}
                st.success(f"已加载: 好评 {len(pos)} 个, 差评 {len(neg)} 个")

# ------------------------------------------
# Tab 3: Prompt 生成 (核心修改逻辑)
# ------------------------------------------
with tab3:
    st.header("Step 3: 生成指令")
    
    if st.session_state.normalized_df is None:
        st.warning("请先完成 Step 1")
        st.stop()

    batch_size = st.number_input("每批条数", value=30, min_value=10)
    
    # --- 核心 Prompt 构建函数 ---
    def build_prompt(data_chunk, rating_mode):
        # 准备标签字符串
        pos_tags_str = ", ".join([f'"{t}"' for t in st.session_state.tag_config['pos']])
        neg_tags_str = ", ".join([f'"{t}"' for t in st.session_state.tag_config['neg']])
        
        # 基础系统设定 (强制 JSON 以保证程序可运行，但逻辑遵循您的要求)
        system_part = """You are a customer review analysis assistant.
Your goal is to assign a summary tag to each review based on strict rules.
Output Format: Strictly Valid JSON list: [{"id": "...", "label": "..."}].
Do not output CSV text, output JSON structure so the system can parse it."""

        # 根据星级定制逻辑
        if rating_mode == '1-3':
            # 1-3星：只看差评
            task_part = f"""
TASK:
Please assign a summary tag to each customer review from the specific "Negative Tag Library" provided below.
CONTEXT:
These are 1-3 Star reviews (Negative).
RULES:
1. You must ONLY use tags from the NEGATIVE LIBRARY.
2. If none of the tags fit, leave the label value as an empty string.
3. Place the tag in the 'label' field.

NEGATIVE LIBRARY:
[{neg_tags_str}]
"""
        elif rating_mode == '5':
            # 5星：只看好评
            task_part = f"""
TASK:
Please assign a summary tag to each customer review from the specific "Positive Tag Library" provided below.
CONTEXT:
These are 5 Star reviews (Positive).
RULES:
1. You must ONLY use tags from the POSITIVE LIBRARY.
2. If none of the tags fit, leave the label value as an empty string.
3. Place the tag in the 'label' field.

POSITIVE LIBRARY:
[{pos_tags_str}]
"""
        else:
            # 4星：综合分析 (Both Lists)
            task_part = f"""
TASK:
Please assign a summary tag to each customer review.
CONTEXT:
These are 4 Star reviews. They can be ambiguous.
RULES:
1. Analyze the review content carefully.
2. Choose ONE best suitable tag from EITHER the "Positive Library" OR the "Negative Library".
3. If the review contains a complaint, prioritize the Negative Library.
4. If the review is purely praise, use the Positive Library.
5. If none fit, leave the label empty.

POSITIVE LIBRARY:
[{pos_tags_str}]

NEGATIVE LIBRARY:
[{neg_tags_str}]
"""

        data_part = f"DATA PAYLOAD:\n{json.dumps(data_chunk, ensure_ascii=False, indent=2)}"
        return f"{system_part}\n{task_part}\n{data_part}"

    if st.button("生成 Prompt"):
        df = st.session_state.normalized_df
        batches = []
        
        # 1. 自动根据星级分流
        groups = {
            '1-3': df[df['rating'] <= 3],
            '4':   df[df['rating'] == 4],
            '5':   df[df['rating'] == 5]
        }
        
        for r_mode, g_df in groups.items():
            if g_df.empty: continue
            records = g_df.to_dict(orient='records')
            
            # 切片
            for i in range(0, len(records), batch_size):
                chunk = records[i:i+batch_size]
                prompt_text = build_prompt(chunk, r_mode)
                batches.append({
                    "title": f"[{r_mode}星组] 批次 {i//batch_size + 1} ({len(chunk)}条)",
                    "prompt": prompt_text
                })
        
        st.session_state.generated_batches = batches
        st.success(f"生成了 {len(batches)} 个任务卡片")

    # 展示
    for b in st.session_state.generated_batches:
        with st.expander(b["title"]):
            st.text_area("Prompt", b["prompt"], height=200)
            st.info("复制上方内容 -> 发送给 AI")

# ------------------------------------------
# Tab 4: 结果回填
# ------------------------------------------
with tab4:
    st.header("Step 4: 结果回填")
    st.caption("请将 AI 返回的 JSON 粘贴到下方")
    
    json_input = st.text_area("JSON 结果", height=200)
    
    if st.button("合并结果"):
        data = safe_json_parse(json_input)
        if data:
            res_df = pd.DataFrame(data)
            if 'id' in res_df.columns and 'label' in res_df.columns:
                st.session_state.temp_result_df = res_df
                
                # 执行合并
                main = st.session_state.main_df
                # 寻找ID列
                id_col = 'sys_uuid' if 'sys_uuid' in main.columns else st.session_state.normalized_df.columns[0]
                
                id_map = dict(zip(res_df['id'], res_df['label']))
                
                if 'AI_Label' not in main.columns: main['AI_Label'] = None
                
                main['AI_Label'] = main.apply(
                    lambda row: id_map.get(str(row.get(id_col)), row['AI_Label']), axis=1
                )
                
                st.session_state.main_df = main
                st.success("合并成功！")
                st.dataframe(main[['rating', 'AI_Label']].head())
            else:
                st.error("JSON 缺少 id 或 label 字段")
        else:
            st.error("无法解析 JSON")

    if st.session_state.main_df is not None:
        csv = st.session_state.main_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载最终 CSV", csv, "final_result.csv", "text/csv")
