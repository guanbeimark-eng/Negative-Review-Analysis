import streamlit as st
import pandas as pd
import json
import uuid
import io

# ==========================================
# 0. 基础配置与安全登录
# ==========================================
st.set_page_config(
    page_title="LLM 评论智能打标 (思维链版)", 
    page_icon="🧠", 
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
            try:
                return pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding='gbk')
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
st.title("🧠 评论数据分析与打标系统 (思维链增强版)")

tab1, tab2, tab3, tab4 = st.tabs(["1. 数据看板 & 清洗", "2. 评价库配置", "3. 生成 Prompt (Updated)", "4. 结果回填"])

# ------------------------------------------
# Tab 1: 数据导入 & 可视化看板
# ------------------------------------------
with tab1:
    st.header("Step 1: 数据导入与概览")
    uploaded_file = st.file_uploader("上传 Excel/CSV 文件", type=['csv', 'xlsx'])

    if uploaded_file:
        df_raw = load_file(uploaded_file)
        
        if df_raw is not None:
            st.info(f"📄 文件读取成功！检测到 **{len(df_raw)}** 行数据。")
            st.dataframe(df_raw.head(3))

            st.markdown("---")
            st.subheader("🔧 关键字段设置")
            
            all_cols = df_raw.columns.tolist()
            c1, c2, c3, c4 = st.columns(4)
            
            idx_rating = all_cols.index('rating') if 'rating' in all_cols else 0
            idx_content = all_cols.index('content') if 'content' in all_cols else 0
            idx_date = all_cols.index('date') if 'date' in all_cols else 0
            
            with c1: col_rating = st.selectbox("Rating (星级)", all_cols, index=idx_rating)
            with c2: col_content = st.selectbox("Content (内容)", all_cols, index=idx_content)
            with c3: col_date = st.selectbox("Date (时间 - 可选)", ["--不分析--"] + all_cols, index=idx_date + 1 if 'date' in all_cols else 0)
            with c4: col_id_opt = st.selectbox("ID (唯一标识)", ["-- 自动生成 UUID --"] + all_cols)

            if st.button("生成看板并标准化", type="primary"):
                clean_df = df_raw.copy()
                
                # 清洗星级
                clean_df['rating_numeric'] = pd.to_numeric(clean_df[col_rating], errors='coerce')
                clean_df = clean_df.dropna(subset=['rating_numeric'])
                clean_df['rating_int'] = clean_df['rating_numeric'].round().astype(int)
                clean_df = clean_df[clean_df['rating_int'].between(1, 5)]

                # 清洗时间
                time_parse_success = False
                if col_date != "--不分析--":
                    clean_df['date_parsed'] = pd.to_datetime(clean_df[col_date], errors='coerce')
                    if clean_df['date_parsed'].notna().sum() > 0: time_parse_success = True

                # ID处理
                if col_id_opt.startswith("--"):
                    clean_df['sys_uuid'] = [str(uuid.uuid4())[:8] for _ in range(len(clean_df))]
                    target_id_col = 'sys_uuid'
                else:
                    clean_df[col_id_opt] = clean_df[col_id_opt].astype(str)
                    target_id_col = col_id_opt

                st.session_state.main_df = clean_df
                st.session_state.normalized_df = clean_df[[target_id_col, 'rating_int', col_content]].rename(
                    columns={target_id_col: 'id', 'rating_int': 'rating', col_content: 'text'}
                )
                
                # 看板
                st.markdown("---")
                k1, k2, k3 = st.columns(3)
                total = len(clean_df)
                neg_rate = (len(clean_df[clean_df['rating_int'] <= 3]) / total * 100) if total > 0 else 0
                k1.metric("有效评论数", total)
                k2.metric("平均分", f"{clean_df['rating_int'].mean():.2f}")
                k3.metric("差评率", f"{neg_rate:.1f}%")

                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    counts = clean_df['rating_int'].value_counts().reindex([1,2,3,4,5], fill_value=0).sort_index()
                    st.bar_chart(counts)
                with c_chart2:
                    if time_parse_success:
                        st.line_chart(clean_df.set_index('date_parsed').resample('M').size())
                    else:
                        st.info("暂无时间趋势数据")

                st.success("✅ 数据准备就绪")

# ------------------------------------------
# Tab 2: 评价库配置
# ------------------------------------------
with tab2:
    st.header("Step 2: 导入标签库")
    st.info("表头需包含: `label`, `polarity` (positive/negative)")
    tag_file = st.file_uploader("上传标签库", type=['csv', 'xlsx'], key="tag_uploader")
    
    if tag_file:
        tag_df = load_file(tag_file)
        if tag_df is not None:
            c1, c2 = st.columns(2)
            lbl_col = c1.selectbox("标签列", tag_df.columns)
            pol_col = c2.selectbox("极性列", tag_df.columns)
            
            if st.button("加载标签"):
                tag_df['p_lower'] = tag_df[pol_col].astype(str).str.lower()
                pos = tag_df[tag_df['p_lower'].str.contains('pos|good|好|正')][lbl_col].dropna().unique().tolist()
                neg = tag_df[tag_df['p_lower'].str.contains('neg|bad|差|负')][lbl_col].dropna().unique().tolist()
                
                st.session_state.tag_config = {"pos": pos, "neg": neg, "all": list(set(pos + neg))}
                st.success(f"已加载: 好评 {len(pos)} 个, 差评 {len(neg)} 个")

# ------------------------------------------
# Tab 3: Prompt 生成 (重点更新)
# ------------------------------------------
with tab3:
    st.header("Step 3: 生成思维链 Prompt")
    st.markdown("💡 **新逻辑**：模型将先生成“临时总结标签”，再映射到标准库。")
    
    if st.session_state.normalized_df is None:
        st.warning("请先完成 Step 1")
        st.stop()

    batch_size = st.number_input("每批条数", value=30, min_value=10)
    
    def build_prompt(data_chunk, rating_mode):
        pos_tags_str = ", ".join([f'"{t}"' for t in st.session_state.tag_config['pos']])
        neg_tags_str = ", ".join([f'"{t}"' for t in st.session_state.tag_config['neg']])
        
        # System Prompt: 设定 JSON 输出格式
        system_part = """You are an expert customer review analyzer.
Your goal is to assign a standardized summary tag to each review.
OUTPUT FORMAT: Strictly Valid JSON list: [{"id": "...", "label": "..."}].
Do not output CSV text or explanations, only the JSON structure."""

        # Shared Logic: 定义思维链过程
        reasoning_logic = """
### THINKING PROCESS (Internal Step):
1. **Analyze**: Read the review content carefully.
2. **Draft Temporary Label**: Mentally generate a "Temporary Generic Summary Label" that best describes the review content.
3. **Map to Library**: Compare your "Temporary Label" with the provided [STANDARD TAG LIBRARY] below.
4. **Final Decision**: 
   - If your temporary label matches (or is a synonym of) a tag in the Library, output the **Library Tag**.
   - If the review does not fit any tag in the Library, output an empty string "".
"""

        if rating_mode == '1-3':
            task_part = f"""
{reasoning_logic}
### CONTEXT
These are 1-3 Star reviews (Negative). 
**STANDARD TAG LIBRARY (Negative)**:
[{neg_tags_str}]
"""
        elif rating_mode == '5':
            task_part = f"""
{reasoning_logic}
### CONTEXT
These are 5 Star reviews (Positive).
**STANDARD TAG LIBRARY (Positive)**:
[{pos_tags_str}]
"""
        else: # 4 star
            task_part = f"""
{reasoning_logic}
### CONTEXT
These are 4 Star reviews (Ambiguous).
**STANDARD TAG LIBRARY (Combined)**:
- **Positive List**: [{pos_tags_str}]
- **Negative List**: [{neg_tags_str}]

**Priority Rule**: If the review contains ANY complaint, prioritize the Negative List. Otherwise, use the Positive List.
"""
        data_part = f"DATA PAYLOAD:\n{json.dumps(data_chunk, ensure_ascii=False, indent=2)}"
        return f"{system_part}\n{task_part}\n{data_part}"

    if st.button("生成 Prompt"):
        df = st.session_state.normalized_df
        batches = []
        
        groups = {
            '1-3': df[df['rating'] <= 3],
            '4':   df[df['rating'] == 4],
            '5':   df[df['rating'] == 5]
        }
        
        for r_mode, g_df in groups.items():
            if g_df.empty: continue
            records = g_df.to_dict(orient='records')
            for i in range(0, len(records), batch_size):
                chunk = records[i:i+batch_size]
                prompt_text = build_prompt(chunk, r_mode)
                batches.append({
                    "title": f"[{r_mode}星组] 批次 {i//batch_size + 1} ({len(chunk)}条)",
                    "prompt": prompt_text
                })
        
        st.session_state.generated_batches = batches
        st.success(f"已生成 {len(batches)} 个任务包")

    for b in st.session_state.generated_batches:
        with st.expander(b["title"]):
            st.text_area("Prompt", b["prompt"], height=200)
            st.caption("复制上方内容发送给 AI。AI 会在内部进行‘临时总结->标准映射’的过程，但最终只返回符合格式的 JSON。")

# ------------------------------------------
# Tab 4: 结果回填
# ------------------------------------------
with tab4:
    st.header("Step 4: 结果回填")
    json_input = st.text_area("粘贴 LLM 返回的 JSON", height=200)
    
    if st.button("合并结果"):
        data = safe_json_parse(json_input)
        if data:
            res_df = pd.DataFrame(data)
            if 'id' in res_df.columns and 'label' in res_df.columns:
                main = st.session_state.main_df
                id_col = 'sys_uuid' if 'sys_uuid' in main.columns else st.session_state.normalized_df.columns[0]
                
                id_map = dict(zip(res_df['id'], res_df['label']))
                
                if 'AI_Label' not in main.columns: main['AI_Label'] = None
                
                main['AI_Label'] = main.apply(
                    lambda row: id_map.get(str(row.get(id_col)), row['AI_Label']), axis=1
                )
                
                st.session_state.main_df = main
                st.success(f"合并成功！")
                st.dataframe(main[['rating_int', 'AI_Label']].head())
            else:
                st.error("JSON 格式错误")
        else:
            st.error("JSON 解析失败")

    if st.session_state.main_df is not None:
        csv = st.session_state.main_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载结果 CSV", csv, "tagged_result.csv", "text/csv")
