import streamlit as st
import pandas as pd
import json
import uuid
import io

# ==========================================
# 0. 基础配置与安全登录
# ==========================================
st.set_page_config(
    page_title="LLM 评论智能清洗屋", 
    page_icon="🧹", 
    layout="wide"
)

# --- 简单的密码保护 (适合公开部署) ---
# 修改这里的密码
ACCESS_PASSWORD = "admin123" 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    """验证密码回调"""
    if st.session_state["password_input"] == ACCESS_PASSWORD:
        st.session_state.logged_in = True
    else:
        st.error("密码错误，请重试")

if not st.session_state.logged_in:
    st.markdown("## 🔒 系统锁定")
    st.markdown("该工具已部署在云端，请输入访问密码以继续。")
    st.text_input("访问密码", type="password", key="password_input", on_change=check_password)
    st.stop()  # 停止执行后续代码

# ==========================================
# 1. 全局 Session State 初始化
# ==========================================
if 'main_df' not in st.session_state:
    st.session_state.main_df = None       # 原始数据
if 'normalized_df' not in st.session_state:
    st.session_state.normalized_df = None # 标准化后的精简数据
if 'tag_config' not in st.session_state:
    st.session_state.tag_config = {"pos": [], "neg": [], "all": []} # 标签库配置
if 'generated_batches' not in st.session_state:
    st.session_state.generated_batches = [] # 生成的 Prompt 批次

# ==========================================
# 2. 工具函数
# ==========================================
def load_file(uploaded_file):
    """兼容 CSV 和 Excel 的加载函数"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def safe_json_parse(json_str):
    """清洗并解析 LLM 返回的 JSON"""
    if not json_str: return None
    try:
        # 1. 移除 Markdown 代码块标记
        clean_str = json_str.replace("```json", "").replace("```", "").strip()
        # 2. 尝试解析
        return json.loads(clean_str)
    except json.JSONDecodeError:
        st.error("JSON 解析失败。请检查模型返回的内容是否包含非 JSON 文字。")
        return None

# ==========================================
# 3. 页面主体布局
# ==========================================
st.title("🚀 LLM 评论数据清洗流水线")
st.markdown("### 流程：导入数据 → 配置标签 → 生成指令 → 回填结果")

# 创建四个功能标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 1. 数据导入", 
    "🏷️ 2. 评价库配置", 
    "🤖 3. Prompt 生成器", 
    "📥 4. 结果回填与导出"
])

# ------------------------------------------
# Tab 1: 数据导入与清洗
# ------------------------------------------
with tab1:
    st.header("Step 1: 上传原始评论表")
    uploaded_file = st.file_uploader("支持 CSV / Excel", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            st.session_state.main_df = df
            st.success(f"成功加载 {len(df)} 行数据")
            
            with st.expander("查看原始数据预览", expanded=True):
                st.dataframe(df.head(3))

            st.markdown("---")
            st.subheader("🔧 字段映射 (告诉程序哪列是哪列)")
            
            all_cols = df.columns.tolist()
            c1, c2, c3, c4 = st.columns(4)
            
            # 智能预选列名
            idx_rating = all_cols.index('rating') if 'rating' in all_cols else 0
            idx_content = all_cols.index('content') if 'content' in all_cols else 0
            
            with c1:
                col_rating = st.selectbox("星级 (Rating) *必选", all_cols, index=idx_rating)
            with c2:
                col_title = st.selectbox("标题 (Title)", ["--忽略--"] + all_cols)
            with c3:
                col_content = st.selectbox("内容 (Content) *必选", all_cols, index=idx_content)
            with c4:
                col_trans = st.selectbox("翻译 (Translation)", ["--忽略--"] + all_cols)

            col_id_opt = st.selectbox("唯一ID (Review ID)", ["-- 自动生成 UUID (推荐) --"] + all_cols)

            if st.button("开始标准化处理", type="primary"):
                # 1. 复制副本
                norm_df = df.copy()
                
                # 2. 处理 ID
                if col_id_opt.startswith("--"):
                    # 生成8位UUID
                    norm_df['sys_uuid'] = [str(uuid.uuid4())[:8] for _ in range(len(norm_df))]
                    # 同时回写到主表，方便后续合并
                    st.session_state.main_df['sys_uuid'] = norm_df['sys_uuid'] 
                    target_id_col = 'sys_uuid'
                else:
                    # 强制转为字符串防止匹配错误
                    norm_df[col_id_opt] = norm_df[col_id_opt].astype(str)
                    target_id_col = col_id_opt

                # 3. 处理星级 (清洗非数字字符)
                norm_df['rating_std'] = pd.to_numeric(norm_df[col_rating], errors='coerce').fillna(0).astype(int)

                # 4. 拼接文本
                def combine_text(row):
                    parts = []
                    if col_title != "--忽略--" and pd.notna(row[col_title]):
                        parts.append(f"Title: {row[col_title]}")
                    if pd.notna(row[col_content]):
                        parts.append(f"Content: {row[col_content]}")
                    if col_trans != "--忽略--" and pd.notna(row[col_trans]):
                        parts.append(f"Trans: {row[col_trans]}")
                    return "\n".join(parts)

                norm_df['text_combined'] = norm_df.apply(combine_text, axis=1)

                # 5. 保存标准化结果到 Session
                st.session_state.normalized_df = norm_df[[target_id_col, 'rating_std', 'text_combined']].rename(
                    columns={target_id_col: 'id', 'rating_std': 'rating', 'text_combined': 'text'}
                )
                
                st.success("✅ 数据标准化完成！已生成标准中间表。请前往 Step 2。")
                st.dataframe(st.session_state.normalized_df.head())

# ------------------------------------------
# Tab 2: 评价库配置
# ------------------------------------------
with tab2:
    st.header("Step 2: 导入标签库规则")
    st.info("上传表头说明：必须包含 `label` (标签名) 和 `polarity` (positive/negative) 两列")
    
    tag_file = st.file_uploader("上传标签库 Excel/CSV", type=['csv', 'xlsx'])
    
    if tag_file:
        tag_df = load_file(tag_file)
        if tag_df is not None:
            c1, c2 = st.columns(2)
            lbl_col = c1.selectbox("选择标签列", tag_df.columns)
            pol_col = c2.selectbox("选择极性列", tag_df.columns)
            
            if st.button("解析标签库"):
                # 统一转小写进行匹配
                tag_df['pol_lower'] = tag_df[pol_col].astype(str).str.lower()
                
                # 提取好评/差评
                pos_list = tag_df[tag_df['pol_lower'].str.contains('pos|good|好|正')][lbl_col].dropna().unique().tolist()
                neg_list = tag_df[tag_df['pol_lower'].str.contains('neg|bad|差|负')][lbl_col].dropna().unique().tolist()
                
                st.session_state.tag_config = {
                    "pos": pos_list,
                    "neg": neg_list,
                    "all": list(set(pos_list + neg_list))
                }
                
                st.success("✅ 标签库加载成功！")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("好评标签数", len(pos_list))
                col_res2.metric("差评标签数", len(neg_list))
                
                with st.expander("查看解析后的列表"):
                    st.write("**Positive Tags:**", pos_list)
                    st.write("**Negative Tags:**", neg_list)

# ------------------------------------------
# Tab 3: Prompt 生成器
# ------------------------------------------
with tab3:
    st.header("Step 3: 生成分批指令")
    
    if st.session_state.normalized_df is None:
        st.warning("⚠️ 请先在 Step 1 完成数据标准化")
        st.stop()
    if not st.session_state.tag_config['all']:
        st.warning("⚠️ 请先在 Step 2 加载标签库")
        st.stop()
        
    # --- 配置区域 ---
    with st.container():
        c1, c2 = st.columns(2)
        batch_size = c1.number_input("每批次评论条数 (防止模型截断)", min_value=10, max_value=200, value=30)
        target_group = c2.selectbox("处理目标", ["自动处理所有星级", "仅 1-3 星 (差评)", "仅 4 星 (摇摆)", "仅 5 星 (好评)"])

    # --- 核心 Prompt 模板构建逻辑 ---
    def build_prompt(data_chunk, rating_mode):
        """
        data_chunk: JSON list of reviews
        rating_mode: '1-3', '4', '5'
        """
        # 获取标签
        pos_tags = json.dumps(st.session_state.tag_config['pos'], ensure_ascii=False)
        neg_tags = json.dumps(st.session_state.tag_config['neg'], ensure_ascii=False)
        
        # 基础系统指令
        sys_prompt = """## Role
You are an expert e-commerce review classifier.
## Output Format
Strictly valid JSON list: [{"id": "...", "label": "..."}]
Do not add any markdown blocks or explanations outside the JSON.
## Constraints
1. Only use tags from the provided lists.
2. If no tag fits, return empty string for label."""

        # 动态任务指令 (核心逻辑)
        if rating_mode == '1-3':
            task_prompt = f"""## Task (Negative Focus)
These are low-rated reviews (1-3 stars).
Please select the best match from this **NEGATIVE TAG LIST**:
{neg_tags}"""
        elif rating_mode == '5':
            task_prompt = f"""## Task (Positive Focus)
These are high-rated reviews (5 stars).
Please select the best match from this **POSITIVE TAG LIST**:
{pos_tags}"""
        else: # 4 Stars
            task_prompt = f"""## Task (Critical Analysis)
These are 4-star reviews. They are tricky.
**Rule 1**: First check for ANY complaints. Prioritize this **NEGATIVE TAG LIST**:
{neg_tags}
**Rule 2**: If absolutely no complaints, check this **POSITIVE TAG LIST**:
{pos_tags}
**Rule 3**: Negative tags have HIGHER PRIORITY than positive ones."""

        # 组装
        payload = json.dumps(data_chunk, ensure_ascii=False, indent=2)
        return f"{sys_prompt}\n\n{task_prompt}\n\n## Data Payload\n{payload}"

    if st.button("🚀 生成 Prompt 批次", type="primary"):
        df = st.session_state.normalized_df
        batches = []
        
        # 定义分组策略
        groups = {}
        if target_group in ["自动处理所有星级", "仅 1-3 星 (差评)"]:
            groups['1-3'] = df[df['rating'] <= 3]
        if target_group in ["自动处理所有星级", "仅 4 星 (摇摆)"]:
            groups['4'] = df[df['rating'] == 4]
        if target_group in ["自动处理所有星级", "仅 5 星 (好评)"]:
            groups['5'] = df[df['rating'] == 5]
            
        # 循环切片
        for g_name, g_df in groups.items():
            if g_df.empty: continue
            
            # 转字典列表
            records = g_df.to_dict(orient='records')
            
            # 切分
            for i in range(0, len(records), batch_size):
                chunk = records[i:i+batch_size]
                prompt_text = build_prompt(chunk, g_name)
                
                batches.append({
                    "name": f"[{g_name}星] 第 {i//batch_size + 1} 批 (共{len(chunk)}条)",
                    "prompt": prompt_text,
                    "count": len(chunk)
                })
        
        st.session_state.generated_batches = batches
        st.success(f"已生成 {len(batches)} 个任务包！")

    # --- 展示批次卡片 ---
    if st.session_state.generated_batches:
        for idx, batch in enumerate(st.session_state.generated_batches):
            with st.expander(f"📦 {batch['name']}", expanded=(idx==0)):
                st.text_area("Prompt (点击右上角复制)", value=batch['prompt'], height=200, key=f"b_{idx}")
                st.caption("👆 全选复制上面的内容，发送给 ChatGPT / Claude / DeepSeek")

# ------------------------------------------
# Tab 4: 结果回填
# ------------------------------------------
with tab4:
    st.header("Step 4: 结果回填与合并")
    
    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 1. 粘贴 LLM 返回的 JSON")
        json_input = st.text_area("在此粘贴...", height=300, placeholder='[{"id":"...", "label":"..."}, ...]')
        
        if st.button("解析并校验"):
            data = safe_json_parse(json_input)
            if data:
                res_df = pd.DataFrame(data)
                
                # 基础校验
                if 'id' not in res_df.columns or 'label' not in res_df.columns:
                    st.error("❌ 格式错误：JSON 必须包含 'id' 和 'label' 字段")
                else:
                    # 标签合法性校验
                    valid_tags = set(st.session_state.tag_config['all'])
                    # 如果还没导标签库，暂时跳过校验
                    if not valid_tags:
                        res_df['is_valid'] = True
                    else:
                        res_df['is_valid'] = res_df['label'].apply(
                            lambda x: x in valid_tags or x == "" or x is None
                        )
                    
                    invalid_count = len(res_df[~res_df['is_valid']])
                    
                    if invalid_count > 0:
                        st.warning(f"⚠️ 发现 {invalid_count} 个非法标签（不在库内），将标记为 INVALID_TAG")
                        res_df.loc[~res_df['is_valid'], 'label'] = "INVALID_TAG"
                    else:
                        st.success("✅ 所有标签校验通过！")
                    
                    # 存入 Session 暂存以便下载
                    st.session_state.temp_result_df = res_df
            else:
                st.error("无法解析，请检查是否完整复制了 [ ... ]")

    with col_preview:
        st.markdown("### 2. 合并回主表")
        if 'temp_result_df' in st.session_state:
            res_df = st.session_state.temp_result_df
            st.dataframe(res_df)
            
            if st.button("🔄 确认合并到主表", type="primary"):
                # 准备主表
                main = st.session_state.main_df
                
                # 确定主表的 ID 列
                # 如果 Step 1 生成了 sys_uuid，用它；否则用用户指定的列
                if 'sys_uuid' in main.columns:
                    join_key = 'sys_uuid'
                elif 'id' in st.session_state.normalized_df.columns:
                    # 这种情况比较复杂，简单起见，我们在 Step 1 已经把 sys_uuid 写入 main 了
                    join_key = 'sys_uuid' 
                else:
                    # 兜底：假设用户第一步选了 ID 列，我们需要找回那个列名
                    # 这里为了代码健壮性，建议强依赖 Step 1 的 uuid
                    st.error("无法定位主表 ID，请重新在 Step 1 生成 UUID")
                    st.stop()

                # 创建字典映射
                id_map = dict(zip(res_df['id'], res_df['label']))
                
                # 创建新列名 (防止覆盖)
                new_col = 'AI_Label'
                if new_col not in main.columns:
                    main[new_col] = None
                
                # 更新逻辑
                def update_row(row):
                    rid = str(row[join_key])
                    if rid in id_map:
                        return id_map[rid]
                    return row[new_col] # 保持原样

                main[new_col] = main.apply(update_row, axis=1)
                st.session_state.main_df = main
                st.success(f"已成功更新 {len(res_df)} 条数据！")

    st.markdown("---")
    st.header("📥 下载最终表格")
    
    if st.session_state.main_df is not None:
        final_df = st.session_state.main_df
        
        # CSV 下载
        csv_data = final_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "下载 CSV 格式",
            data=csv_data,
            file_name="tagged_reviews_final.csv",
            mime="text/csv"
        )
        
        # Excel 下载 (需 openpyxl)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        st.download_button(
            "下载 Excel 格式",
            data=buffer.getvalue(),
            file_name="tagged_reviews_final.xlsx",
            mime="application/vnd.ms-excel"
        )