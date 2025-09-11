
import requests
import json
from openai import OpenAI
from agents import function_tool
from collections import defaultdict
import time
from datetime import datetime, timedelta
from pydantic import BaseModel
import os
from google.cloud import secretmanager

# 全局 session，避免重複創建
_session = requests.Session()
_session.timeout = 30

# 全局 OpenAI client cache，避免重複創建
_openai_clients = {}

def get_openai_client(media_name=None):
    """
    獲取 OpenAI 客戶端，優先使用已驗證的環境變數中的 API key
    
    Args:
        media_name (str): 媒體名稱（主要用於快取識別）
        
    Returns:
        OpenAI: OpenAI 客戶端實例
    """
    cache_key = media_name or "default"
    
    # 檢查快取中是否已有客戶端
    if cache_key in _openai_clients:
        return _openai_clients[cache_key]
    
    # 優先使用環境變數中的 API key（已經在 set_openai_api_key 中驗證過）
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        # 如果環境變數中沒有，嘗試載入 .env 檔案
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        except ImportError:
            pass
    
    if not api_key:
        raise ValueError(f"No OpenAI API key available. Please ensure set_openai_api_key() was called successfully.")
    
    # 建立並快取客戶端
    client = OpenAI(api_key=api_key)
    _openai_clients[cache_key] = client
    
    print(f"Created OpenAI client using verified API key")
    return client

def get_media_apikey_from_secret_manager(media_name: str):
    try:
        # Secret Manager 中的 secret 名稱
        secret_name = f"Jiming-ext-{media_name}"
        # 建立 Secret Manager 客戶端
        client = secretmanager.SecretManagerServiceClient()
        
        # 構建 secret 的完整路徑
        name = f"projects/medialab-356306/secrets/{secret_name}/versions/latest"
        
        # 獲取 secret
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        print(secret_value[-5:])
        return secret_value

    except Exception as e:
        print(f"Error getting secret for {media_name}: {e}")
        return None

def set_openai_api_key(media_name=None):
    """Set OpenAI API key based on media_name"""
    if media_name:
        # 使用 Secret Manager 獲取 OpenAI API key
        api_key = get_media_apikey_from_secret_manager(media_name)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"Using OpenAI API key for {media_name}")
            return True
        else:
            # 如果 Secret Manager 失敗，回退到預設值
            print(f"Failed to get OpenAI API key for {media_name}, falling back to default")
    
    # 使用預設的 .env 配置
    if not os.getenv("OPENAI_API_KEY"):
        # 載入 .env 檔案
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: No OpenAI API key found in environment")
            return False
    
    print("Using default OpenAI API key from .env")
    return True

def get_editor_rules(media_name):
    with open("utils/media_file_vector_map.json", "r", encoding="utf-8") as f:
        media_file_vector_map = json.load(f)
    vector_store_id = media_file_vector_map[media_name]["vector_store_id"]
    # print(type(vector_store_id))
    # print(f"vector_store_id: {vector_store_id}")
    return vector_store_id

@function_tool
def think(thought: str):
    """Use this tool for internal reasoning and analysis. This tool allows you to think through complex problems step by step without producing output for the user. Use it when you need to analyze, reason, or organize your thoughts before providing your final response.

    Args:
        thought: Your internal reasoning or analysis that helps you think through the problem.

    """
    return "思考完成"  # 簡單確認，不暴露思考內容

@function_tool
def detect_gov_news(input_text: str, media_name: str = None) -> str:
    """判斷是否為政府新聞稿、公關稿
    
    Args:
        input_text: 要判斷的文本內容
        media_name: 媒體名稱，用於獲取對應的 API key
    """
    client = get_openai_client(media_name)
    
    print(f"⚙️ Tool Used: {detect_gov_news.__name__}")

    prompt = """
    <role>你是新聞編輯，負責判斷政府機關新聞稿、公關稿。</role>
    <instruction>
    新聞稿、公關稿特徵：
    1. 官方語氣正式且嚴謹
    2. 內容只有政府政策宣導、國家立場說明、公共安全警示及政府措施的介紹
    3. 整篇文章只有單一部門的聲明、澄清、說明
    4. 提及特定政府網頁（網址包含".gov.tw"）、政府專線（通常是4位數字或0800-開頭的電話）、服務據點等等。 例如：如有傳染病相關疑問，可至疾管署全球資訊網(https://www.cdc.gov.tw)，或撥打免付費防疫專線1922(或0800-001922)洽詢。
    5. 單一活動、產品的介紹文章，例如：旅遊展、航空公司新增航線、觀光景點限定活動等等

    非公關稿的政府新聞特徵：
    1. 有提及媒體名稱、記者，例如：中央社記者求證某某某、某某某接受媒體聯訪
    2. 消息來源是「記者會」、「政府發言人表示...」
    </instruction>

    <output_format>
    output as json format
    {
        "is_gov_news": True or False,
        "reason": "..."
    }
    </output_format>
    """
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=f"{prompt}\n\n<input_text>{input_text}</input_text>",
            response_format={"type": "json_object"}
        )

        content_json = json.loads(response.content)
        is_gov_news = content_json.get("is_gov_news", False)
        reason = content_json.get("reason", "")

        return f"is_gov_news: {is_gov_news}, reason: {reason}"
        
    except Exception as e:
        return f"is_gov_news: False, reason: {str(e)}"

## 譯名檔檢查
def _openai_replace(text, name_translation_map, media_name=None):
    """OpenAI 譯名替換"""
    print(f"==> 譯名替換...")
    client = get_openai_client(media_name)
    
    prompt = """
    請根據提供的譯名對照表，替換文本中括號前的中文人名。

    處理規則：
    - 找到文本中 "中文譯名（外文名）"或 "外文名"
    - 如果外文名在譯名對照表中，將括號前的中文名替換為對照表中的標準譯名，或者加上中文譯名
    - 保持括號內的外文名不變
    - 人名第一次出現改成中文譯名（外文名），第二次以後只需要留下中文譯名
    - 輸出只能用全形括號（ ）

    請直接返回替換後的完整文本，不需要markdown 跟 <text></text>標記
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"<text>{text}</text><譯名檔>{name_translation_map}</譯名檔>"}
            ],
            timeout=60
        )
        
        content = response.choices[0].message.content
        return content.strip() if content else text
            
    except Exception as e:
        print(f">>> [Error] 譯名替換失敗: {e}")
        return text

def process_translated_name(text, media_name=None, mode="sentence"):
    """
    一個函數搞定所有譯名處理
    
    整合流程：
    1. API 取得譯名檔 → 2. 去重複 → 3. 改文章
    
    Args:
        text (str): 要處理的文章內容
        media_name (str): 媒體名稱，用於獲取對應的 API key
        mode (str): 檢查模式，默認為 "sentence"
        
    Returns:
        tuple: (譯名對照表, 處理後的文章)
    """
    url = "https://wordpool-namecheck-1007110536706.asia-east1.run.app"
    
    print(f"==> 開始處理文章譯名...")
    start_time = time.time()    
    
    # 步驟 1: 調用 API 取得譯名
    try:
        response = _session.post(url, json={"content": text, "mode": mode})
        response.raise_for_status()
        response_data = response.json()
        
        if response_data['Result'] != 'Y':
            print(f">>> [Info] API 未找到譯名")
            end_time = time.time()
            print(f"==> 譯名處理耗時 {end_time - start_time:.2f} 秒")
            return {}, text
            
        result_data = response_data['ResultData']
        
    except Exception as e:
        print(f">>> [Error] API 調用失敗: {e}")
        end_time = time.time()
        print(f"==> 譯名處理耗時 {end_time - start_time:.2f} 秒")
        return {}, text
    
    # 步驟 2: 提取並去重複譯名
    name_translation_map = {}
    standard_names = defaultdict(list)
    
    # 提取譯名
    for item in result_data:
        person_name = item.get('Person')
        translations = item.get('Translations', [])
        
        if person_name and translations:
            keyword = translations[0].get('Keyword', '')
            if '|' in keyword:
                name_translation_map[person_name] = keyword
                # 同時記錄標準譯名以便去重複
                standard_name = keyword.split('|', 1)[0]
                standard_names[standard_name].append((person_name, keyword))
                print(f">>> [Info] 人名: {person_name} -> 中央社譯名檔：{standard_name} ({keyword})")
            else:
                print(f">>> [Warning] 人名 {person_name} 的 Keyword 格式異常: {keyword}")
    
    if not name_translation_map:
        print(f">>> [Info] 沒有有效譯名")
        end_time = time.time()
        print(f"==> 譯名處理耗時 {end_time - start_time:.2f} 秒")
        return {}, text
    
    # 去重複：每個標準譯名只保留一個條目
    deduplicated_map = {}
    for standard_name, entries in standard_names.items():
        if len(entries) == 1:
            key, value = entries[0]
            deduplicated_map[key] = value
        else:
            # 優先選擇中文 key
            chinese_entry = next(((k, v) for k, v in entries 
                                if any('\u4e00' <= char <= '\u9fff' for char in k)), 
                               entries[0])
            deduplicated_map[chinese_entry[0]] = chinese_entry[1]
            print(f">>> [Info] 去重複: 保留 {chinese_entry[0]} -> {chinese_entry[1]}")
    
    print(f">>> [Info] 找到 {len(deduplicated_map)} 個譯名")
    
    # 步驟 3: 用 OpenAI 修改文章
    if deduplicated_map:
        processed_text = _openai_replace(text, deduplicated_map, media_name)
        end_time = time.time()
        print(f"==> 文章內譯名替換完成，耗時 {end_time - start_time:.2f} 秒")
        return deduplicated_map, processed_text
    else:
        end_time = time.time()
        print(f"==> 文章內譯名替換完成，耗時 {end_time - start_time:.2f} 秒")
        return {}, text


### 查核點
async def get_check_points(text, media_name=None):
    import httpx
    input = {
        "text": text,
        "media_name": media_name
    }
    url = "https://get-check-points-1007110536706.asia-east1.run.app"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # 設置 30 秒超時
            response = await client.post(url, json=input)
    except httpx.ReadTimeout:
        print(f">>> [Warning] get_check_points API 超時，返回空結果")
        return {
            "Result": "N",
            "ResultData": {"check_points": None},
            "Message": "API超時"
        }
    if response.status_code == 200:
        response_json = response.json()
        check_points = []
        if response_json.get("Result") == "Y" and "ResultData" in response_json:
            check_points = response_json["ResultData"].get("check_points", [])

            if check_points:
                print(f">>> [Info] 成功取得查核點")
                return {
                    "Result": "Y",
                    "ResultData": {"check_points": check_points},
                    "Message": "API成功回傳結果"
                }
            else:
                print(f">>> [Info] 查核點為空")
                return {
                    "Result": "N",
                    "ResultData": {"check_points": None},
                    "Message": "查核點為空"
                }
    else:
        print(f">>> [Error] 取得查核點時發生錯誤：{response.status_code}")
        return {
            "Result": "N",
            "ResultData": {"check_points": None},
            "Message": "API回傳None"
        }

@function_tool    
def find_related_news(text: str, media_name: str = None):
    """
    從relatednews API 取得一周內的相關新聞
    Args:
        text (str): 新聞文章內容
        media_name (str): 媒體名稱

    Returns:
        list: 相關新聞的 標題 日期 內文 連結
    """
    url = "https://relatednews-1007110536706.asia-east1.run.app"

    payload = {
        "inputSTR": text,
        "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"), # 7天前
        "end_date": datetime.now().strftime("%Y-%m-%d"), # 今天
        "media_name": media_name,
        "count": 20
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()  # 檢查 HTTP 狀態碼
        result = response.json()
        
        # 檢查 result 是否為 None 或空
        if result is None:
            print(f"relatednews API 錯誤: API 返回 None")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"relatednews API 請求錯誤: {e}")
        return None
    except ValueError as e:
        print(f"relatednews API JSON 解析錯誤: {e}")
        return None

    # 處理 API 回應
    if result.get('Result') == 'Y' and 'ResultData' in result:
        similar_docs = result['ResultData'].get('similar_docs', [])

        # 處理相關新聞
        related_news = []
        for doc in similar_docs[:3]:
            data = {
                "title": doc.get('title', ''),
                "date": doc.get('dt', '').replace('/', '-'),
                "article": doc.get('article', ''),
                "url": doc.get('url', '')
            }
            related_news.append(data)

        return related_news
    else:
        print(f"relatednews API 錯誤: {result.get('Message', '未知錯誤')}")
        return None
    

def get_rewrite_api_result(media_name, input_text, edit_advice):
    input_dict = {
        "media_name": media_name,
        "text": input_text,
        "edit_advice": edit_advice,
    }

    try:
        # url = "https://cna-agent-rewrite-1007110536706.asia-east1.run.app"
        url = "https://dev-agent-rewrite-1007110536706.asia-east1.run.app"
        response = requests.post(url, json=input_dict, timeout=540)
        
        if response.status_code == 200:
            response_json = response.json()
            # print(">>> rewrite api 原始結果")
            # print(response_json)
            
            if response_json.get("Result") == "Y":
                return {
                    "Result": "Y",
                    "ResultData": response_json.get("ResultData", {}),
                    "Message": "API成功回傳結果"
                }
            else:
                print(f">>> [Error] 改稿 API 回傳失敗: {response_json.get('Message', '未知錯誤')}")
                return {
                    "Result": "N",
                    "ResultData": {},
                    "Message": response_json.get('Message', '未知錯誤')
                }
        else:
            # 嘗試解析 JSON 來取得正確的錯誤訊息
            try:
                error_json = response.json()
                error_message = error_json.get('Message', '未知錯誤')
                print(f">>> [Error] 改稿 API HTTP 錯誤: {response.status_code} - {error_message}")
                return {
                    "Result": "N",
                    "ResultData": {},
                    "Message": error_message
                }
            except:
                # 如果無法解析 JSON，使用原始文字
                print(f">>> [Error] 改稿 API HTTP 錯誤: {response.status_code} - {response.text}")
                return {
                    "Result": "N",
                    "ResultData": {},
                    "Message": "API回傳錯誤"
                }
            
    except Exception as e:
        print(f">>> [Error] 改稿 API 呼叫異常: {str(e)}")
        return None


### 改稿

# 1. 合併不同的prompt
def generate_rewrite_prompt(edit_role: str, edit_mode: str, output_format: str, background_summary: str):
    """
    生成改稿 prompt
    
    Args:
        edit_role: 新聞編輯角色 (gov_roles, foreign_roles, normal_roles)
        edit_mode: 新聞改寫模式 (gov_rewrite_rules, foreign_rewrite_rules, less_mode, medium_mode, full_mode)  
        output_format: 輸出格式 (full_output_format, normal_output_format)
    
    Returns:
        str: 完整的改稿指令
    """
    import prompts
    
    # 映射前端簡短名稱到完整變數名
    mode_mapping = {
        'less': 'less_mode',
        'medium': 'medium_mode', 
        'full': 'full_mode'
    }
    
    # 如果 edit_mode 在映射表中，使用映射後的名稱
    actual_edit_mode = mode_mapping.get(edit_mode, edit_mode)
    
    # 從 prompts 模組取得對應變數
    roles = getattr(prompts, edit_role)
    mode_instruction = getattr(prompts, actual_edit_mode) 
    output_format_content = getattr(prompts, output_format)
    
    return f"""
    <role> {roles} </role>

    <task>
    1. 改稿
        - 請嚴格遵守以下改稿規則
            <rewrite rules>
            {mode_instruction}
            </rewrite rules>
        - 如果<編輯建議>有補充事件背景摘要，請參考<事件背景摘要>進行改稿，或者直接把<事件背景摘要>補充到文中適當的段落。
            <事件背景摘要>
            {background_summary}
            </事件背景摘要>

    2. 潤稿
        完成改稿後，請仔細檢查以下事項：
        1. 除非<原文>就有稿頭稿尾，某則改稿後文章內不要出現"（中央社華盛頓16日綜合外電報導）"、"（編輯：唐聲揚）" 這類型的稿頭、稿尾
        2. 禁止使用小標、分號、驚嘆號、問號
        3. 禁止主觀評論、猜測、總結
        4. 不要改動<原文>裡的separator，只改動content
        5. 出現「今天（2025年7月21日）」的格式，請直接改成「今天」或「21日」
    </task>

    <output format>
    {output_format_content}
    </output format>
    """

# 2. 文章分段落
def detect_paragraph_structure(text):
    """
    偵測文章的段落結構，以出現最多次的換行組合為主
    
    Returns:
        dict: 包含段落分隔符和結構資訊
    """
    import re
    from collections import Counter
    
    # 找出所有換行符號組合
    newline_patterns = re.findall(r'\n+', text)
    
    if not newline_patterns:
        # 沒有換行符號，整篇為一段
        return {
            'separator': '',
            'split_pattern': None,
            'pattern_counts': {},
            'most_common': 'no_newlines'
        }
    
    # 統計各種換行組合的出現次數
    pattern_counter = Counter(newline_patterns)
    
    # 取得最常見的換行模式
    most_common_pattern, most_common_count = pattern_counter.most_common(1)[0]
    
    # 直接使用最常見的換行模式
    separator = most_common_pattern
    split_pattern = re.escape(most_common_pattern)
    
    return {
        'separator': separator,
        'split_pattern': split_pattern,
        'pattern_counts': dict(pattern_counter),
        'most_common': most_common_pattern,
        'most_common_count': most_common_count
    }

def text_split(text, edit_mode=None):
    """
    根據 edit_mode 分割文章，返回簡單的字符串列表
    返回格式: [content1, content2, separator, content3, separator, ...]
    
    Args:
        text (str): 原始文章
        edit_mode (str): 改稿模式
        
    Returns:
        tuple: (分割後的字符串列表, 結構資訊)
    """
    import re
    
    # 移除時間置換的標記 <2025年6月20日>
    text_cleaned = re.sub(r'<[^>]*>', '', text)

    # 偵測段落結構
    structure = detect_paragraph_structure(text_cleaned)
    
    if structure['split_pattern'] is None:
        # 沒有換行符號，整篇為一個內容
        if edit_mode in ['less_mode', 'medium_mode']:
            # 按句子分割
            parts = re.split(r'([。！？：])', text_cleaned)
            elements = []
            current_sentence = ""
            
            for part in parts:
                current_sentence += part
                if part in ['。', '！', '？', '：']:
                    elements.append(current_sentence)
                    current_sentence = ""
            
            if current_sentence:
                elements.append(current_sentence)
            
            return elements, structure
        else:
            return [text], structure
    
    # 使用 split 配合捕獲組來保留分隔符
    split_pattern_with_capture = f'({structure["split_pattern"]})'
    parts = re.split(split_pattern_with_capture, text_cleaned)
    
    elements = []
    separator = structure['separator']
    
    for part in parts:
        if part == separator:
            # 這是分隔符
            elements.append(separator)
        elif part:  # 非空內容
            if edit_mode in ['less_mode', 'medium_mode']:
                # 在段落內按句子分割
                sentence_parts = re.split(r'([。！？：])', part)
                current_sentence = ""
                
                for sentence_part in sentence_parts:
                    current_sentence += sentence_part
                    if sentence_part in ['。', '！', '？', '：']:
                        if current_sentence.strip():
                            elements.append(current_sentence)
                        current_sentence = ""
                
                # 處理沒有標點符號結尾的內容
                if current_sentence.strip():
                    elements.append(current_sentence)
            else:
                # 按段落處理
                elements.append(part)
    
    return elements, structure

# 3. 合併改稿結果 + 反白標記
def formatted_output(rewrite_result, source_url_list, edit_mode=None):
    """
    格式化改稿輸出結果，生成帶有標記的文章
    
    Args:
        rewrite_result: rewrite_process 的輸出結果 (舊版格式字典)
        source_url_list (list): 參考資料 URL 列表
        edit_mode (str): 改稿模式
        
    Returns:
        dict: 格式化後的結果
    """
    if rewrite_result.get("Result") != "Y":
        return rewrite_result
    
    # 從 rewrite_result 提取數據 (舊版格式)
    result_data = rewrite_result["ResultData"]
    original_list = result_data["original_list"]
    edited_list = result_data["edited_list"]
    reason_list = result_data["reason_list"]
    references_list = result_data["references_list"]
    elements = result_data["elements"]
    structure = result_data["structure"]
    
    # 建立改稿對照表
    edit_map = {}
    for i in range(len(original_list)):
        edit_map[original_list[i]] = {
            'edited': edited_list[i],
            'reason': reason_list[i],
            'references_list': references_list[i] if i < len(references_list) else None
        }
    
    # 重組文章，保持原始順序和結構
    new_elements = []
    applied_edits = []
    
    # 使用傳入的 URL 列表
    url_list = source_url_list
    
    for element in elements:
        if element == structure['separator']:
            # 保留分隔符
            new_elements.append(element)
        elif element in edit_map:
            # 找到對應的改稿
            edit_info = edit_map[element]
            marked_content = f"/@{edit_info['edited']}@/"
            
            # 如果有參考資料，只附上數字標註
            if edit_info['references_list']:
                ref_numbers = []
                for ref_index in edit_info['references_list']:
                    ref_numbers.append(f"<{ref_index + 1}>")
                
                if ref_numbers:
                    marked_content += " " + "".join(ref_numbers)
            
            new_elements.append(marked_content)
            applied_edits.append({
                'original': element,
                'edited': edit_info['edited'],
                'reason': edit_info['reason']
            })
        else:
            # 沒有改稿的內容，保持原樣
            new_elements.append(element)
    
    # 生成最終結果
    marked_text = ''.join(new_elements)
    
    formatted_result = {
        "edit_mode": edit_mode or "rewrite", 
        "edit_count": len(applied_edits),
        "edit_list": applied_edits,
        "marked_text": marked_text,
        "source_url_list": url_list
    }
    
    return formatted_result

async def formatted_output_streaming(rewrite_result, source_url_list, edit_mode=None):
    """
    格式化改稿輸出結果的 streaming 版本，逐字符傳送 marked_text
    
    Args:
        rewrite_result: rewrite_process 的輸出結果 (舊版格式字典)
        source_url_list (list): 參考資料 URL 列表
        edit_mode (str): 改稿模式
        
    Yields:
        tuple: ("streaming", char) 或 ("completed", formatted_result)
    """
    if rewrite_result.get("Result") != "Y":
        yield ("completed", rewrite_result)
        return
    
    # 從 rewrite_result 提取數據 (舊版格式)
    result_data = rewrite_result["ResultData"]
    original_list = result_data["original_list"]
    edited_list = result_data["edited_list"]
    reason_list = result_data["reason_list"]
    references_list = result_data["references_list"]
    elements = result_data["elements"]
    structure = result_data["structure"]
    
    # 建立改稿對照表
    edit_map = {}
    for i in range(len(original_list)):
        edit_map[original_list[i]] = {
            'edited': edited_list[i],
            'reason': reason_list[i],
            'references_list': references_list[i] if i < len(references_list) else None
        }
    
    # 重組文章，保持原始順序和結構
    new_elements = []
    applied_edits = []
    
    # 使用傳入的 URL 列表
    url_list = source_url_list
    
    for element in elements:
        if element == structure['separator']:
            # 保留分隔符
            new_elements.append(element)
        elif element in edit_map:
            # 找到對應的改稿
            edit_info = edit_map[element]
            marked_content = f"/@{edit_info['edited']}@/"
            
            # 如果有參考資料，只附上數字標註
            if edit_info['references_list']:
                ref_numbers = []
                for ref_index in edit_info['references_list']:
                    ref_numbers.append(f"<{ref_index + 1}>")
                
                if ref_numbers:
                    marked_content += " " + "".join(ref_numbers)
            
            new_elements.append(marked_content)
            applied_edits.append({
                'original': element,
                'edited': edit_info['edited'],
                'reason': edit_info['reason']
            })
        else:
            # 沒有改稿的內容，保持原樣
            new_elements.append(element)
    
    # 生成最終的 marked_text
    marked_text = ''.join(new_elements)
    
    # streaming 傳送 marked_text 的每個字符
    for char in marked_text:
        yield ("streaming", char)
        # 添加小延遲以模擬真實的串流效果（可選）
        import asyncio
        await asyncio.sleep(0.001)  # 1ms 延遲
    
    # 生成完整的結果（供後端記錄用）
    formatted_result = {
        "edit_mode": edit_mode or "rewrite", 
        "edit_count": len(applied_edits),
        "edit_list": applied_edits,
        "marked_text": marked_text,
        "source_url_list": url_list
    }
    
    yield ("completed", formatted_result)

