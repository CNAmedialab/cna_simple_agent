
from agents import Agent, Runner, FileSearchTool, ModelSettings
from datetime import datetime, timedelta
from token_manager import get_token_manager
from typing import Dict
from pydantic import BaseModel
from func import *
from prompts import *
import asyncio
import time
import os
from dotenv import load_dotenv

# Get today's date
today_date = datetime.now().strftime("%Y-%m-%d")

# Initialize token manager
token_manager = get_token_manager()

# Token-aware agent runner
async def run_agent_with_token_tracking(agent: Agent, 
                                      input_data: str, 
                                      context=None,
                                      max_retries: int = 2) -> object:
    """
    Run agent with token usage tracking and overflow protection
    
    Args:
        agent: The agent to run
        input_data: Input data for the agent
        context: Optional context
        max_retries: Maximum retries if token limit exceeded
        
    Returns:
        Agent result with token tracking
    """
    for attempt in range(max_retries + 1):
        try:
            # Estimate input tokens
            estimated_input_tokens = token_manager.estimate_tokens_for_agent_input(
                instructions=agent.instructions,
                user_input=input_data,
                model=agent.model
            )
            
            # Check token limits
            is_within_limit, available_tokens, model_limit = token_manager.check_token_limit(
                estimated_input_tokens, agent.model
            )
            
            if not is_within_limit:
                print(f"⚠️  Token 限制警告 - {agent.name} ({agent.model})")
                print(f"   預估 input tokens: {estimated_input_tokens:,}")
                print(f"   模型限制: {model_limit:,}")
                print(f"   可用 tokens: {available_tokens:,}")
                
                if attempt < max_retries:
                    # Try to truncate input
                    if isinstance(input_data, str):
                        truncated_input = token_manager.truncate_text_to_fit(
                            input_data, agent.model, reserved_tokens=2000
                        )
                        if truncated_input != input_data:
                            print(f"   第 {attempt + 1} 次嘗試: 截斷輸入文本")
                            input_data = truncated_input
                            continue
                
                raise Exception(f"Token 超出限制: 需要 {estimated_input_tokens:,} tokens，但模型 {agent.model} 限制為 {model_limit:,}")
            
            # Run the agent
            print(f"🤖 執行 {agent.name} - 預估 input tokens: {estimated_input_tokens:,}")
            
            if context:
                result = await Runner.run(agent, input_data, context=context)
            else:
                result = await Runner.run(agent, input_data)
            
            # Estimate output tokens and log usage
            output_text = str(result.final_output) if hasattr(result, 'final_output') else str(result)
            output_tokens = token_manager.count_tokens(output_text, agent.model)
            
            token_manager.log_usage(
                model=agent.model,
                input_tokens=estimated_input_tokens,
                output_tokens=output_tokens,
                agent_name=agent.name
            )
            
            return result
            
        except Exception as e:
            if attempt == max_retries:
                print(f"❌ {agent.name} 執行失敗 (已重試 {max_retries} 次): {str(e)}")
                raise
            else:
                print(f"⚠️  {agent.name} 第 {attempt + 1} 次嘗試失敗，重試中...")
                continue

# Token-aware streamed agent runner
async def run_agent_with_token_tracking_streamed(agent: Agent, 
                                               input_data: str, 
                                               context=None,
                                               max_retries: int = 2):
    """
    Run agent with token usage tracking and overflow protection for streamed responses
    
    Args:
        agent: The agent to run
        input_data: Input data for the agent
        context: Optional context
        max_retries: Maximum retries if token limit exceeded
        
    Returns:
        Streamed agent result with token tracking
    """
    for attempt in range(max_retries + 1):
        try:
            # Estimate input tokens
            estimated_input_tokens = token_manager.estimate_tokens_for_agent_input(
                instructions=agent.instructions,
                user_input=input_data,
                model=agent.model
            )
            
            # Check token limits
            is_within_limit, available_tokens, model_limit = token_manager.check_token_limit(
                estimated_input_tokens, agent.model
            )
            
            if not is_within_limit:
                print(f"⚠️  Token 限制警告 - {agent.name} ({agent.model})")
                print(f"   預估 input tokens: {estimated_input_tokens:,}")
                print(f"   模型限制: {model_limit:,}")
                print(f"   可用 tokens: {available_tokens:,}")
                
                if attempt < max_retries:
                    # Try to truncate input
                    if isinstance(input_data, str):
                        truncated_input = token_manager.truncate_text_to_fit(
                            input_data, agent.model, reserved_tokens=2000
                        )
                        if truncated_input != input_data:
                            print(f"   第 {attempt + 1} 次嘗試: 截斷輸入文本")
                            input_data = truncated_input
                            continue
                
                raise Exception(f"Token 超出限制: 需要 {estimated_input_tokens:,} tokens，但模型 {agent.model} 限制為 {model_limit:,}")
            
            # Run the agent with streaming
            print(f"🤖 執行 {agent.name} (串流模式) - 預估 input tokens: {estimated_input_tokens:,}")
            
            if context:
                result = Runner.run_streamed(agent, input_data, context=context)
            else:
                result = Runner.run_streamed(agent, input_data)
            
            # Create a wrapper to track output tokens
            
            class TokenTrackingStreamResult:
                def __init__(self, original_result, agent, estimated_input_tokens):
                    self.original_result = original_result
                    self.agent = agent
                    self.estimated_input_tokens = estimated_input_tokens
                    self.output_content = ""
                    self.final_output = None
                
                async def stream_events(self):
                    async for event in self.original_result.stream_events():
                        # Track streaming content for token counting
                        if hasattr(event, 'data') and hasattr(event.data, 'delta'):
                            self.output_content += event.data.delta
                        yield event
                    
                    # After streaming is complete, log token usage
                    self.final_output = self.original_result.final_output
                    output_tokens = token_manager.count_tokens(str(self.final_output), self.agent.model)
                    
                    token_manager.log_usage(
                        model=self.agent.model,
                        input_tokens=self.estimated_input_tokens,
                        output_tokens=output_tokens,
                        agent_name=self.agent.name
                    )
                    
                    print(f"✅ {self.agent.name} 完成 - input: {self.estimated_input_tokens:,} tokens, output: {output_tokens:,} tokens")
            
            return TokenTrackingStreamResult(result, agent, estimated_input_tokens)
            
        except Exception as e:
            if attempt == max_retries:
                print(f"❌ {agent.name} 執行失敗 (已重試 {max_retries} 次): {str(e)}")
                raise
            else:
                print(f"⚠️  {agent.name} 第 {attempt + 1} 次嘗試失敗，重試中...")
                continue

# Convenience functions for token management
def get_session_token_summary() -> str:
    """Get formatted token usage summary for current session"""
    return token_manager.get_session_summary()

def save_token_usage_log(filepath: str = "token_usage_log.json"):
    """Save token usage log to file"""
    token_manager.save_usage_log(filepath)
    print(f"💾 Token 使用記錄已保存至: {filepath}")

def check_model_token_limits(model: str) -> Dict[str, int]:
    """Get token limit information for a specific model"""
    return {
        "model": model,
        "max_tokens": token_manager.get_model_limit(model),
        "encoding": token_manager.MODEL_ENCODINGS.get(model, "cl100k_base")
    }

def estimate_text_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Estimate tokens for a given text and model"""
    return token_manager.count_tokens(text, model)


### 編輯準則搜尋Agent
async def edit_advice_process(media_name, input_text, news_map, yield_callback=None, edit_mode=None):
    print(f"==> 生成編輯建議...")
    start_time = time.time()

    vector_store_ids = get_editor_rules(media_name)

    edit_advice_agent = Agent(
        name="get_edit_advice",
        instructions=edit_advice_prompt,
        model="gpt-4.1",
        tools=[FileSearchTool(vector_store_ids=[vector_store_ids]), think],
        model_settings=ModelSettings(tool_choice="required")
    )

    user_input = f"使用FileSearch檢查新聞文章是否符合編輯守則的寫作規範。新聞文章：<article>{input_text}</article> 譯名檔：{news_map} 。使用者指定的編輯模式：{edit_mode}"
    
    # 使用串流模式來追蹤工具調用
    stream_result = await run_agent_with_token_tracking_streamed(edit_advice_agent, user_input)
    
    final_edit_advice = ""
    async for event in stream_result.stream_events():
        if event.type == "raw_response_event" and hasattr(event, 'data') and hasattr(event.data, 'type'):
            if event.data.type == "response.file_search_call.in_progress":
                print("🔧 工具調用: FileSearchTool")
            elif event.data.type == "response.function_call.start":
                print(f"🔧 工具調用: {event.data.name}")
            elif event.data.type == "response.output_text.delta":
                delta = getattr(event.data, 'delta', '')
                if delta:
                    final_edit_advice += delta
                    print(delta, end="", flush=True)
                    # 如果有回調函數，將 delta 傳遞給前端
                    if yield_callback:
                        await yield_callback("edit_advice_delta", delta)

    end_time = time.time()
    print(f"\n>>> 編輯建議耗時 {end_time - start_time:.2f} 秒")
    
    return final_edit_advice

async def edit_advice_process_streaming(media_name, input_text, news_map):
    """專為 streaming 設計的 edit_advice_process generator 版本"""
    print(f"==> 生成編輯建議...")
    start_time = time.time()

    vector_store_ids = get_editor_rules(media_name)

    edit_advice_agent = Agent(
        name="get_edit_advice",
        instructions=edit_advice_prompt,
        model="gpt-4.1",
        tools=[FileSearchTool(vector_store_ids=[vector_store_ids]), think],
        model_settings=ModelSettings(tool_choice="required")
    )

    user_input = f"使用FileSearch檢查新聞文章是否符合編輯守則的寫作規範。新聞文章：<article>{input_text}</article> 譯名檔：{news_map}"
    
    # 使用串流模式來追蹤工具調用
    stream_result = await run_agent_with_token_tracking_streamed(edit_advice_agent, user_input)
    
    final_edit_advice = ""
    async for event in stream_result.stream_events():
        if event.type == "raw_response_event" and hasattr(event, 'data') and hasattr(event.data, 'type'):
            if event.data.type == "response.file_search_call.in_progress":
                print("🔧 工具調用: FileSearchTool")
            elif event.data.type == "response.function_call.start":
                print(f"🔧 工具調用: {event.data.name}")
            elif event.data.type == "response.output_text.delta":
                delta = getattr(event.data, 'delta', '')
                if delta:
                    final_edit_advice += delta
                    print(delta, end="", flush=True)
                    # 直接 yield streaming delta
                    yield ("streaming", delta)

    end_time = time.time()
    print(f"\n>>> 編輯建議耗時 {end_time - start_time:.2f} 秒")
    
    # 最後 yield 完成結果
    yield ("completed", final_edit_advice)

### 背景摘要Agent
async def background_summary_process(input_text, media_name=None):

    print(f"==> 生成背景摘要...")
    start_time = time.time()

    class RelatedNews(BaseModel):
        title: str
        url: str

    class SummaryOutput(BaseModel):
        summary: str | None
        related_news: list[RelatedNews] | None

    # 動態生成 prompt，包含 media_name 指示
    dynamic_prompt = background_summary_prompt
    if media_name:
        dynamic_prompt += f"\n\n重要：使用find_related_news工具時，media_name參數請設為: {media_name}"

    background_summary_agent = Agent(
        name="background_summary",
        instructions=dynamic_prompt,
        model="gpt-4.1",
        tools=[find_related_news],
        model_settings=ModelSettings(tool_choice="find_related_news"),
        output_type=SummaryOutput
    )

    result = await run_agent_with_token_tracking_streamed(background_summary_agent, input_text)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event, 'data') and hasattr(event.data, 'type'):
            if event.data.type == "response.function_call.start":
                print(f"🔧 工具調用: {event.data.name}")

    end_time = time.time()
    print(f" >>> 背景摘要耗時 {end_time - start_time:.2f} 秒")
    
    return result.final_output

### 改稿Agent

# 1. 判斷改稿類型
async def classify_edit_mode(input_text: str, apponited_edit_mode: str = None) -> dict:
    """
    單一函數完成新聞類型分類
    """
    class news_type_output(BaseModel):
        edit_role: str  # "gov_roles", "foreign_roles", "normal_roles"
        edit_mode: str  # "gov_rewrite_rules", "foreign_rewrite_rules", "less_mode", "medium_mode", "full_mode"
        output_format: str  # "full_output_format", "normal_output_format"

    news_classifier = Agent(
        name="edit_mode_and_prompt",
        instructions="""
    <role>你是台灣的新聞編輯，快速判斷新聞類型並生成改稿指令</role>
    <task>
    1. 語言判斷：
        - 包含英文、日文、印尼文等外語內容 → edit_role="foreign_roles", edit_mode="foreign_rewrite_rules"
        - 繁體中文內容 → 繼續下一步

    2. 內容類型判斷（只在無法確定時使用工具）：
        - 明顯政府新聞稿特徵（官方用語、政策宣導、部會發言） → edit_role="gov_roles", edit_mode="gov_rewrite_rules"
        - 一般新聞報導 → edit_role="normal_roles", edit_mode="less_mode"
        - 不確定時才使用 detect_gov_news 工具

    3. 輸出格式判斷：
        - 如果 edit_role="gov_roles" or edit_role="foreign_roles"，output_format="full_output_format"
        - 如果 edit_role="normal_roles"，output_format="normal_output_format"
    </task>

    <final_output_format>
    {{
        "edit_role": "新聞編輯角色",
        "edit_mode": "新聞編輯模式: gov_rewrite_rules, foreign_rewrite_rules, less_mode, medium_mode, full_mode"
        "output_format": "新聞編輯格式: full_output_format, normal_output_format"
    }}
    </final_output_format>
        """,
        tools=[detect_gov_news],
        model="gpt-4.1",
        output_type=news_type_output
    )
    
    result = await run_agent_with_token_tracking(news_classifier, input_text)

    if apponited_edit_mode is not None:
        result.final_output.edit_mode = apponited_edit_mode
    
    classify_result =  {
        "edit_role": result.final_output.edit_role,
        "edit_mode": result.final_output.edit_mode,
        "output_format": result.final_output.output_format
    }
    return classify_result

# 2. 改稿
async def rewrite_process(input_text, edit_advice, edit_mode, prompt, source_url_list=None):
    start_time = time.time()
    print("==> 啟動改稿...")

    # 文章分段落
    elements, structure = text_split(input_text, edit_mode)
    paragraphs = [elem for elem in elements if elem != structure['separator']]

    # output format
    class Paragraph(BaseModel):
        original: str
        edited: str
        reason: str
        references_list: list[int]|None = None # 參考資料編號

    class RewriteArticleOutput(BaseModel):
        rewrite_list: list[Paragraph]

    user_input = f"根據<編輯建議>{edit_advice}</編輯建議>，對<原文>{paragraphs}</原文>進行改稿。"
    
    rewrite_agent = Agent(
        name="rewrite",
        instructions=prompt,
        model="gpt-4.1",
        output_type=RewriteArticleOutput,
    )
    
    result = await run_agent_with_token_tracking(rewrite_agent, user_input)
    
    # async for event in result.stream_events():
    #     if event.type == "raw_response_event" and hasattr(event, 'data') and hasattr(event.data, 'type'):
    #         if event.data.type == "response.function_call.start":
    #             print(f"🔧 工具調用: {event.data.name}")
    
    end_time = time.time()
    print(f" >>> 改稿耗時 {end_time - start_time:.2f} 秒\n")

    # 轉換為與 rewrite_workflow 相同的輸出格式
    rewrite_list = result.final_output.rewrite_list
    original_list = [paragraph.original for paragraph in rewrite_list]
    edited_list = [paragraph.edited for paragraph in rewrite_list]
    reason_list = [paragraph.reason for paragraph in rewrite_list]
    references_list = [paragraph.references_list for paragraph in rewrite_list if paragraph.references_list]

    final_output = {
        "Result": "Y",
        "ResultData": {
            "edit_type": edit_mode or "normal",
            "original_list": original_list,
            "edited_list": edited_list,
            "reason_list": reason_list,
            "references_list": references_list,
            "url_list": source_url_list or [],
            "elements": elements,
            "structure": structure 
        },
        "Message": "改稿完成"
    }

    return final_output

# if __name__ == "__main__":
#     load_dotenv()
#     api_key = os.getenv("OPENAI_API_KEY")
    
#     text = """美國總統川普今天針對俄羅斯總統蒲亭表達「極其失望」之意，並表示他的執政團隊計劃採取行動降低俄烏戰爭死亡人數，但未說明細節。

# 綜合法新社和路透社報導，自8月在阿拉斯加州會見蒲亭（Vladimir Putin）以來，川普（Donald Trump）持續推動這位俄國領袖與烏克蘭總統澤倫斯基（Volodymyr Zelenskyy）舉行雙邊會談。然而，俄方反而卻加強攻擊烏克蘭的力道。

# 川普在廣播節目被問及是否有遭蒲亭背叛的感覺，告訴保守派評論家詹寧斯（Scott Jennings）說：「我對總統蒲丁極其失望，我可以這麼說。」他說：「我們有很好的交情，我非常失望。」

# 然而，川普並沒有說明俄羅斯是否會面臨任何後果，即便他不久前要求俄烏於兩星期內達成和平協議，而這個期限即將於本週稍晚到期。

# 他表示自己會「做些什麼來幫助民眾生存下去」，但沒有具體說明細節。

# 不久後在橢圓形辦公室時，問及他近期是否曾與蒲亭通過話，川普回答說：「我得知某些非常有意思的事，我想諸位未來幾天就會知曉了。」

# 他補充說，如果蒲亭和澤倫斯基未能會面、終結自2022年開打迄今的俄烏戰爭，將會引發「後果」。

# 另一方面，至於蒲亭昨天在北京會見中國國家主席習近平，並將出席大型閱兵式一事，川普表示對俄中可能形成的聯盟並不擔心。

# 被問及是否擔心中俄結盟對抗美國？川普回答「一點都不擔心」，接著表示「我們有全球最強大的軍隊，而且強很多。他們絕對不會對我們動武。相信我。」（編譯：蔡佳敏）1140903"""
#     media_name = "CNA-medialab"

#     # 譯名檔檢查
#     news_map, converted_text = process_translated_name(text, api_key)

#     # 搜尋編輯守則 -> 編輯建議
#     edit_advice = asyncio.run(edit_advice_process(media_name, converted_text, news_map))

#     # 事件背景摘要
#     background_summary = asyncio.run(background_summary_process(text, media_name))
    
#     print(background_summary.summary)
    
#     source_url = []
#     if background_summary.related_news:
#         for i, news in enumerate(background_summary.related_news, 1):
#             print(f"{i}. {news.title}")
#             print(f"   🔗 {news.url}\n")
#             source_url.append(news.url)
#     else:
#         source_url.append(None)
#         print("無相關新聞")

#     # 判斷新聞類型
#     news_type_result = asyncio.run(classify_edit_mode(converted_text, user_insert_mode="less_mode"))
#     print(news_type_result)

#     rewrite_prompt = generate_rewrite_prompt(news_type_result['edit_role'], news_type_result['edit_mode'], news_type_result['output_format'], background_summary.summary)

#     rewrite_result = asyncio.run(rewrite_process(converted_text, edit_advice, news_type_result['edit_mode'], rewrite_prompt, source_url))
    
#     # 生成給前端的格式化結果
#     if source_url:
#         formatted_result = formatted_output(rewrite_result, source_url, news_type_result['edit_mode'])
#         print(formatted_result)
