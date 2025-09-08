from func import *
from prompts import *
from dotenv import load_dotenv
import asyncio
import json
from agentic import *
import os
from typing import AsyncGenerator
import functions_framework
import flask
from flask import Response, stream_with_context

CORS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST',
}

def create_sse_response(result=None, message=None, result_data="", response_dict=None) -> str:
    """建立 SSE 格式的回應"""
    if response_dict is not None:
        response_data = {
            "Result": response_dict.get("Result", "N"),
            "Message": response_dict.get("Message", ""),
            "ResultData": response_dict.get("ResultData") or ""
        }
    else:
        response_data = {
            "Result": result or "N",
            "Message": message or "",
            "ResultData": result_data or ""
        }
    return f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

def create_error_response(message: str) -> str:
    """建立錯誤回應的快捷函數"""
    return create_sse_response("N", message, {})

async def async_workflow(input_text: str, media_name: str, apponited_edit_mode: str = None) -> AsyncGenerator[str, None]:
    """
    異步執行工作流程，並通過 SSE 即時返回進度
    """
    try:
        start_flow_time = time.time()

        # 1. 先執行譯名檢查
        names_map, converted_text = process_translated_name(input_text, media_name)
        
        yield create_sse_response("Y", "譯名處理完成", names_map)

        # 2. 併行執行三個核心任務
        print(f"==> 開始併行處理查核點、編輯建議、背景摘要")

        async def run_task_with_name(task_name: str, coro):
            """執行任務並附加名稱，統一錯誤處理"""
            try:
                result = await coro
                print(f">>> {task_name} 任務完成")
                return task_name, result, None
            except Exception as e:
                print(f">>> {task_name} 任務失敗: {e}")
                return task_name, None, e

        # 創建其他兩個任務
        tasks = [
            run_task_with_name("check_points", get_check_points(converted_text, media_name)),
            run_task_with_name("background_summary", background_summary_process(converted_text, media_name))
        ]
        
        results = {}

        edit_advice_result = None
        try:
            async for event_type, data in edit_advice_process_streaming(media_name, converted_text, names_map):
                if event_type == "streaming":
                    yield create_sse_response("Y", "edit_advice_streaming", {"delta": data})
                elif event_type == "completed":
                    edit_advice_result = data
                    serializable_result = data[:100] + "..." if len(data) > 100 else data
                    yield create_sse_response("Y", "edit_advice 完成", serializable_result)
            
            results["edit_advice"] = edit_advice_result
        except Exception as e:
            results["edit_advice"] = "編輯建議獲取失敗"
            yield create_sse_response("Y", "edit_advice 失敗，使用預設值", {"task": "edit_advice", "error": str(e)})
        
        # 併行執行其他兩個任務
        for completed_task in asyncio.as_completed(tasks):
            task_name, result, error = await completed_task
            
            if error:
                # 設置預設值
                if task_name == "check_points":
                    results[task_name] = {"Result": "N", "Message": "API超時或錯誤"}
                elif task_name == "edit_advice":
                    results[task_name] = "編輯建議獲取失敗"
                elif task_name == "background_summary":
                    results[task_name] = type('obj', (object,), {'summary': '背景摘要獲取失敗', 'related_news': None})()
                
                # 只對前端關心的任務發送失敗訊息
                if task_name in ["check_points"]:
                    yield create_sse_response("Y", f"{task_name} 失敗，使用預設值", {"task": task_name, "error": str(error)})
            else:
                results[task_name] = result
                
                # 只對前端關心的任務發送完成訊息
                if task_name == "check_points":
                    serializable_result = result if isinstance(result, (dict, str, int, float, bool, list)) else str(result)
                    yield create_sse_response("Y", f"{task_name} 完成", serializable_result)
                elif task_name == "background_summary":
                    # 只在後端記錄，不發送給前端
                    print(f">>> background_summary 完成: {len(result.summary) if hasattr(result, 'summary') else 0} 字摘要, {len(result.related_news) if hasattr(result, 'related_news') and result.related_news else 0} 條相關新聞")
        
        print(f"==> 查核點、編輯建議、背景摘要 完成")

        # 3. 執行改稿流程
        print(f"==> 開始改稿流程...")
        
        # 判斷新聞類型
        news_type_result = await classify_edit_mode(converted_text, apponited_edit_mode)
        
        # 安全地獲取背景摘要
        background_summary = results.get("background_summary")
        summary_text = getattr(background_summary, 'summary', '無背景摘要') if background_summary else '無背景摘要'

        # 準備改稿提示
        rewrite_prompt = generate_rewrite_prompt(
            news_type_result['edit_role'], 
            news_type_result['edit_mode'], 
            news_type_result['output_format'], 
            summary_text
        )
        
        # 提取 source_url
        background_summary = results.get("background_summary")
        source_url = [news.url for news in background_summary.related_news] if background_summary and background_summary.related_news else [None]
        
        # 執行改稿
        rewrite_result = await rewrite_process(
            converted_text, 
            results["edit_advice"], 
            news_type_result['edit_mode'], 
            rewrite_prompt, 
            source_url
        )
        
        print(f"==> 改稿完成")

        # 4. 生成格式化結果 (streaming)
        print(f"==> 格式化改稿結果...")
        if source_url:
            formatted_result = None
            async for event_type, data in formatted_output_streaming(rewrite_result, source_url, news_type_result['edit_mode']):
                if event_type == "streaming":
                    if event_type == "streaming":
                        yield create_sse_response("Y", "marked_output_streaming", {"delta": data})
                elif event_type == "completed":
                    formatted_result = data
                    yield create_sse_response("Y", "source_url_list", {"source_url_list": formatted_result.get("source_url_list", source_url)})
                    print("==> 完成改稿處理")
                    print(formatted_result)

            end_flow_time = time.time()
            print(f"==> 總流程耗時 {end_flow_time - start_flow_time:.2f} 秒")
            
            # 輸出 token 使用摘要
            token_summary = get_session_token_summary()
            print(token_summary)
            
            # 後端記錄完整資訊
        else:
            yield create_sse_response("N", "無法生成格式化結果", {"error": "缺少參考資料"})

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"錯誤詳情: {error_detail}")
        yield create_sse_response("N", f"處理過程發生錯誤: {str(e)}", {"error": str(e), "traceback": error_detail})

# async def main():
#     load_dotenv()
#     media_name = "CNA-medialab"
#     input_text = """美國總統川普今天針對俄羅斯總統蒲亭表達「極其失望」之意，並表示他的執政團隊計劃採取行動降低俄烏戰爭死亡人數，但未說明細節。

# 綜合法新社和路透社報導，自8月在阿拉斯加州會見蒲亭（Vladimir Putin）以來，川普（Donald Trump）持續推動這位俄國領袖與烏克蘭總統澤倫斯基（Volodymyr Zelenskyy）舉行雙邊會談。然而，俄方反而卻加強攻擊烏克蘭的力道。

# 川普在廣播節目被問及是否有遭蒲亭背叛的感覺，告訴保守派評論家詹寧斯（Scott Jennings）說：「我對總統蒲丁極其失望，我可以這麼說。」他說：「我們有很好的交情，我非常失望。」

# 然而，川普並沒有說明俄羅斯是否會面臨任何後果，即便他不久前要求俄烏於兩星期內達成和平協議，而這個期限即將於本週稍晚到期。

# 他表示自己會「做些什麼來幫助民眾生存下去」，但沒有具體說明細節。

# 不久後在橢圓形辦公室時，問及他近期是否曾與蒲亭通過話，川普回答說：「我得知某些非常有意思的事，我想諸位未來幾天就會知曉了。」

# 他補充說，如果蒲亭和澤倫斯基未能會面、終結自2022年開打迄今的俄烏戰爭，將會引發「後果」。

# 另一方面，至於蒲亭昨天在北京會見中國國家主席習近平，並將出席大型閱兵式一事，川普表示對俄中可能形成的聯盟並不擔心。

# 被問及是否擔心中俄結盟對抗美國？川普回答「一點都不擔心」，接著表示「我們有全球最強大的軍隊，而且強很多。他們絕對不會對我們動武。相信我。」"""
    
#     # 消費異步生成器的每個 yield 值，不再需要傳入 api_key
#     async for sse_data in async_workflow(input_text, media_name, None):
#         print(sse_data, end='', flush=True)

@functions_framework.http
def agentic_flow(request: flask.Request):
    """Cloud Function 主要入口點"""
    
    # 處理 CORS 預檢請求
    if request.method == 'OPTIONS':
        return ('', 204, CORS)
    
    headers = CORS.copy()
    headers['Content-Type'] = 'text/event-stream'
    headers['Cache-Control'] = 'no-cache'
    headers['Connection'] = 'keep-alive'
    
    try:
        # 解析請求內容
        payload = request.get_json(force=True)
        print(f">>> [Debug] payload: {payload}")

        input_text = payload.get("input_text", "")
        media_name = payload.get("media_name", None)  # 新增 media_name 參數，預設為 None
        apponited_edit_mode = payload.get("edit_mode", None)  # 0708 新增 edit_mode 控制改稿幅度，預設是None

        if not input_text or len(input_text) < 10:
            error_response = create_sse_response("N", "沒有提供訊息、訊息長度不夠", "")
            return Response(error_response, headers=headers)
        
        # 設定 OpenAI API KEY
        if not set_openai_api_key(media_name):
            error_response = create_sse_response("N", "OpenAI API KEY 設定失敗", "")
            return Response(error_response, headers=headers)
        
        # 簡化版本：直接在新線程中運行異步代碼
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def run_workflow():
            try:
                # 在新線程中運行異步工作流程
                async def run():
                    async for sse_data in async_workflow(input_text, media_name, apponited_edit_mode):
                        result_queue.put(('data', sse_data))
                    result_queue.put(('done', None))
                
                asyncio.run(run())
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        # 啟動工作線程
        thread = threading.Thread(target=run_workflow)
        thread.start()
        
        def stream_response():
            while True:
                try:
                    msg_type, content = result_queue.get(timeout=300)
                    if msg_type == 'data':
                        yield content
                    elif msg_type == 'done':
                        break
                    elif msg_type == 'error':
                        yield create_sse_response("N", f"處理錯誤: {content}", "")
                        break
                except queue.Empty:
                    yield create_sse_response("N", "處理超時", "")
                    break
            
            thread.join(timeout=30)
        
        return Response(
            stream_with_context(stream_response()),
            headers=headers
        )
    
    except Exception as e:
        error_response = create_sse_response("N", f"請求處理錯誤：{str(e)}", "")
        return Response(error_response, headers=headers) 

