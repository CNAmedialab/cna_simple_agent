# SSE API 文檔 - 新聞改稿流程

## 概述
本 API 使用 Server-Sent Events (SSE) 串流方式回傳新聞改稿處理的即時進度和結果。
支援基於 Google Secret Manager 的多媒體單位權限驗證。

## API 端點
```
POST https://cna-simple-agent-1007110536706.asia-east1.run.app/
Content-Type: application/json
Accept: text/event-stream
```

## 請求參數
```json
{
  "input_text": "要處理的新聞內容",
  "media_name": "媒體單位名稱（用於 Secret Manager 驗證）",
  "edit_mode": "改稿模式（可選）"
}
```

## 基本格式
所有 SSE 訊息遵循以下格式：
```
data: {"Result": "Y|N", "Message": "訊息類型", "ResultData": {...}}
```

- `Result`: "Y" 表示成功，"N" 表示錯誤
- `Message`: 描述當前處理階段或錯誤類型
- `ResultData`: 包含具體數據內容

---

## 權限驗證錯誤

### API Key 驗證失敗
```json
data: {"Result": "N", "Message": "OpenAI API KEY 設定失敗", "ResultData": ""}
```

### 輸入驗證錯誤
```json
data: {"Result": "N", "Message": "沒有提供訊息、訊息長度不夠", "ResultData": ""}
```

### 權限不足
```json
data: {"Result": "N", "Message": "無權限使用 {media_name} 或 API key 不存在", "ResultData": ""}
```

---

## 流程階段與訊息類型

### 1. 譯名處理
```json
data: {"Result": "Y", "Message": "譯名處理完成", "ResultData": {"name1": "translation1", "name2": "translation2"}}
```
- 包含人名、地名等專有名詞的翻譯對照表

### 2. 並行任務執行

#### 2.1 編輯建議 (Streaming)
```json
data: {"Result": "Y", "Message": "edit_advice_streaming", "ResultData": {"delta": "建"}}
data: {"Result": "Y", "Message": "edit_advice_streaming", "ResultData": {"delta": "議"}}
data: {"Result": "Y", "Message": "edit_advice_streaming", "ResultData": {"delta": "內"}}
```
- 編輯建議內容會逐字符串流輸出
- `delta` 為單個字符

#### 2.2 查核點完成
```json
data: {"Result": "Y", "Message": "check_points 完成", "ResultData": {"Result": "Y", "ResultData": {"check_points": ["川普是否真的與蒲亭在阿拉斯加州會面", "川普與蒲亭的『交情』真實性", "川普聲稱的『降低俄烏戰爭死亡人數』計畫是否屬實", "川普對蒲亭『極其失望』的背景與脈絡", "川普推動俄烏領袖會談的實際進展"]}, "Message": "API成功回傳結果"}}
```

- `check_points` 是字串陣列，每個元素為一個查核要點

**失敗情況**：
```json
data: {"Result": "Y", "Message": "check_points 失敗，使用預設值", "ResultData": {"Result": "N", "ResultData": {"check_points": None}, "Message": "API超時或錯誤"}}
```

### 3. 最終改稿結果 (Streaming)

#### 3.1 標記文本串流
```json
data: {"Result": "Y", "Message": "marked_output_streaming", "ResultData": {"delta": "美"}}
data: {"Result": "Y", "Message": "marked_output_streaming", "ResultData": {"delta": "國"}}
data: {"Result": "Y", "Message": "marked_output_streaming", "ResultData": {"delta": "/@"}}
data: {"Result": "Y", "Message": "marked_output_streaming", "ResultData": {"delta": "總"}}
data: {"Result": "Y", "Message": "marked_output_streaming", "ResultData": {"delta": "統"}}
```
- 改稿後的標記文本會逐字符串流輸出
- 標記格式：`/@修改後的內容@/` 表示已修改的部分
- 參考資料標記：`<1>`, `<2>` 等數字標註

#### 3.2 參考資料列表
```json
data: {"Result": "Y", "Message": "source_url_list", "ResultData": {"source_url_list": ["url1", "url2"]}}
```

---

## 錯誤處理

### 任務失敗
```json
data: {"Result": "Y", "Message": "edit_advice 失敗，使用預設值", "ResultData": {"task": "edit_advice", "error": "錯誤原因"}}
```

### 處理錯誤
```json
data: {"Result": "N", "Message": "處理錯誤: 錯誤描述", "ResultData": ""}
```

### 處理超時
```json
data: {"Result": "N", "Message": "處理超時", "ResultData": ""}
```

### 系統錯誤
```json
data: {"Result": "N", "Message": "處理過程發生錯誤: 錯誤描述", "ResultData": {"error": "錯誤訊息", "traceback": "詳細錯誤堆疊"}}
```

---

## 前端實作建議

### Google Chrome Extension 範例

#### manifest.json 權限設定
```json
{
  "manifest_version": 3,
  "permissions": ["activeTab"],
  "host_permissions": [
    "https://your-api-domain.com/*"
  ]
}
```

#### Content Script / Service Worker 實作
```javascript
// Google Extension API 調用函數
async function processNewsWithAPI(newsContent, mediaName = "CNA-medialab") {
    const API_ENDPOINT = "https://cna-simple-agent-1007110536706.asia-east1.run.app/";
    
    const requestData = {
        input_text: newsContent,
        media_name: mediaName,      // 媒體單位名稱（權限驗證）
        edit_mode: "less_mode"      // 可選改稿模式
    };

    try {
        // 顯示處理中狀態
        updateStatus("正在處理新聞...", "loading");
        
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // 處理 SSE 串流回應
        await processStreamResponse(response);
        
    } catch (error) {
        console.error('API 調用失敗:', error);
        updateStatus(`錯誤: ${error.message}`, "error");
    }
}

// 處理 SSE 串流回應
async function processStreamResponse(response) {
    const reader = response.body.getReader();
    let editAdviceBuffer = '';
    let markedTextBuffer = '';
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = new TextDecoder().decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const response = JSON.parse(line.substring(6));
                        await handleSSEMessage(response, editAdviceBuffer, markedTextBuffer);
                    } catch (e) {
                        console.warn('解析 SSE 數據失敗:', line, e);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

// 處理不同類型的 SSE 訊息
async function handleSSEMessage(response, editAdviceBuffer, markedTextBuffer) {
    // 檢查錯誤狀態
    if (response.Result === 'N') {
        updateStatus(`錯誤: ${response.Message}`, "error");
        return;
    }
    
    switch (response.Message) {
        case 'edit_advice_streaming':
            editAdviceBuffer += response.ResultData.delta;
            updateEditAdviceDisplay(editAdviceBuffer);
            break;
            
        case 'marked_output_streaming':
            markedTextBuffer += response.ResultData.delta;
            updateMarkedTextDisplay(markedTextBuffer);
            break;
            
        case '譯名處理完成':
            updateStatus("譯名處理完成", "success");
            displayNameTranslations(response.ResultData);
            break;
            
        case 'check_points 完成':
            updateStatus("查核點分析完成", "success");
            displayCheckPoints(response.ResultData);
            break;
            
        case 'source_url_list':
            displaySourceUrls(response.ResultData.source_url_list);
            updateStatus("處理完成！", "success");
            break;
            
        default:
            console.log('未知訊息類型:', response.Message);
    }
}

// UI 更新函數 - 適用於 Chrome Extension 環境
function updateStatus(message, type = "info") {
    const statusElement = document.getElementById('processing-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
    }
    
    // 也可以發送到 popup 或其他頁面
    if (typeof chrome !== 'undefined' && chrome.runtime) {
        chrome.runtime.sendMessage({
            type: 'statusUpdate',
            message: message,
            status: type
        });
    }
}

function updateEditAdviceDisplay(content) {
    const element = document.getElementById('edit-advice');
    if (element) {
        element.textContent = content;
    }
}

function updateMarkedTextDisplay(content) {
    const element = document.getElementById('marked-text');
    if (element) {
        // 處理標記格式 /@...@/ 顯示為高亮
        const formatted = content.replace(
            /\/@([^@]*?)@\//g, 
            '<mark class="edited">$1</mark>'
        );
        element.innerHTML = formatted;
    }
}

function displayNameTranslations(translations) {
    const element = document.getElementById('name-translations');
    if (element && Object.keys(translations).length > 0) {
        const list = Object.entries(translations)
            .map(([key, value]) => `<li>${key} → ${value}</li>`)
            .join('');
        element.innerHTML = `<ul>${list}</ul>`;
    }
}

function displaySourceUrls(urls) {
    const element = document.getElementById('source-urls');
    if (element && urls && urls.length > 0) {
        const list = urls
            .filter(url => url) // 過濾空值
            .map((url, index) => `<li><a href="${url}" target="_blank">參考資料 ${index + 1}</a></li>`)
            .join('');
        element.innerHTML = `<ul>${list}</ul>`;
    }
}

// Extension 特定功能：從頁面提取新聞內容
function extractNewsFromPage() {
    // 根據不同新聞網站的結構調整選擇器
    const selectors = [
        'article',
        '.article-content', 
        '.news-content',
        '[data-testid="article-body"]',
        '.story-body'
    ];
    
    for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (element) {
            return element.innerText.trim();
        }
    }
    
    return null;
}

// 使用範例
document.addEventListener('DOMContentLoaded', () => {
    const processButton = document.getElementById('process-news');
    if (processButton) {
        processButton.addEventListener('click', async () => {
            const newsContent = extractNewsFromPage() || 
                              document.getElementById('news-input')?.value;
            
            if (newsContent && newsContent.length >= 10) {
                await processNewsWithAPI(newsContent, "CNA-medialab");
            } else {
                updateStatus("請提供至少 10 個字的新聞內容", "error");
            }
        });
    }
});
```

---

## 處理流程時序

1. **權限驗證** → 驗證 `media_name` 是否有權限使用 OpenAI API
2. **譯名處理** → 立即完成
3. **並行任務啟動** → 三個任務同時開始：
   - `edit_advice` (串流輸出)
   - `check_points` (完成時回報)
   - `background_summary` (後端處理，供改稿參考)
4. **改稿處理** → 等待上述任務完成後開始
5. **最終結果** → `marked_output_streaming` 串流輸出
6. **參考資料** → `source_url_list` 回傳

## 技術架構說明

### 簡化的線程處理
- 使用單一工作線程運行異步工作流程
- 通過隊列機制實現串流響應
- 支援 5 分鐘超時機制

### API Key 管理
- 統一的 OpenAI 客戶端工廠
- 基於 Google Secret Manager 的多媒體權限驗證
- 客戶端快取機制，避免重複創建

## 重要注意事項

1. **權限驗證**：必須提供有效的 `media_name` 或確保有預設 API key
2. **串流內容重組**：`edit_advice_streaming` 和 `marked_output_streaming` 需要在前端累積重組
3. **錯誤處理**：任務失敗時會使用預設值繼續流程，不會中斷整體處理
4. **標記格式**：最終結果中的 `/@...@/` 標記表示修改內容，需要前端做特殊顯示處理
5. **參考資料**：`<1>`, `<2>` 等標記對應 `source_url_list` 中的相關新聞
6. **超時處理**：整個流程最多執行 5 分鐘，超時會返回錯誤