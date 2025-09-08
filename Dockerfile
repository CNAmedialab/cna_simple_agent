# 使用 Python 3.11 官方映像
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 設置環境變量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY . .

# 移除不必要的文件（可選）
RUN rm -rf .git .gitignore .vscode __pycache__ venv

# 設置正確的權限
RUN chmod +x /app

# 暴露端口
EXPOSE $PORT

# 使用 functions-framework 運行應用
CMD exec functions-framework --target main.agentic_flow --port $PORT