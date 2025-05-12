FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем саму аппу и модель
COPY app/ ./app/
COPY app/model/best_model.pkl ./app/model/best_model.pkl

# Указываем порт
EXPOSE 8501

# Запуск streamlit
CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
