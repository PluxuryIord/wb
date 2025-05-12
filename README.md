# 🚨 Fraud Detection UI App

## 📦 Запуск локально

1. Установите зависимости:
    ```bash
    pip install -r app/requirements.txt
    ```

2. Запустите:
    ```bash
    streamlit run app/ui.py
    ```

---

## 🐳 Запуск в Docker

1. Соберите образ:
    ```bash
    docker build -t fraud_ui .
    ```

2. Запустите:
    ```bash
    docker run -p 8501:8501 fraud_ui
    ```

3. Перейдите в браузере:
    ```
    http://localhost:8501
    ```

---

## 📤 Входной файл

CSV-файл должен содержать данные в табличном виде. Например:

| user_id | nm_id | CreatedDate | service | total_ordered | ... |
|---------|--------|--------------|---------|----------------|-----|
| ...     | ...    | ...          | ...     | ...            | ... |

