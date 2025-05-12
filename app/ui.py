import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Жестко заданный список фичей, которые ожидает модель (28 штук)
MODEL_FEATURES = [
    'IsPaid', 'count_items', 'avg_unique_purchase', 'is_courier',
    'NmAge', 'Distance', 'DaysAfterRegistration', 'number_of_ordered_items',
    'year', 'month', 'weekday', 'weekend', 'service_nnsz',
    'PaymentType_ACC', 'PaymentType_BAL', 'PaymentType_CRE',
    'PaymentType_MPM', 'PaymentType_PDL', 'PaymentType_QRS',
    'far_and_large_order', 'order_vs_mean_order',
    'is_new_user', 'is_new_product', 'far_distance',
    'dist_<50', 'dist_50-200', 'dist_200-1000', 'dist_>1000'
]


def preprocess_input(df):
    df = df.copy()

    # 1. Обработка даты
    if 'CreatedDate' in df.columns:
        dt = pd.to_datetime(df['CreatedDate'])
        df['year'] = dt.dt.year
        df['month'] = dt.dt.month
        df['weekday'] = dt.dt.weekday
        df['weekend'] = (dt.dt.weekday >= 5).astype(int)

    # 2. Категориальные признаки
    if 'service' in df.columns:
        df['service_nnsz'] = (df['service'] == 'nnsz').astype(int)

    if 'PaymentType' in df.columns:
        for pt in ['ACC', 'BAL', 'CRE', 'MPM', 'PDL', 'QRS']:
            df[f'PaymentType_{pt}'] = (df['PaymentType'] == pt).astype(int)

    # 3. Создаем только нужные фичи
    processed_df = pd.DataFrame()

    for feature in MODEL_FEATURES:
        if feature in df.columns:
            processed_df[feature] = df[feature]
        else:
            processed_df[feature] = 0  # Заполняем нулями отсутствующие

    # 4. Приведение типов
    bool_cols = ['IsPaid', 'is_courier', 'weekend', 'service_nnsz',
                 'is_new_user', 'is_new_product', 'far_distance']
    for col in bool_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(int)

    # 5. Удаление возможных NaN
    processed_df.fillna(0, inplace=True)

    return processed_df[MODEL_FEATURES]


@st.cache_resource
def load_model():
    artifact = joblib.load("app/model/best_model.pkl")
    model = artifact["model"]
    return model


model = load_model()

st.title("🚨 Fraud Detection UI")
st.write("Загрузите CSV-файл с заказами, и мы предскажем фрод.")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df_processed = preprocess_input(df)

        # st.write("Используемые колонки:", df_processed.columns.tolist())
        # st.write("Количество фичей:", len(df_processed.columns))

        # Предсказание
        proba = model.predict_proba(df_processed)[:, 1]
        print(proba)
        thr = 0.49466894668438827
        if proba >= thr:
            df['confidence'] = 0.5 + 0.5 * ((proba - thr) / (1 - thr))
        else:
            df['confidence'] = 0.5 + 0.5 * ((thr - proba) / thr)
        df['prediction'] = (proba >= 0.49466894668438827).astype(int)

        st.success("Предсказания выполнены!")
        st.dataframe(df[['prediction', 'confidence'] +
                        [c for c in df.columns if c not in ['prediction', 'confidence']]])

        csv = df.to_csv(index=False).encode()
        st.download_button("💾 Скачать результат CSV", csv, "fraud_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        if 'df_processed' in locals():
            st.write("Первые строки данных:", df_processed.head())
            st.write("Типы данных:", df_processed.dtypes)
