# В данном блоке сильно помогал ИИ)

import requests
import streamlit as st
import pandas as pd
import json

# Конфигурация страницы
st.set_page_config(
    page_title="ML Training Dashboard",
    layout="wide"
)

st.title("Train Regression Models")
st.markdown("---")

with st.sidebar:
    st.header("Информация")
    st.info("""
    **Доступные модели:**
    - Linear Regression
    - Ridge Regression  
    - Decision Tree
    
    **API:** http://127.0.0.1:8000
    """)
    try:
        health = requests.get("http://127.0.0.1:8000/health")
        if health.status_code == 200:
            st.success("API подключен")
        else:
            st.error("API недоступен")
    except:
        st.error("API не запущен")
    
    st.divider()

    try:
        import os
        if os.path.exists("data"):
            files = os.listdir("data")
            if files:
                st.subheader("Доступные файлы:")
                for f in files:
                    st.text(f"{f}")
    except:
        pass

with st.form("train_form"):
    st.subheader("Параметры обучения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filename = st.text_input(
            "Имя файла с данными",
            placeholder="например: currency_data.csv",
            help="Файл должен находиться в папке data/"
        )
        
        model_type = st.selectbox(
            "Тип модели",
            options=['linear', 'ridge', 'decision_tree'],
            format_func=lambda x: {
                'linear': 'Linear Regression',
                'ridge': 'Ridge Regression',
                'decision_tree': 'Decision Tree'
            }[x],
            help="Выберите алгоритм для обучения"
        )
    
    with col2:
        train_size = st.slider(
            "Размер обучающей выборки",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Доля данных для обучения (остальное - тест)"
        )
        
        model_name = st.text_input(
            "Название модели",
            placeholder="например: eur_usd_model",
            help="Имя для сохранения модели"
        )
    
    # Дополнительные параметры для Decision Tree
    if model_type == 'decision_tree':
        st.subheader("Параметры Decision Tree")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_depth = st.number_input(
                "Max Depth",
                min_value=1,
                max_value=50,
                value=10,
                help="Максимальная глубина дерева"
            )
        
        with col2:
            min_samples_split = st.number_input(
                "Min Samples Split",
                min_value=2,
                max_value=20,
                value=5,
                help="Минимальное количество образцов для разделения"
            )
        
        with col3:
            min_samples_leaf = st.number_input(
                "Min Samples Leaf",
                min_value=1,
                max_value=20,
                value=2,
                help="Минимальное количество образцов в листе"
            )
    
    # Кнопка отправки
    submitted = st.form_submit_button("Обучить модель", type="primary", use_container_width=True)

# Обработка отправки формы
if submitted:
    if not filename:
        st.error("Введите имя файла")
    elif not model_name:
        st.error("Введите название модели")
    else:
        with st.spinner(f"Обучение модели {model_type}..."):
            try:
                # Подготовка данных для запроса
                data = {
                    "filename": filename,
                    "model_name": model_name,
                    "train_size": train_size,
                    "model_type": model_type
                }
                
                # Добавляем параметры дерева если нужно
                if model_type == 'decision_tree':
                    data["max_depth"] = max_depth
                    data["min_samples_split"] = min_samples_split
                    data["min_samples_leaf"] = min_samples_leaf
                
                # Отправка запроса к API
                response = requests.post(
                    "http://127.0.0.1:8000/train",
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Показываем успешное завершение
                    st.success("Модель успешно отправлена на обучение!")
                    st.balloons()
                    
                    # Отображаем информацию о задании
                    st.info(f"""
                    **Детали обучения:**
                    - Файл: `{filename}`
                    - Модель: `{model_name}`
                    - Тип: `{model_type}`
                    - Размер выборки: `{train_size}`
                    """)
                    
                    # Проверяем наличие обученных моделей
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Проверить список моделей"):
                            try:
                                models_response = requests.get("http://127.0.0.1:8000/models")
                                if models_response.status_code == 200:
                                    models = models_response.json()
                                    st.write("Доступные модели:", models)
                            except:
                                st.error("Не удалось получить список моделей")
                    
                    with col2:
                        if st.button("🔍 Открыть документацию API"):
                            st.markdown("[Открыть Swagger UI](http://127.0.0.1:8000/docs)")
                
                elif response.status_code == 404:
                    st.error(f"Файл `{filename}` не найден в папке data/")
                else:
                    st.error(f"Ошибка: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Не удалось подключиться к API. Убедитесь, что сервер запущен на порту 8000")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

# Дополнительная секция для просмотра данных
st.divider()
st.subheader("Предпросмотр данных")

col1, col2 = st.columns([2, 1])

with col1:
    preview_file = st.text_input(
        "Введите имя файла для предпросмотра",
        placeholder="например: currency_data.csv",
        key="preview"
    )

with col2:
    preview_rows = st.number_input("Количество строк", 5, 50, 10)

if preview_file:
    try:
        df = pd.read_csv(f"data/{preview_file}")
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Статистика
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Строк", len(df))
        with col2:
            st.metric("Колонок", len(df.columns))
        with col3:
            st.metric("Размер", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
    except FileNotFoundError:
        st.warning(f"Файл `{preview_file}` не найден")
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")

# Футер
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>ML Training Dashboard v1.0 | API: http://127.0.0.1:8000 | Документация: /docs</p>
    </div>
    """,
    unsafe_allow_html=True
)

