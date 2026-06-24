import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

os.makedirs('data', exist_ok=True)

st.title("Data")
st.info("Поддерживаются форматы: CSV")
f = st.file_uploader("Upload file")

if f:
  file_path = os.path.join("data", f.name)

  with open(file_path, 'wb') as w:
    w.write(f.getvalue())

  st.success(f"Файл **{f.name}** успешно загружен!")

  file_size = os.path.getsize(file_path)
  upload_time = datetime.now().strftime("%H:%M:%S")
  col1, col2, col3 = st.columns(3)
  with col1:
      st.markdown(f'📄 Размер файла: {file_size / 1024:.2f} KB')
  with col2:
      st.markdown(f'🕐 Время загрузки: {upload_time}')
      
  with col3:
      st.markdown(f'📊 Тип файла: {f.type}')

files = os.listdir('data')
name = st.selectbox("files in data", files) if files else None


if name:
  st.write(name)
  if name.lower().endswith('.csv'):
    st.dataframe(pd.read_csv(os.path.join("data", name)).head())