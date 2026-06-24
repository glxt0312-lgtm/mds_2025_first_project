import sqlite3
import pandas as pd 
import streamlit as st

st.title("Результаты обучения модели")

def load_data():
  conn = sqlite3.connect("models.db")
  df = pd.read_sql_query("select * from train_results order by id desc", conn)
  conn.close()
  return df

st.button("Обновить")

try:
  df = load_data()
  st.dataframe(df)
except Exception as e:
  st.error(str(e))
