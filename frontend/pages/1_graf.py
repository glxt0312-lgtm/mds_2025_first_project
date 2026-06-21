#$ pip install streamlit
#$ streamlit hello

import streamlit as st 

st.title("Привет, streamlit")

name = st.text_input("Enter your name:", placeholder="Student")

number = st.slider("Choise the number:", 0, 10, 5)

if st.button("Считаем куб"):
  result = number ** 3
  st.success(f"Привет, {name}, КУБ числа {number} = {result}")
  st.balloons()
