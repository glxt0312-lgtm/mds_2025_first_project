#$ pip install streamlit
#$ streamlit hello

import streamlit as st 

st.set_page_config("Первый проект",layout='wide')
st.title(f"Первый проект")

# number = st.slider("Choise the number:", 0, 10, 5)

# if st.button("Считаем квадрат"):
#   result = number ** 2
#   st.success(f"Привет, {name}, Квадрат числа {number} = {result}")
#   st.balloons()
