import streamlit as st
import pandas as pd
import numpy as np

from predicction import predict
st.title("Classifying Iris Flowers")
st.markdown("Toy model to play to classify iris flowers into setosa, versicolor, virginica")
st.header("Plant Features")
col1, col2 = st.columns(2)
with col1:
    
    st.text("Sepal characteristics")
    sepal_l = st.slider("Sepal lenght (cm)", 1, 8)
    sepal_w = st.slider("Sepal width (cm)", 2, 5)
with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider("Petal lenght (cm)", 1, 7)
    petal_w = st.slider("Petal width (cm)", 1, 3)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])
