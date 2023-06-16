import streamlit as st
import pandas as pd
import numpy as np

from predicction import predict
st.title("Classifying Iris Flowers")
st.markdown("Toy model to play to classify iris flowers into setosa, versicolor, virginica")
st.header("Plant Features")
col1, col2 = st.columns(2)
with col1:
    a = float(1.0)
    b = float(8.0)
    c = float(0.5)
    d = float(2.0)
    e = float(4.4)
    f = float(7.0)
    g = float(0.1)
    h = float(2.5)
    st.text("Sepal characteristics")
    sepal_l = st.slider("Sepal lenght (cm)", a, b, c)
    sepal_w = st.slider("Sepal width (cm)", d, e, c)
with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider("Petal lenght (cm)", a, f, c)
    petal_w = st.slider("Petal width (cm)", g, h, c)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])
