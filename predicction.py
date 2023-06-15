import joblib
import streamlit as st
def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)



