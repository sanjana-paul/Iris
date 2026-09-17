
import streamlit as st
import joblib
import numpy as np

# deployment
model = joblib.load("iris_models.pkl")
st.title("Iris Flower Prediction")

# input labels
sepal_length = st.number_input("sepal_length")
sepal_width = st.number_input("sepal_width")
petal_length = st.number_input("petal_length")
petal_width = st.number_input("petal_width")

# prediction
if st.button("predict"):
  input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]]).astype(np.float64)
  prediction = model.predict(input_data)
  st.success(prediction[0])
