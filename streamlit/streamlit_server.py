import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
It's running !!!!
""")


st.sidebar.header("User input data features")
st.sidebar.markdown("""
[CSV input here]
""")

uploaded_file = st.sidebar.file_uploader("CSV file: ", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        SepalLengthCm = st.sidebar.slider('SepalLengthCm', 4.3, 8.0, 6.0)
        SepalWidthCm = st.sidebar.slider('SepalLengthCm', 2.0, 5.0, 3.0)
        PetalLengthCm = st.sidebar.slider('SepalLengthCm', 1.0, 7.0, 4.0)
        PetalWidthCm = st.sidebar.slider('SepalLengthCm', 0.1, 3.0, 2.0)
        data = {
            'SepalLengthCm': SepalLengthCm,
            'SepalWidthCm': SepalWidthCm,
            'PetalLengthCm': PetalLengthCm,
            'PetalWidthCm': PetalWidthCm
        }
        features = pd.DataFrame(data, index=[0])
        return features 
    input_df = user_input_features()


iris_raw = pd.read_csv('data/iris.csv')
iris = iris_raw.drop(['Id', 'Species'], axis=1)

df = pd.concat([input_df, iris], axis=0)

df = df[:1]

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Using sample data, upload a CSV file if needed")
    st.write(df)

# Load the model
load_clf = pickle.load(open('iris_clf.pkl', 'rb'))

# Apply model to make predictions
predictions = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Show the prediction results
st.subheader('Prediction')
iris_species = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
st.write(iris_species[predictions])

st.subheader('Prediction Probability')
st.write(prediction_proba)  