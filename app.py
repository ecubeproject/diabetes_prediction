import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
# Load the serialized model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit webpage configuration
st.title('Diabetes Prediction App')
st.write('This application predicts whether a patient is likely to have diabetes based on diagnostic measures.')


# Collecting user input features into dataframe
def user_input_features():
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
    Glucose = st.number_input('Plasma glucose concentration after 2 hours in an oral glucose tolerance test',
                              min_value=0, max_value=200, step=1)
    BloodPressure = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0, max_value=122, step=1)
    SkinThickness = st.number_input('Triceps skin fold thickness (mm)', min_value=0, max_value=99, step=1)
    Insulin = st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0, max_value=846, step=1)
    BMI = st.slider('BMI: Body mass index (weight in kg/(height in m)^2)', min_value=0.0, max_value=70.0)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5)
    Age = st.number_input('Age (Years)', min_value= 21, max_value=100, step=1)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    features = pd.DataFrame(data, index=[0])
    return features


# Get user inputs
input_df = user_input_features()

if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    # Debugging output
    print("Prediction value:", prediction[0])
    print("Probability:", probability)

    # Adjust threshold if necessary (for example: threshold = 0.5)
    threshold = 0.5
    prediction_label = "Diabetic" if probability >= threshold else "Not Diabetic"

    st.write(f'Prediction: {prediction_label}')
    st.write(f'Probability of being diabetic: {probability:.2f}')
