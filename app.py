import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# App title
st.title('Customer Churn Prediction App')
st.write('This app predicts whether a customer will churn based on simple input features.')

# Input features
age = st.number_input('Age', min_value=18, max_value=100, value=30)
salary = st.number_input('Monthly Salary (in USD)', min_value=1000, max_value=10000, value=3000)
tenure = st.slider('Number of years with the company', 0, 10, 3)

# Sample training data (Normally you'd load a dataset, but for demo purpose we're training it here)
data = {
    'age': [25, 45, 30, 35, 50, 40, 23, 60],
    'salary': [3000, 8000, 4000, 5000, 9000, 6000, 2000, 10000],
    'tenure': [2, 8, 4, 6, 10, 7, 1, 9],
    'churn': [0, 1, 0, 1, 1, 0, 0, 1]  # 1: Churn, 0: Not Churn
}

df = pd.DataFrame(data)

# Model
X = df[['age', 'salary', 'tenure']]
y = df['churn']

model = LogisticRegression()
model.fit(X, y)

# Predict on user input
user_data = [[age, salary, tenure]]
prediction = model.predict(user_data)

if st.button('Predict'):
    if prediction[0] == 1:
        st.error('⚠️ The customer is likely to churn.')
    else:
        st.success('✅ The customer is likely to stay.')

