import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from io import StringIO

# Function to train the model
def train_model(df):
    X = df[['area', 'bedrooms', 'age']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Function to make predictions
def make_prediction(model, area, bedrooms, age):
    prediction = model.predict([[area, bedrooms, age]])
    return prediction[0]

# Main function
def main():
    st.title("House Price Predictor")
    
    # Load the data
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Train the model
        model = train_model(df)
        
        # Create input fields for the user
        area = st.number_input("Area (sqft)", min_value=0)
        bedrooms = st.number_input("Number of Bedrooms", min_value=0)
        age = st.number_input("Age of the House (years)", min_value=0)
        
        # Make predictions when the user clicks the button
        if st.button("Predict Price"):
            prediction = make_prediction(model, area, bedrooms, age)
            st.write(f"The predicted price of the house is: ${prediction:.2f}")

        with st.sidebar:
            st.caption("<p style ='text-align:center'>This is created by Rohit Nihalani</p>",unsafe_allow_html=True)  

if __name__ == "__main__":
    main()
