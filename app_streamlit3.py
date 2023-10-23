import streamlit as st
import numpy as np
from scipy import stats
from joblib import load

import zipfile

with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
model = load('best_model.joblib')


# Load the pre-trained model and the scaler
model = load('best_model.joblib')
scaler = load('scaler_model (2).joblib')

# Define the slider range and step with whole numbers
min_value = 0
step_value = 1  # Change this step as needed

def main():
    st.sidebar.header('Group 20 Player Rating Predictor App')
    # Take user input
    st.sidebar.title("Input Player Features")
    st.title("Player Rating Predictor:")
    st.title("Below are your selected values")
    st.title("Scroll down to see the PREDICTION")
    
    movement_reactions = st.sidebar.slider('Movement Reactions', 0, 100)
    st.write(f"Your selected Value for Movement reactions is: {movement_reactions}")
    
    mentality_composure = st.sidebar.slider('Mentality Composure', 0, 100)
    st.write(f"Your selected Value for Mentality Composure is: {mentality_composure}")
    
    passing = st.sidebar.slider('Passing', 0, 100)
    st.write(f"Your selected Value for Passing is: {passing}")
    
    potential = st.sidebar.slider('Potential', 0, 100)
    st.write(f"Your selected Value for Potential is: {potential}")
    
    release_clause_eur =  st.sidebar.number_input('Release Clause (EUR)', 500, 500000, 10000)
    st.write(f"Your selected Value for Release Clause (EUR) is: {release_clause_eur}")
    
    dribbling = st.sidebar.slider('Dribbling', 10, 100, 50)
    st.write(f"Your selected Value for Dribbling is: {dribbling}")
    
    wage_eur = st.sidebar.number_input('Wage (EUR)', 500, 500000, 10000)
    st.write(f"Your selected Value for Wage (EUR) is: {wage_eur}")
    
    power_shot_power = st.sidebar.slider('Power Shot Power', 10, 100, 50)
    st.write(f"Your selected Value for Power Shot Power is: {power_shot_power}")
    
    value_eur = st.sidebar.number_input('Value (EUR)', 1000, 500000000, 1000000)
    st.write(f"Your selected Value for Value (EUR) is: {value_eur}")
    
    mentality_vision = st.sidebar.slider('Mentality Vision', 10, 100, 50)
    st.write(f"Your selected Value for Mentality Vision is: {mentality_vision}")
    
    attacking_short_passing = st.sidebar.slider('Attacking Short Passing', 10, 100, 50)
    st.write(f"Your selected Value for Attacking Short Passing is: {attacking_short_passing}")
    
    age = st.sidebar.slider('Age', 16, 40, 25)
    st.write(f"Your selected Value for Age is: {age}")
    
    shooting = st.sidebar.slider('Shooting', 10, 100, 50)
    st.write(f"Your selected Value for Shooting is: {shooting}")
    
    skill_ball_control = st.sidebar.slider('Skill Ball Control', 10, 100, 50)
    st.write(f"Your selected Value for Skill Ball Control is: {skill_ball_control}")
    
    work_rate = st.sidebar.slider('work_rate', 0, 100)
    st.write(f"Your selected Value for work rate is: {work_rate}")
    
    
    # Create a button for the user to click and get predictions
    st.title("Here we are")
    st.title("CLICK ON 'PREDICT' TO SEE THE PREDICTION")

    if st.button("Predict"):
        input_features = np.array([
            movement_reactions, mentality_composure, passing, potential, release_clause_eur, dribbling, wage_eur,
            power_shot_power, value_eur, mentality_vision, attacking_short_passing, age, shooting, skill_ball_control, work_rate 
        ]).reshape(1, -1)

        # Scale the inputs
        input_features = scaler.transform(input_features)

        # Make predictions using the model
        prediction = model.predict(input_features)

        # Display the predicted rating and confidence level
        st.write(f'Predicted Rating: {prediction[0]:.2f}')
        



if __name__ == "__main__":
    main()
