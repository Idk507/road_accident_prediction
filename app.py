import streamlit as st
import pandas as pd
import numpy as np
import folium
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from streamlit_folium import folium_static
from PIL import Image
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import random
#st.title("Road Accident Prediction and Visualization")
st.set_page_config(
    page_title="Road Accident Prediction and Visualization",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

model = pickle.load(open('model.pkl', 'rb'))
header_img = Image.open('header.png')
#st.image(header_img, width=700, caption='Road Accident Prediction and Visualization', use_column_width=True)
st.image('./video.gif', width=200, caption='Road Accident Prediction and Visualization', use_column_width=True)
quotes = [
    "Drive safely, reach safely",
    "Safety is never an accident",
    "Accidents hurt - safety doesn't",
    "Stay alert, don't get hurt",
    "Arrive alive, don't text and drive",
    "Your family is waiting for you, drive responsibly",
    "Buckle up - it's the law and your life",
    "Be aware, stay alive",
    "Don't let a moment of distraction cause a lifetime of regret",
]

# List of suggestions based on severity
suggestions = {
    'Slight': [
        "Drive slowly in bad weather",
        "Keep a safe following distance",
        "Check your vehicle regularly for maintenance",
        "Avoid sudden lane changes",
    ],
    'Serious': [
        "Avoid using the phone while driving",
        "Follow traffic rules and speed limits",
        "Always use seat belts",
        "Get regular eye check-ups",
    ],
    'Fatal': [
        "Please be careful!",
        "Remember, your loved ones are waiting for you",
        "Don't risk your life for a momentary thrill",
        "Reckless driving has irreversible consequences",
    ],
}

# Icons for severity levels
icons = {
    'Slight': ":slightly_smiling_face:",
    'Serious': ":worried:",
    'Fatal': ":pensive:",
}

# Function to preprocess input data for prediction
def preprocess_input_data(data):
    label_encoder = LabelEncoder()
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour
    data['Time'] = label_encoder.fit_transform(data['Time'])
    data['Age_band_of_driver'] = data['Age_band_of_driver'].replace('Unknown', 'Over 51')
    data['Age_band_of_driver'] = label_encoder.fit_transform(data['Age_band_of_driver'])
    data['Sex_of_driver'] = label_encoder.fit_transform(data['Sex_of_driver'])
    data['Educational_level'] = label_encoder.fit_transform(data['Educational_level'])
    data['Vehicle_driver_relation'] = label_encoder.fit_transform(data['Vehicle_driver_relation'])
    data['Driving_experience'] = label_encoder.fit_transform(data['Driving_experience'])
    data['Type_of_vehicle'] = label_encoder.fit_transform(data['Type_of_vehicle'])
    data['Area_accident_occured'] = label_encoder.fit_transform(data['Area_accident_occured'])
    data['Road_allignment'] = label_encoder.fit_transform(data['Road_allignment'])
    data['Road_surface_conditions'] = label_encoder.fit_transform(data['Road_surface_conditions'])
    data['Light_conditions'] = label_encoder.fit_transform(data['Light_conditions'])
    data['Weather_conditions'] = label_encoder.fit_transform(data['Weather_conditions'])
    data['Day_of_week'] = label_encoder.fit_transform(data['Day_of_week'])


    return data

# Function to predict accident severity
def predict_accident_severity(data):
    data = preprocess_input_data(data)
    columns_to_drop = ["Owner_of_vehicle", 'Service_year_of_vehicle', 'Defect_of_vehicle', 'Lanes_or_Medians',
                       'Road_surface_type', 'Type_of_collision', 'Number_of_vehicles_involved', 'Casualty_class',
                       'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity', 'Work_of_casuality',
                       'Fitness_of_casuality', 'Cause_of_accident', 'Number_of_casualties', 'Types_of_Junction',
                       'Pedestrian_movement', 'Vehicle_movement']

    data = data.drop(columns_to_drop, axis=1)
    #st.write("Making prediction...")
    prediction = model.predict(data)
    severity_mapping = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
    decoded_prediction = [severity_mapping[pred] for pred in prediction]

    return decoded_prediction
def visualize_accident_map(data):
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    popup_text = f"Year: {data['Year']}<br>Speed Limit: {data['Speed_limit']}<br>Weather Condition: {data['Weather_Conditions']}<br>Time: {data['Time']}"
    folium.Marker(location=[data['Latitude'], data['Longitude']], popup=popup_text).add_to(india_map)
    
    st.header("Accident Location")
    folium_static(india_map)
    return india_map


st.title("Road Accident Prediction and Visualization")

# Sidebar for user input
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.sidebar.subheader("Uploaded Data:")
    st.sidebar.write(input_data)
    prediction_button = st.sidebar.button("Predict Accident Severity")
    if prediction_button:
        prediction_result = predict_accident_severity(input_data)
        #st.subheader("Accident Severity Prediction:")
        #st.write(prediction_result)
        st.header("Predicted Accident Severity is: {} ".format(prediction_result))
        if prediction_result[0] == 'Slight':
            st.info(random.choice(suggestions['Slight']))
        elif prediction_result[0] == 'Serious':
            st.warning(random.choice(quotes))
        elif prediction_result[0] == 'Fatal':
            st.error("Please be careful!")


    visualize_button = st.sidebar.button("Visualize Accident Map")
    if visualize_button:
        map_result = visualize_accident_map(input_data)
        #st.subheader("Accident Map Visualization:")
        #st.write(map_result)
else:
    st.info("Please upload a CSV file to get started.")
st.sidebar.markdown("---")
st.sidebar.info("Powered by Pixels_Infinity")
footer_img = st.sidebar.image('./footer.png', caption='Road Accident Prediction and Visualization', )