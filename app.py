from pathlib import Path
import pickle

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import sklearn

st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Breast Cancer",
    page_icon=":lungs:",
)

@st.cache_data
def load_image(image_path):
    image = Image.open(image_path)
    # обрезка до нужного размера с сохранинием пропорций
    MAX_SIZE = (600, 400)
    image.thumbnail(MAX_SIZE)
    return image

@st.cache_data
def load_model(model_path):
    # загрузка сериализованной модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


image_path = Path.cwd() / 'lungs.jpg'
image = load_image(image_path)

model_path = Path.cwd() / 'model.pkl'
model = load_model(model_path)

st.write(
    """
    # Диагностика рака легких
    Введите ваши данные и получите результат
    """
)

st.image(image)

st.sidebar.header('Входные данные пользователя')

features_dict = {
    "Age": "Age",
    "Gender": "Gender",
    "air_pollution": "air_pollution",
    "alcohol_use": "alcohol_use",
    "dust_allergy": "dust_allergy",
    "occupational_hazards": "occupational_hazards",
    "genetic_risk": "genetic_risk",
    "chronic_lung_disease": "chronic_lung_disease",
    "balanced_diet": "balanced_diet",
    "Obesity": "Obesity",
    "Smoking": "Smoking",
    "passive_smoker": "passive_smoker",
    "chest_pain": "chest_pain",
    "coughing_of_blood": "coughing_of_blood",
    "Fatigue": "Fatigue",
    "weight_loss": "weight_loss",
    "shortness_of_breath": "shortness_of_breath",
    "Wheezing": "Wheezing", 
    "swallowing_difficulty": "swallowing_difficulty",
    "clubbing_of_finger_nails": "clubbing_of_finger_nails",
    "frequent_cold": "frequent_cold",
    "dry_cough": "dry_cough",
    "Snoring": "Snoring",
}

Age = st.sidebar.slider(features_dict['Age'], min_value=1, max_value=100, value=33, step=1)
Gender = st.sidebar.number_input(features_dict['Gender'], min_value=1, max_value=2, value=2, step=1)
air_pollution = st.sidebar.number_input(features_dict['air_pollution'], min_value=1, max_value=10, value=1, step=1)
alcohol_use = st.sidebar.number_input(features_dict['alcohol_use'], min_value=1, max_value=10, value=6, step=1)
dust_allergy = st.sidebar.number_input(features_dict['dust_allergy'], min_value=1, max_value=10, value=7, step=1)
occupational_hazards = st.sidebar.number_input(features_dict['occupational_hazards'], min_value=1, max_value=10, value=8, step=1)
genetic_risk = st.sidebar.number_input(features_dict['genetic_risk'], min_value=1, max_value=10, value=7, step=1)
chronic_lung_disease = st.sidebar.number_input(features_dict['chronic_lung_disease'], min_value=1, max_value=10, value=6, step=1)
balanced_diet = st.sidebar.number_input(features_dict['balanced_diet'], min_value=1, max_value=10, value=7, step=1)
Obesity = st.sidebar.number_input(features_dict['Obesity'], min_value=1, max_value=10, value=7, step=1)
Smoking = st.sidebar.number_input(features_dict['Smoking'], min_value=1, max_value=10, value=3, step=1)
passive_smoker = st.sidebar.number_input(features_dict['passive_smoker'], min_value=1, max_value=10, value=4, step=1)
chest_pain = st.sidebar.number_input(features_dict['chest_pain'], min_value=1, max_value=10, value=8, step=1)
coughing_of_blood = st.sidebar.number_input(features_dict['coughing_of_blood'], min_value=1, max_value=10, value=7, step=1)
Fatigue = st.sidebar.number_input(features_dict['Fatigue'], min_value=1, max_value=10, value=3, step=1)
weight_loss = st.sidebar.number_input(features_dict['weight_loss'], min_value=1, max_value=10, value=2, step=1)
shortness_of_breath = st.sidebar.number_input(features_dict['shortness_of_breath'], min_value=1, max_value=10, value=6, step=1)
Wheezing = st.sidebar.number_input(features_dict['Wheezing'], min_value=1, max_value=10, value=4, step=1)
swallowing_difficulty = st.sidebar.number_input(features_dict['swallowing_difficulty'], min_value=1, max_value=10, value=2, step=1)
clubbing_of_finger_nails = st.sidebar.number_input(features_dict['clubbing_of_finger_nails'], min_value=1, max_value=10, value=3, step=1)
frequent_cold = st.sidebar.number_input(features_dict['frequent_cold'], min_value=1, max_value=10, value=1, step=1)
dry_cough = st.sidebar.number_input(features_dict['dry_cough'], min_value=1, max_value=10, value=2, step=1)
Snoring = st.sidebar.number_input(features_dict['Snoring'], min_value=1, max_value=10, value=1, step=1)

data = {
    'Age': Age,
    'Gender': Gender,
    'air_pollution': air_pollution,
    'alcohol_use': alcohol_use,
    'dust_allergy': dust_allergy,
    'occupational_hazards': occupational_hazards,
    'genetic_risk': genetic_risk,
    'chronic_lung_disease': chronic_lung_disease,
    'balanced_diet': balanced_diet,
    'Obesity': Obesity,
    'Smoking': Smoking,
    'passive_smoker': passive_smoker,
    'chest_pain': chest_pain,
    'coughing_of_blood': coughing_of_blood,
    'Fatigue': Fatigue,
    'weight_loss': weight_loss,
    'shortness_of_breath': shortness_of_breath,
    'Wheezing': Wheezing,
    'swallowing_difficulty': swallowing_difficulty,
    'clubbing_of_finger_nails': clubbing_of_finger_nails,
    'frequent_cold': frequent_cold,
    'dry_cough': dry_cough,
    'Snoring': Snoring,
}

df = pd.DataFrame(data, index=[0])

st.write("### Ваши данные")
st.write(df)

predicted_level = model.predict(df.values)

st.write("### Степень тяжести рака: ")
st.write(f'{predicted_level[0]}')