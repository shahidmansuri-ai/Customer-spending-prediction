#Streamlit UI

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "best_model" / "best_model.pkl"
DATA_PATH = ROOT_DIR / "datasets" / "data_cleaned.csv"
LOGO_PATH = ROOT_DIR / "assets" / "logo.png"

st.set_page_config(
    page_title="Customer Spending Prediction",
    page_icon=str(LOGO_PATH),
)

with MODEL_PATH.open("rb") as file:
    best_pipeline = pickle.load(file)


#load dataset
df = pd.read_csv(DATA_PATH)


with st.sidebar:
    st.title("Cutomer Spending Prediction")
    st.image(str(LOGO_PATH), width=200)





with st.container(border=True):
    col1,col2 = st.columns(2)
    with col1:
        Age  =st.number_input("Age:",min_value=1,max_value=100) 
        Annual_Income=st.number_input("Annual Income : ",min_value=100000)
        Website_Visits=st.number_input("Website Visits : ",min_value=1)
        Previous_Purchases=st.number_input("Previous Purchases :")

    with col2:
        Membership_Type=st.selectbox("Membership:",options=df["Membership_Type"].unique())
        Preferred_Category=st.selectbox("Preferred Category:",options=df["Preferred_Category"].unique())
        City_Tier=st.selectbox("City Tier:",options=df["City_Tier"].unique())
        Discount_Used=st.selectbox("Discount Used:",options=df["Discount_Used"].unique())


#new data

new_data = {'Age':Age, 'Annual_Income':Annual_Income, 'Website_Visits':Website_Visits, 'Previous_Purchases':Previous_Purchases,
       'Membership_Type':Membership_Type, 'Preferred_Category':Preferred_Category, 'City_Tier':City_Tier, 'Discount_Used':Discount_Used}


new_data = pd.DataFrame(new_data,index=[0])

if st.button("Predict Spend"):
    prediction= best_pipeline.predict(new_data)[0]
    prediction = round(prediction,ndigits=2)
    st.header(f"Prediction spend :{prediction} Rs.")