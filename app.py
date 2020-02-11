# Primary Packages
import pandas as pd
import numpy as np
import streamlit as st

# Package to Load Model
from sklearn.externals import joblib

@st.cache
def load_data():
    return pd.read_csv('data/Sell-Out Data.csv')

data = load_data()
data = data[data['COMPANY']!='Competitors']


### Set Title
st.title("Cortex Sales Predictor")
st.write("""From the AC metadata, we built a machine learning-based forecasting model 
to predict ac unit sales.""")

# Show data
st.subheader('AC Metadata')
if st.checkbox('Show Raw Data'):
    st.write(data.head(20))

st.sidebar.title('Parameters')


# Unit Level
unit_values = pd.Series(data['UNIT'].unique()).str.strip()
unit_dummies = pd.get_dummies(unit_values)

unit_sample = st.sidebar.selectbox("AC Unit", unit_values.values.tolist())

unit_sample_dummies = (unit_dummies.loc[np.where(unit_values.values == unit_sample)[0]]
                                  .values.tolist()[0])


# Dealer Level
dealer_values = pd.Series(data['DEALER'].unique()).str.strip()
dealer_dummies = pd.get_dummies(dealer_values)

dealer_sample = st.sidebar.selectbox("Dealer", dealer_values.values.tolist())

dealer_sample_dummies = (dealer_dummies.loc[np.where(dealer_values.values == dealer_sample)[0]]
                                  .values.tolist()[0])


# Channel Level
channel_values = pd.Series(data['CHANNEL'].unique()).str.strip()
channel_dummies = pd.get_dummies(channel_values)

channel_sample = st.sidebar.selectbox("Channel", channel_values.values.tolist())

channel_sample_dummies = (channel_dummies.loc[np.where(channel_values.values == channel_sample)[0]]
                                  .values.tolist()[0])


# Sales Channel Level
sales_channel_values = pd.Series(data['SALES CHANNEL'].unique()).str.strip()
sales_channel_dummies = pd.get_dummies(sales_channel_values)

sales_channel_sample = st.sidebar.selectbox("Sales Channel", sales_channel_values.values.tolist())

sales_channel_sample_dummies = (sales_channel_dummies.loc[np.where(sales_channel_values.values == sales_channel_sample)[0]]
                                  .values.tolist()[0])


# Region Level
region_values = pd.Series(data['REGION'].unique()).str.strip()
region_dummies = pd.get_dummies(region_values)

region_sample = st.sidebar.selectbox("Region", region_values.values.tolist())

region_sample_dummies = (region_dummies.loc[np.where(region_values.values == region_sample)[0]]
                                  .values.tolist()[0])


# Brand Level
brand_values = pd.Series(data['BRAND'].unique()).str.strip()
brand_dummies = pd.get_dummies(brand_values)

brand_sample = st.sidebar.selectbox("Brand", brand_values.values.tolist())

brand_sample_dummies = (brand_dummies.loc[np.where(brand_values.values == brand_sample)[0]]
                                  .values.tolist()[0])

# Capacity Level
cap_values = pd.Series(data['CAPACITY'].unique()).str.strip()
cap_dummies = pd.get_dummies(cap_values)

cap_sample = st.sidebar.selectbox("Capacity", cap_values.values.tolist())

cap_sample_dummies = (cap_dummies.loc[np.where(cap_values.values == cap_sample)[0]]
                                  .values.tolist()[0])


# Compressor Level
comp_values = pd.Series(data['COMPRESSOR'].unique()).str.strip()
comp_dummies = pd.get_dummies(comp_values)

comp_sample = st.sidebar.selectbox("Compressor", comp_values.values.tolist())

comp_sample_dummies = (comp_dummies.loc[np.where(comp_values.values == comp_sample)[0]]
                                  .values.tolist()[0])

# Prediction
st.title("Predicted Sales")

# Load Model
model = joblib.load('model/cortex_model.sav')

# Input
sample_features = (unit_sample_dummies + dealer_sample_dummies + channel_sample_dummies + sales_channel_sample_dummies +
                   region_sample_dummies + brand_sample_dummies + cap_sample_dummies + comp_sample_dummies + [1])

# Make Prediction
prediction = model.predict([sample_features])[0]

# Write Prediction
st.write('Predicted sales is %s units.' % int(prediction))