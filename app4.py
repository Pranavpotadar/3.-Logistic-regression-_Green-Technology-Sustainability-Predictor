import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('lrmodel_sustainable.pkl')

# Set page configuration
st.set_page_config(
    page_title="Green Tech Sustainability Predictor",
    page_icon="🌱",
    layout="wide"
)

# Add title and description
st.title("🌱 Green Technology Sustainability Predictor")
st.markdown("""
This application predicts whether a green technology solution is sustainable based on key metrics:
- Carbon Emissions
- Energy Output
- Renewability Index
- Cost Efficiency
""")

# Create input form
st.subheader("Enter Technology Metrics")

col1, col2 = st.columns(2)

with col1:
    carbon_emissions = st.number_input(
        "Carbon Emissions (units)",
        min_value=0.0,
        max_value=500.0,
        value=200.0,
        help="Enter carbon emissions value between 0 and 500"
    )
    
    energy_output = st.number_input(
        "Energy Output (units)",
        min_value=0.0,
        max_value=1000.0,
        value=500.0,
        help="Enter energy output value between 0 and 1000"
    )

with col2:
    renewability_index = st.slider(
        "Renewability Index",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Scale from 0 (non-renewable) to 1 (fully renewable)"
    )
    
    cost_efficiency = st.number_input(
        "Cost Efficiency Score",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        help="Enter cost efficiency score between 0 and 5"
    )

# Create prediction button
if st.button("Predict Sustainability"):
    # Prepare input data
    input_data = pd.DataFrame({
        'carbon_emissions': [carbon_emissions],
        'energy_output': [energy_output],
        'renewability_index': [renewability_index],
        'cost_efficiency': [cost_efficiency]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.success("🌟 This technology is predicted to be SUSTAINABLE!")
    else:
        st.error("⚠️ This technology is predicted to be NOT SUSTAINABLE")
    
    # Display prediction probability
    st.write("Prediction Probability:")
    prob_df = pd.DataFrame({
        'Category': ['Not Sustainable', 'Sustainable'],
        'Probability': probability
    })
    st.bar_chart(prob_df.set_index('Category'))
    
    # Display feature importance
    st.subheader("Feature Analysis")
    features = ['Carbon Emissions', 'Energy Output', 'Renewability Index', 'Cost Efficiency']
    coefficients = model.coef_[0]
    
    coef_df = pd.DataFrame({
        'Feature': features,
        'Impact': coefficients
    })
    coef_df = coef_df.sort_values('Impact', ascending=True)
    
    st.bar_chart(coef_df.set_index('Feature'))
    
    st.info("""
    💡 **Understanding Feature Impact:**
    - Positive values indicate the feature contributes to sustainability
    - Negative values indicate the feature detracts from sustainability
    - Larger absolute values indicate stronger influence
    """)

# Add footer with additional information
st.markdown("""
---
### About this Model
This predictor uses a Logistic Regression model trained on green technology data. 
The model considers multiple factors to determine sustainability:
- Carbon emissions impact
- Energy generation efficiency
- Renewable resource usage
- Cost effectiveness

For best results, ensure all input values are within the specified ranges.
""")