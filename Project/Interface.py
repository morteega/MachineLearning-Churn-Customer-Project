import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Client Churn Detector")
st.write("Introduce the data of the client to give a prediction with the options on the left sidebar.")
@st.cache_resource 
def load_model():
    model=joblib.load('churn_model.pkl')
    columnas=joblib.load('feature_names.pkl')
    return model, columnas
try:
    model, columnas = load_model()
except:
    st.error("Model files not found, please run the training script first")
    st.stop()
st.sidebar.header("Client Data")

def user_input_features():
    st.sidebar.subheader('Client Numerical Data')
    tenure = st.sidebar.slider('Months of Stay (Tenure)', 0, 72, 12)
    monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input('Total Charges', min_value=0.0, value=500.0)
    
    st.sidebar.subheader('Client`s Demography')
    gender=st.sidebar.selectbox('Gender',["Male", "Female"])
    senior=st.sidebar.selectbox('Senior Citizen', [0, 1],format_func=lambda x: 'No' if x==0 else 'Yes')
    partner=st.sidebar.selectbox('Partner', ['No', 'Yes'])
    dependents=st.sidebar.selectbox('Dependents', ['No', 'Yes'])
    
    st.sidebar.subheader('Contracted Services')
    tech_support = st.sidebar.selectbox('Technical Support?', ['No', 'Yes', 'No internet service'])
    online_security = st.sidebar.selectbox('Online Security?', ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup?', ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection?', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV?', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies?', ['No', 'Yes', 'No internet service'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    phone_service = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    
    st.sidebar.subheader('Contract Details and Billing')
    contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    paper_billing=st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method= st.sidebar.selectbox('Payment Method', 
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
  
    data = {col: 0 for col in columnas}
    
    data['tenure'] = tenure
    data['MonthlyCharges'] = monthly_charges
    data['TotalCharges'] = total_charges
    
    binary_map={'No': 0, 'Yes': 1}
    data['SeniorCitizen'] = senior
    data['gender']=0 if gender=='Female' else 1
    data['Partner']=binary_map[partner]
    data['Dependents']=binary_map[dependents]
    
    trinary_map={'No': 0, 'Yes': 1, 'No internet service': 2}
    data['TechSupport'] = trinary_map[tech_support]
    data['OnlineSecurity'] = trinary_map[online_security]
    data['OnlineBackup'] = trinary_map[online_backup]
    data['DeviceProtection'] = trinary_map[device_protection]
    data['StreamingTV'] = trinary_map[streaming_tv]
    data['StreamingMovies'] = trinary_map[streaming_movies]
    data['InternetService'] = {'DSL': 0, 'Fiber optic': 1, 'No': 2}[internet_service]
    data['PhoneService'] = binary_map[phone_service]
    
    data['Contract'] = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract]
    data['PaperlessBilling'] = binary_map[paper_billing]
    data['PaymentMethod'] = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3}[payment_method]
    data['MultipleLines'] =  {'No': 0, 'Yes': 1, 'No phone service': 2}[multiple_lines]

    features=pd.DataFrame(data, index=[0])
    return features

input_df=user_input_features()
backmapdata=input_df.copy()
def backmap(backmapdata):
    reverse_binary_map={0:'No',1:'Yes'}
    reverse_trinary_map={0:'No',1:'Yes',2:'No internet service'}
    reverse_internet_service={0:'DSL',1:'Fiber optic',2:'No'}
    reverse_contract={0:'Month-to-month',1:'One year',2:'Two year'}
    reverse_payment_method={0:'Electronic check',1:'Mailed check',2:'Bank transfer',3:'Credit card'}
    backmapdata['gender'] = backmapdata['gender'].map({0: 'Female', 1: 'Male'})
    backmapdata['SeniorCitizen'] = backmapdata['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    backmapdata['Partner'] = backmapdata['Partner'].map(reverse_binary_map)
    backmapdata['Dependents'] = backmapdata['Dependents'].map(reverse_binary_map)
    backmapdata['TechSupport'] = backmapdata['TechSupport'].map(reverse_trinary_map)
    backmapdata['OnlineSecurity'] = backmapdata['OnlineSecurity'].map(reverse_trinary_map)
    backmapdata['OnlineBackup'] = backmapdata['OnlineBackup'].map(reverse_trinary_map)
    backmapdata['DeviceProtection'] = backmapdata['DeviceProtection'].map(reverse_trinary_map)
    backmapdata['StreamingTV'] = backmapdata['StreamingTV'].map(reverse_trinary_map)
    backmapdata['StreamingMovies'] = backmapdata['StreamingMovies'].map(reverse_trinary_map)
    backmapdata['InternetService'] = backmapdata['InternetService'].map(reverse_internet_service)
    backmapdata['PhoneService'] = backmapdata['PhoneService'].map(reverse_binary_map)
    backmapdata['Contract'] = backmapdata['Contract'].map(reverse_contract)
    backmapdata['PaperlessBilling'] = backmapdata['PaperlessBilling'].map(reverse_binary_map)
    backmapdata['PaymentMethod'] = backmapdata['PaymentMethod'].map(reverse_payment_method)
    backmapdata['MultipleLines'] = backmapdata['MultipleLines'].map({
        0: 'No',
        1: 'Yes',
        2: 'No phone service'
    })
    return backmapdata
backmapdata=backmap(backmapdata)
st.subheader('Summary of the Client')
st.dataframe(backmapdata, use_container_width=True) 
if st.button('Calculate Churn Risk'):
    prediction=model.predict(input_df)
    probability=model.predict_proba(input_df)
    churn_risk = probability[0][1] 
    st.divider()
    
    if churn_risk > 0.5:
        st.error(f"Risk of churn ({churn_risk*100:.2f}%)")
        st.write("Client is likely to leave. Consider retention strategies.")
    else:
        st.success(f"Safe Client ({churn_risk*100:.2f}% of risk)")
        st.write("Client seems to be happy.")