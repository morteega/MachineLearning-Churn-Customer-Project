import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Fugas", page_icon="🔮")

st.title("🔮 Detector de Fugas de Clientes")
st.write("Introduce los datos del cliente para saber si tiene riesgo de irse.")

# 2. Cargamos el modelo y las columnas
@st.cache_resource # Esto hace que no se recargue todo el rato
def cargar_modelo():
    model = joblib.load('churn_model.pkl')
    columnas = joblib.load('feature_names.pkl')
    return model, columnas

try:
    model, columnas = cargar_modelo()
except:
    st.error("⚠️ No se encontraron los archivos .pkl. Ejecuta primero App.py para generar el modelo.")
    st.stop()

# 3. Formulario para el usuario (Sidebar)
st.sidebar.header("Datos del Cliente")

def user_input_features():
    # --- VARIABLES NUMÉRICAS ---
    tenure = st.sidebar.slider('Meses de Permanencia (Tenure)', 0, 72, 12)
    monthly_charges = st.sidebar.number_input('Cargos Mensuales ($)', min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input('Cargos Totales ($)', min_value=0.0, value=500.0)
    
    # --- VARIABLES CATEGÓRICAS (Las más importantes) ---
    # Nota: Aquí simulamos la conversión manual para simplificar
    # En un entorno real, usaríamos el mismo encoder, pero para este demo esto funciona perfecto.
    
    contract = st.sidebar.selectbox('Tipo de Contrato', ['Month-to-month', 'One year', 'Two year'])
    tech_support = st.sidebar.selectbox('¿Tiene Soporte Técnico?', ['No', 'Yes', 'No internet service'])
    online_security = st.sidebar.selectbox('¿Tiene Seguridad Online?', ['No', 'Yes', 'No internet service'])
    payment_method = st.sidebar.selectbox('Método de Pago', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    internet_service = st.sidebar.selectbox('Servicio de Internet', ['DSL', 'Fiber optic', 'No'])
    
    # Creamos un diccionario con TODAS las columnas que espera el modelo
    # Inicializamos todo a 0 por defecto
    data = {col: 0 for col in columnas}
    
    # Rellenamos con lo que ha puesto el usuario
    # OJO: Aquí hacemos una simplificación. El Random Forest necesita números.
    # Como tu preprocesador usaba LabelEncoder (0, 1, 2...), vamos a hacer un mapeo rápido
    # para que la app funcione visualmente.
    
    data['tenure'] = tenure
    data['MonthlyCharges'] = monthly_charges
    data['TotalCharges'] = total_charges
    
    # Mapeos manuales (Basados en cómo suele ordenar LabelEncoder alfabéticamente)
    # Month-to-month=0, One year=1, Two year=2
    map_contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    data['Contract'] = map_contract[contract]
    
    # No=0, No internet=1, Yes=2
    map_yes_no = {'No': 0, 'No internet service': 1, 'Yes': 2}
    data['TechSupport'] = map_yes_no[tech_support]
    data['OnlineSecurity'] = map_yes_no[online_security]
    
    # DSL=0, Fiber=1, No=2
    map_internet = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    data['InternetService'] = map_internet[internet_service]
    
    # Electronic=2, Mailed=3, Bank=0, Credit=1 (Aproximación por orden alfabético)
    map_pay = {'Bank transfer': 0, 'Credit card': 1, 'Electronic check': 2, 'Mailed check': 3}
    data['PaymentMethod'] = map_pay[payment_method]

    # Convertimos a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 4. Mostrar lo que ha elegido el usuario
st.subheader('Resumen del Cliente')
st.write(input_df)

# 5. Botón de Predicción
if st.button('🚀 Calcular Riesgo de Churn'):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    churn_risk = probability[0][1] # Probabilidad de que sea 1 (Churn)
    
    st.divider()
    
    if churn_risk > 0.5:
        st.error(f"🚨 ALERTA: ¡Alto riesgo de fuga! ({churn_risk*100:.2f}%)")
        st.write("Se recomienda ofrecer un descuento o llamar al cliente.")
    else:
        st.success(f"✅ Cliente Seguro ({churn_risk*100:.2f}% de riesgo)")
        st.write("El cliente parece estar contento.")