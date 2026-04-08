import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AQI Smart Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- MODERN UI --------------------
st.markdown("""
<style>

/* MOBILE */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    h1, h2, h3 { font-size: 18px !important; }
}

/* BACKGROUND */
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.7);
    text-align: center;
    margin-bottom: 15px;
}

/* METRIC */
.stMetric {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    transition: 0.3s;
}
.stMetric:hover {
    transform: scale(1.05);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(20,30,48,0.95);
}

</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🌍 AQI AI Smart Dashboard")

# -------------------- DATA SOURCE --------------------
st.sidebar.header("📁 Data Source")

data_option = st.sidebar.radio(
    "Choose Dataset:",
    ["Default Dataset", "Upload CSV"]
)

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded")
    else:
        st.warning("Upload dataset")
        st.stop()
else:
    df = pd.read_csv("city_day.csv")
    st.success("Default dataset loaded")

# -------------------- CLEANING --------------------
cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']

for col in cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

if 'AQI' in df.columns:
    df['AQI'] = df['AQI'].fillna(df['AQI'].median())

# -------------------- DATE FIX --------------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)

# -------------------- REMOVE STRINGS --------------------
if 'AQI_Bucket' in df.columns:
    df.drop('AQI_Bucket', axis=1, inplace=True)

# -------------------- CITY --------------------
if 'City' in df.columns:
    le = LabelEncoder()
    df['City_Label'] = le.fit_transform(df['City'])
    cities = df['City'].unique()
else:
    cities = ["Unknown"]

# -------------------- FEATURES --------------------
X = df.select_dtypes(include=['number']).drop('AQI', axis=1)
y = df['AQI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- MODEL --------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=80)
    model.fit(X, y)
    return model

model = train_model(X_scaled, y)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙ Controls")

    selected_city_name = st.selectbox("Select City", cities)

    if 'City' in df.columns:
        selected_city = le.transform([selected_city_name])[0]

# -------------------- USER INPUT --------------------
input_data = {}

for col in X.columns:
    if col != "City_Label":
        input_data[col] = st.sidebar.slider(col, 0.0, 500.0, 50.0)

if 'City_Label' in X.columns:
    input_data["City_Label"] = selected_city

input_df = pd.DataFrame(input_data, index=[0])[X.columns]

# -------------------- PREDICTION --------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# -------------------- CATEGORY --------------------
def get_category(aqi):
    if aqi <= 50: return "Good 😊"
    elif aqi <= 100: return "Satisfactory 🙂"
    elif aqi <= 200: return "Moderate 😐"
    elif aqi <= 300: return "Poor 😷"
    elif aqi <= 400: return "Very Poor 🚨"
    else: return "Severe ☠️"

category = get_category(prediction)

# -------------------- HEADER --------------------
st.markdown(f"""
<div class="card">
    <h2>🌆 {selected_city_name}</h2>
    <h1>🌫 AQI: {prediction:.2f}</h1>
    <h3>{category}</h3>
</div>
""", unsafe_allow_html=True)

# -------------------- GAUGE (NO API) --------------------
st.subheader("🎯 AQI Speedometer")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={'text': "Air Quality Index"},
    gauge={
        'axis': {'range': [0, 500]},
        'bar': {'color': "white"},
        'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [50, 100], 'color': "yellow"},
            {'range': [100, 200], 'color': "orange"},
            {'range': [200, 300], 'color': "red"},
            {'range': [300, 500], 'color': "purple"},
        ],
    }
))

fig_gauge.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_gauge, use_container_width=True)

# -------------------- DISTRIBUTION --------------------
st.subheader("📊 AQI Distribution")

fig = px.histogram(df, x="AQI", nbins=50,
                   color_discrete_sequence=["#00C9FF"])

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# -------------------- FEATURE IMPORTANCE --------------------
st.subheader("📈 Feature Importance")

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance")

fig2 = px.bar(
    imp_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="teal"
)

fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# -------------------- TREND --------------------
if all(col in df.columns for col in ['Year','Month','Day']):
    st.subheader("📅 AQI Trend")

    df['Date'] = pd.to_datetime(df[['Year','Month','Day']])

    fig3 = px.line(df.sort_values("Date"),
                   x="Date",
                   y="AQI",
                   color_discrete_sequence=["#00FFAA"])

    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- INSIGHTS --------------------
st.subheader("🤖 AI Insights")

top = imp_df.sort_values(by="Importance", ascending=False).head(3)["Feature"]
st.info(f"Top factors affecting AQI: {', '.join(top)}")

# -------------------- FOOTER --------------------
st.markdown("<div class='footer'>© Copyright 2026 Team ABH Smart AQI Dashboard</div>", unsafe_allow_html=True)