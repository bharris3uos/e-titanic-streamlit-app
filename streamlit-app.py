
### Titanic Survival Prediction with Password Protection ###

import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ======================================================
# MODERN UI + FIX FOR WHITE TEXT IN INPUT FIELDS
# ======================================================
st.markdown("""
    <style>

    /* Fix white-on-white input text */
    input, textarea, select, option {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    .stTextInput input, 
    .stNumberInput input, 
    .stPassword input, 
    .stDateInput input, 
    .stSelectbox div[data-baseweb="select"] * {
        color: #000000 !important;
    }

    ::placeholder {
        color: #444444 !important;
    }

    /* Sidebar input text (fix) */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    /* Modern blue sidebar theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1f3c88 0%, #2e5aac 100%);
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Page background */
    .main {
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
    }

    /* Buttons styling */
    .stButton>button {
        background: #4c8bf5;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: #2f6fed;
        transform: scale(1.02);
    }

    /* Headers styling */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #1a2e66;
    }

    /* Data table header */
    thead tr th {
        background-color: #4c8bf5 !important;
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)


# ===== Full-screen Background + Frosted Glass =====
st.markdown("""
    <style>
    /* Full-page background image */
    .stApp {
        background: url('https://www.bbc.co.uk/newsround/68569886') no-repeat center center fixed !important;
        background-size: cover !important;
    }

    /* Optional: If external URL is blocked, comment ^ and use a local image:
       .stApp { background: url('background.jpg') no-repeat center center fixed !important; background-size: cover !important; }
    */

    /* Frosted glass containers (main + sidebar) */
    .main, [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.55) !important; /* translucent white layer */
        -webkit-backdrop-filter: blur(8px) saturate(110%) !important;
        backdrop-filter: blur(8px) saturate(110%) !important;
        border-radius: 12px;
    }

    /* Improve spacing so the glass panels don't stick to the edges */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stSidebar"] {
        padding: 1rem 0.6rem !important;
    }

    /* Make widgets also sit on subtle glass cards */
    .stContainer, .stRadio, .stSelectbox, .stNumberInput, .stDateInput, .stTextInput, .stPassword, .stFileUploader, .stDataFrame {
        background: rgba(255, 255, 255, 0.50) !important;
        border-radius: 10px;
        padding: 0.25rem 0.35rem;
    }

    /* Keep data table readable */
    thead tr th {
        background: rgba(76, 139, 245, 0.9) !important;
        color: #fff !important;
    }

    /* Keep buttons visible atop glass */
    .stButton>button {
        position: relative;
        z-index: 1;
    }
    </style>
""", unsafe_allow_html=True)
``


# ======================================================
# PASSWORD PROTECTION
# ======================================================
password = st.sidebar.text_input("Enter password", type="password")
if password != "1234":
    st.warning("🔒 Please enter the correct password to use the app.")
    st.stop()

# ======================================================
# LOAD MODEL
# ======================================================
tree_clf = joblib.load('model_dt.pickle')

# ======================================================
# MAIN PAGE TITLE
# ======================================================
st.title("Titanic Survival Prediction")
st.markdown("Use the sidebar to enter passenger details and click **Predict**.")

# ======================================================
# SIDEBAR INPUTS
# ======================================================
st.sidebar.header("Passenger Information")

sex = st.sidebar.selectbox("Sex", ["female", "male"])

# --- Replace Age with Date of Birth ---
dob = st.sidebar.date_input(
    "Date of Birth",
    value=date(2000, 1, 1),
    min_value=date(1900, 1, 1),
    max_value=date.today()
)

# Calculate age from DOB
today = date.today()
age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
st.sidebar.caption(f"Calculated age: {age}")

sib_sp = int(st.sidebar.number_input(
    "# of siblings / spouses aboard",
    min_value=0,
    max_value=10,
    value=0
))

pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])

# Simplified fare selection
fare_range = st.sidebar.selectbox(
    "Approximate Ticket Price",
    [
        "Very Low (≤8)",
        "Low (8.01–15)",
        "Medium (15.01–30)",
        "High (30.01–55)",
        "Very High (>55)"
    ]
)

FARE_MAP = {
    "Very Low (≤8)": 8,
    "Low (8.01–15)": 12,
    "Medium (15.01–30)": 22,
    "High (30.01–55)": 40,
    "Very High (>55)": 75
}

fare = FARE_MAP[fare_range]

predict_button = st.sidebar.button("Predict")

# ======================================================
# PREDICTION LOGIC
# ======================================================
if predict_button:
    passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sib_sp],
        "Fare": [fare]
    })

    y_pred = tree_clf.predict(passenger)
    proba = tree_clf.predict_proba(passenger)

    st.subheader("Prediction Result")

    if y_pred[0] == 0:
        st.error("This passenger is predicted to have **died**.")
        st.image(
            "https://miro.medium.com/1*zyYiHYZlMvKLRaFZXellAQ.gif",
            caption="Sad outcome",
            use_column_width=True
        )
    else:
        st.success("This passenger is predicted to have **survived**.")

    st.markdown(f"### Survival Probability: **{proba[0][1]:.2%}**")

# ======================================================
# SHOW SAMPLE DATA
# ======================================================
with st.expander("Show sample of Titanic dataset"):
    df = pd.read_csv("titanic.csv")
    st.dataframe(df.head(20))

# ======================================================
# ABOUT THE MODEL
# ======================================================
LAST_TRAINING_DATE = "2026-02-20"
FEATURES_USED = ["Pclass", "Sex", "Age", "SibSp", "Fare"]

with st.expander("ℹ️ About the model"):
    st.markdown("**Model:** Decision Tree Classifier (`model_dt.pickle`)")
    st.markdown(f"**Features used:** {', '.join(FEATURES_USED)}")
    st.markdown(f"**Last training date:** {LAST_TRAINING_DATE}")

    try:
        training_df = pd.read_csv("titanic.csv")
        st.markdown(f"**Training data size:** {len(training_df)} rows")
    except:
        st.markdown("**Training data size:** unavailable")

    st.markdown("### Limitations")
    st.markdown("- The model uses historical Titanic data and may not generalize to other contexts.")
    st.markdown("- Decision trees can overfit; predictions may vary with small input changes.")
    st.markdown("- Inputs like fare and class are approximations and may affect accuracy.")
    st.markdown("- The model does not consider all real-world survival factors (e.g., crew assistance, cabin location).")

