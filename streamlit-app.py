###
# Titanic Survival Prediction with Password Protection
###

import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ===== Modern UI Styling =====
st.markdown("""
    <style>

    /* Global background */
    .main {
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1f3c88 0%, #2e5aac 100%);
        color: white;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Buttons */
    .stButton>button {
        background: #4c8bf5;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        transition: 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background: #2f6fed;
        transform: scale(1.02);
    }

    /* Prediction message containers */
    .stSuccess, .stError {
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.1rem;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #1a2e66;
    }

    /* Tables */
    thead tr th {
        background-color: #4c8bf5 !important;
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)


# =========================
# Password check
# =========================
password = st.sidebar.text_input("Enter password", type="password")
if password != "1234":
    st.warning("🔒 Please enter the correct password to use the app.")
    st.stop()  # Stops the rest of the app if password is wrong

# =========================
# Load machine learning model
# =========================
tree_clf = joblib.load('model_dt.pickle')

# =========================
# Main page
# =========================
st.title('Titanic Survival Prediction')
st.markdown('Use the sidebar to enter passenger details and click **Predict**.')

# =========================
# Sidebar inputs
# =========================
st.sidebar.header('Passenger Information')

sex = st.sidebar.selectbox('Sex', ['female', 'male'])

# ✅ Date of Birth input (replaces Age)
dob = st.sidebar.date_input(
    'Date of Birth',
    value=date(2000, 1, 1),
    min_value=date(1900, 1, 1),
    max_value=date.today()
)

# Calculate age from DOB (model still needs Age)
today = date.today()
age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# Optional: show calculated age
st.sidebar.caption(f'Calculated age: {age}')

sib_sp = int(st.sidebar.number_input(
    '# of siblings / spouses aboard',
    min_value=0,
    max_value=10,
    value=0
))

pclass = st.sidebar.selectbox('Ticket Class', [1, 2, 3])

# =========================
# Simplified Fare Selection
# =========================
fare_range = st.sidebar.selectbox(
    'Approximate Ticket Price',
    [
        'Very Low (≤8)',
        'Low (8.01–15)',
        'Medium (15.01–30)',
        'High (30.01–55)',
        'Very High (>55)'
    ]
)

FARE_MAP = {
    'Very Low (≤8)': 8,
    'Low (8.01–15)': 12,
    'Medium (15.01–30)': 22,
    'High (30.01–55)': 40,
    'Very High (>55)': 75
}

fare = FARE_MAP[fare_range]

predict_button = st.sidebar.button('Predict')

# =========================
# Prediction logic
# =========================
if predict_button:

    passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sib_sp],
        'Fare': [fare]
    })

    y_pred = tree_clf.predict(passenger)
    proba = tree_clf.predict_proba(passenger)

    st.subheader('Prediction Result')

    if y_pred[0] == 0:
        st.error('This passenger is predicted to have **died**.')
        st.image(
            'https://miro.medium.com/1*zyYiHYZlMvKLRaFZXellAQ.gif',
            caption='Sad outcome',
            use_column_width=True
        )
    else:
        st.success('This passenger is predicted to have **survived**.')

    st.markdown(f'### Survival Probability: **{proba[0][1]:.2%}**')

# =========================
# Optional: Show sample data
# =========================
with st.expander('Show sample of Titanic dataset'):
    df = pd.read_csv('titanic.csv')
    st.dataframe(df.head(20))
