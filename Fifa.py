import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import gdown


# path =r"C:\Users\hp\OneDrive\Desktop\Project\fifa_dataset3.pkl"
# path1 =r"C:\Users\hp\OneDrive\Desktop\Project\fifa_model3.pkl"
# path2 =r"C:\Users\hp\OneDrive\Desktop\Project\scaler.pkl"

# https://drive.google.com/file/d//

MODEL_FILE = "fifa_model3.pkl"

if not os.path.exists(MODEL_FILE):
    file_id = "1fDbC8Q5-nFZAQ-8KOTI5yVHF7ntTMvL1"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_FILE, quiet=False)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# path1 = os.path.join(MODEL_FILE, "fifa_model3.pkl")
path = os.path.join(BASE_DIR, "fifa_dataset3.pkl")
# path2 = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_FILE)
fifa = joblib.load(path)
# scaler = joblib.load(path2)

teams = fifa['home_team'].unique()
teams1 = fifa['away_team'].unique()
st.header("FIFA WORLD CUP PREDICTION DASHBOARD")
home = st.selectbox("Select Home Team:", teams)
away = st.selectbox("Select Away Team:", teams1)


prediction = st.button('Predict')

if prediction:
    st.subheader("Predicted Outcome:")
    home_set = fifa[fifa['home_team'] == home].iloc[0]
    away_set = fifa[fifa['away_team'] == away].iloc[0]
    match = {
        'home_avg_goal': home_set['home_avg_goal'], 'away_avg_goal': away_set['away_avg_goal'], 'home_avg_form':home_set['home_avg_form'], 'away_avg_form':away_set['away_avg_form'], 'home_5_conceded': home_set['home_5_conceded'], 'away_5_conceded': away_set['away_5_conceded']
        }
    df = pd.DataFrame([match])
    # df = scaler.transform(df)
    confidence = model.predict_proba(df)[0]
    # st.write(df)

    if model.predict(df) == 2:
        st.write(f"******* {home} wins *******")
    elif model.predict(df) == 0:
        st.write(f"******* {away} wins ********")
    else:
        st.write(f"******* {home} vs {away} Draws *******")
    # with col2:
    #     # st.write("Confidence Level Table")

    outcome = {
        "Home Team":home,
        "Away Team":away,
        "Home Level": f"{confidence[2]*100:.0f}%",
        "Away Level": f"{confidence[0]*100:.0f}%",

        "Draw Level": f"{confidence[1]*100:.0f}%"
        }
    level = pd.DataFrame([outcome])
    st.write("""________Confidence Level_______""")
    st.write(level)
