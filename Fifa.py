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

# https://drive.google.com/file/d/1GRQfhBfvexUnAjF16vV_uQd5TWPh5t6N/view?usp=sharing

MODEL_FILE = "fifa_model3.pkl"
if not os.path.exists(MODEL_FILE):
    file_id = "1GRQfhBfvexUnAjF16vV_uQd5TWPh5t6N"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_FILE, quiet=False)

path = "fifa_dataset3.pkl"
# https://drive.google.com/file/d/1WEG7iu3xmo9InUmQmwhilKiyqZdRsmVf/view?usp=sharing
if not os.path.exists(path):
    file_id = "1WEG7iu3xmo9InUmQmwhilKiyqZdRsmVf"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, path, quiet=False)

path2 = "over1.5_model.pkl"
# https://drive.google.com/file/d/1RHj-oXq9YW_9xsKiG7QeAIJXthxys371/view?usp=sharing
if not os.path.exists(path2):
    file_id = "1RHj-oXq9YW_9xsKiG7QeAIJXthxys371"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, path2, quiet=False)



# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# path1 = os.path.join(MODEL_FILE, "fifa_model3.pkl")
# path = os.path.join(BASE_DIR, "fifa_dataset3.pkl")
# path2 = os.path.join(BASE_DIR, "over1.5_model.pkl")

model = joblib.load(MODEL_FILE)
fifa = joblib.load(path)
over1_5 = joblib.load(path2)

teams = fifa['home_team'].unique()
teams1 = fifa['away_team'].unique()
st.header("FIFA WORLD CUP PREDICTION DASHBOARD")
home = st.selectbox("Select Home Team:", teams)
away = st.selectbox("Select Away Team:", teams1)


prediction = st.button('Predict')

home_set = fifa[fifa['home_team'] == home].iloc[0]
away_set = fifa[fifa['away_team'] == away].iloc[0]
    
match = {
    'home_avg_goal': home_set['home_avg_goal'], 'away_avg_goal': away_set['away_avg_goal'], 'home_avg_form':home_set['home_avg_form'], 'away_avg_form':away_set['away_avg_form'], 'home_5_conceded': home_set['home_5_conceded'], 'away_5_conceded': away_set['away_5_conceded']
        }
df = pd.DataFrame([match])
# st.write("Away Set Columns:", away_set.index.tolist())
# st.write("Home Set Columns:", home_set.index.tolist())
if prediction:
    st.subheader("Predicted Outcome:")
    
    # df = scaler.transform(df)
    confidence = model.predict_proba(df)[0]
    overs = over1_5.predict_proba(df)[0]
    # st.write(df)
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.subheader("Double Chance Option")
            # if model.predict(df) == 2:
            #     st.write(f"{home} wins")
            # elif model.predict(df) == 0:
            #     st.write(f"{away} wins")
            # else:
            #     st.write(f"{home} vs {away} Draws")
            if (confidence[2] > confidence[0] and confidence[0] <= confidence[1]):
                st.write(f"{home} wins or Draw")
            elif confidence[2] > confidence[0] and confidence[0] >= confidence[1]:
                st.write(f"{home} wins or {away}")
            elif confidence[2] < confidence[0] and confidence[2] <= confidence[1]:
                st.write(f"{away} wins or Draw")
    
    with col4:
        with st.container(border=True):
            st.subheader("Over/Under Option")
            if over1_5.predict(df) == 1:
                st.write(f"over 1.5 full time")
            else:
                st.write(f"Under 1.5 full time")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader(f"{home}")
            st.write(f"In the Last 5 Matches, {home} scored {home_set['home_avg_goal']*5:.0f} goals (average {home_set['home_avg_goal']}).")
            st.write(f"In the Last 5 Matches, {home} conceded an average {home_set['home_5_conceded']} goals per match.")
        
    with col2:
        with st.container(border=True):
            st.subheader(f"{away}")
            st.write(f"In the Last 5 Matches, {away} scored {away_set['away_avg_goal']*5:.0f} goals (average {away_set['away_avg_goal']}).")
            st.write(f"In the Last 5 Matches, {away} conceded an average {away_set['away_5_conceded']} goals per match.")
        
        
        #     # st.write("Confidence Level Table")

    outcome = {
        "Home Team":home,
        "Away Team":away,
        "Home Level": f"{confidence[2]*100:.0f}%",
        "Away Level": f"{confidence[0]*100:.0f}%",

        "Draw Level": f"{confidence[1]*100:.0f}%"
        }
    level = pd.DataFrame([outcome])
    st.subheader("Confidence Level")
    st.write(level)
