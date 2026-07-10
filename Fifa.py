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

# https://drive.google.com/file/d/1S8qcDReelgokkHFJQ0ivoTsvgrNwJ3Nr/view?usp=sharing
st.set_page_config(page_title= "2026 FIFA WORLDCUP")
MODEL_FILE = "fifa_model.pkl"
if not os.path.exists(MODEL_FILE):
    file_id = "1S8qcDReelgokkHFJQ0ivoTsvgrNwJ3Nr"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_FILE, quiet=False)

path = "fifa_dataset.pkl"
# https://drive.google.com/file/d/1__QzazmjLtJbxE_ROg2XMSQ6czWiqzHG/view?usp=sharing
if not os.path.exists(path):
    file_id = "1__QzazmjLtJbxE_ROg2XMSQ6czWiqzHG"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, path, quiet=False)

path2 = "over1.5_model.pkl"
# https://drive.google.com/file/d/1v5gBTW1puYfYSFqCBqgO8erm6T70_6hg/view?usp=sharing
if not os.path.exists(path2):
    file_id = "1v5gBTW1puYfYSFqCBqgO8erm6T70_6hg"
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
st.markdown("""
            <h3 style = "font-size:24px;">⚽FIFA WORLD CUP PREDICTION DASHBOARD</h3>
            """, unsafe_allow_html=True)
home = st.selectbox("Select Home Team:", teams)
away = st.selectbox("Select Away Team:", teams1)


with st.container(border = True):
    prediction = st.button('Predict Match')

home_set = fifa[fifa['home_team'] == home].iloc[0]
away_set = fifa[fifa['away_team'] == away].iloc[0]
    
match = {
    'home_avg_goal': home_set['home_avg_goal'], 'away_avg_goal': away_set['away_avg_goal'], 'home_avg_form':home_set['home_avg_form'], 'away_avg_form':away_set['away_avg_form'], 'home_5_conceded': home_set['home_5_conceded'], 'away_5_conceded': away_set['away_5_conceded']
        }
df = pd.DataFrame([match])
# st.write("Away Set Columns:", away_set.index.tolist())
# st.write("Home Set Columns:", home_set.index.tolist())
if prediction:
    st.markdown("""
                <h4 style ="font-size:18px;">Predicted Outcome</h4>
                """, unsafe_allow_html=True)
    
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
            st.markdown("""
                        <h4 style = "font-size:18px;">f"{home}"<h/4>
                        """, unsafe_allow_html=True)
            st.write(f"• In the Last 5 Matches, {home} scored {home_set['home_avg_goal']*5:.0f} goals (average {home_set['home_avg_goal']}).")
            st.write(f"• In the Last 5 Matches, {home} conceded an average {home_set['home_5_conceded']} goals per match.")
        
    with col2:
        with st.container(border=True):
            st.markdown("""
                        <h4 style= "font-size:18px;">f"{away}"</h4>
                        """, unsafe_allow_html=True)
            st.write(f"• In the Last 5 Matches, {away} scored {away_set['away_avg_goal']*5:.0f} goals (average {away_set['away_avg_goal']}).")
            st.write(f"• In the Last 5 Matches, {away} conceded an average {away_set['away_5_conceded']} goals per match.")
        
        
        #     # st.write("Confidence Level Table")

    outcome = {
        "Home Team":home,
        "Away Team":away,
        "Home Level": f"{confidence[2]*100:.0f}%",
        "Away Level": f"{confidence[0]*100:.0f}%",

        "Draw Level": f"{confidence[1]*100:.0f}%"
        }
    level = pd.DataFrame([outcome])
    # st.subheader("Confidence Level")
    # st.write(level)
    import plotly.graph_objects as go
    import streamlit as st

    # Prediction probabilities
    # home_prob = 64
    # draw_prob = 22
    # away_prob = 14
    
    colors = ["#9b1fb4", "#b4b41f", "#1f30b4"]
    labels = [home, "Draw", away]
    values = [confidence[2]*100, confidence[1]*100, confidence[1]*100]
    m = [f"{confidence[2]*100:.1f}", f"{confidence[1]*100:.1f}", f"{confidence[0]*100:.1f}"]
    # max_index = confidence.index(max(confidence))
    # max_index = np.argmax(confidence)
    # colors[max_index] = "#2ecc71"
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            # text=[f"{v:.1f}%" for v in values],
            text = m,
            textposition="outside",
            marker_color=colors
        )
    )

    fig.update_layout(
        title="",
        xaxis_title="Confidence Level (%)",
        xaxis=dict(range=[0, 100]),
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    with st.container(border =True):
        st.markdown("""
                    <h4 style="font-size:18px;"PREDICTION LEVEL</h4>
                    """, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
