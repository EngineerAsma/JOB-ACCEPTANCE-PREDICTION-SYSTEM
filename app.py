# STREAMLIT DASHBOOARD

import streamlit as st
import  pandas as pd

# KEY KPIS 
df_dash = pd.read_csv("C:\\Users\\User\\.vscode\\PLACEMENT DATASET\\cleaned_placement_data.csv")

total_candidates = df_dash.shape[0]

placement_rate = (df_dash['status'].sum() / total_candidates) * 100

job_acceptance_rate = placement_rate  # status = 1 means accepted

avg_interview_score = (df_dash['interview_skills_score'].mean()) * 100

avg_skills_match = df_dash['skills_match_percentage'].mean()

offer_dropout_rate = (
    ((df_dash['placement_probability_score'] >= 0.7) & (df_dash['status'] == 0)).sum()
    / total_candidates
) * 100

high_risk_percentage = (
    (df_dash['placement_probability_score'] < 50).sum()
    / total_candidates
) * 100

# --------------------------------------------------
# STREAMLIT DASHBOARD
# --------------------------------------------------
st.set_page_config(page_title="Job Acceptance Prediction Dashboard", layout="wide")

st.title("📊 Job Acceptance Prediction System – KPI Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Candidates", total_candidates)
col1.metric("Placement Rate (%)", round(placement_rate, 2))
col1.metric("Job Acceptance Rate (%)", round(job_acceptance_rate, 2))

col2.metric("Average Interview Score", round(avg_interview_score, 2))
col2.metric("Average Skills Match (%)", round(avg_skills_match, 2))

col3.metric("Offer Dropout Rate (%)", round(offer_dropout_rate, 2))
col3.metric("High-Risk Candidate (%)", round(high_risk_percentage, 2))

st.markdown("---")

st.subheader("📌 KPI Interpretation")
st.write("""
- **Placement Rate / Job Acceptance Rate:** Overall success of recruitment process  
- **Offer Dropout Rate:** Candidates having higher placement probability score but still got rejected / not accepted offers
- **High-Risk Candidates:** Candidates having low placement probability score  
- **Interview & Skills KPIs:** Quality of candidate-job fit
""")
