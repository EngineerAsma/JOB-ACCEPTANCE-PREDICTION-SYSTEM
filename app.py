import streamlit as st
import pandas as  pd
import numpy as np
import joblib

model = joblib.load("random forest classifier.pkl")

st.title("JOB ACCEPTANCE PREDICTION SYSTEM\n")
st.text("This is a simple job acceptance  prediction system. It uses Machine Learning algorithm Random Forest to predict whther the  person accepts job offer and will get placed. So its simply a binary classification problem where the target is placed or not placed. The placement probability is predicted with the use of metrics such as interview_score, academic_score, job_readiness_score, skills_match_percentage, etc.")

age_years = st.number_input("age", min_value=18, max_value=45)
ssc_percentage = st.slider("ssc_percentage")
hsc_percentage = st.slider("hsc_percentage")
degree_percentage = st.slider("degree_percentage")
technical_score = st.slider("technical_score")
aptitude_score = st.slider('aptitude_score')
communication_score = st.slider('communication_score')
skills_match_percentage = st.slider('skills_match_percentage')
certifications_count = st.number_input('certifications_count', min_value=0, max_value = 10)     
internship_experience = st.number_input('internship_experience', min_value=0, max_value=10)
years_of_experience = st.number_input('years_of_experience', min_value=0, max_value=10)
career_switch_willingness = st.selectbox('career_switch_willingness (0-no, 1-yes)', [0,1])
relevant_experience = st.number_input('relevant_experience', min_value=0, max_value=10)
previous_ctc_lpa = st.number_input('previous_ctc_lpa')
expected_ctc_lpa = st.number_input('expected_ctc_lpa')
company_tier = st.selectbox('company_tier1.0-tier1, 0.7-tier2, 0.4-tier3)', [1.0, 0.7, 0.4])
job_role_match = st.selectbox('job_role_match(0-no, 1-yes)', [0,1])
competition_level = st.selectbox('competition_level(1-low, 0.7-medium, 0.4-high', [1.0,0.7,0.4])
bond_requirement = st.selectbox('bond_requirement (0-no, 1-yes)', [0,1])
notice_period_days = st.slider('notice_period_days')
layoff_history = st.selectbox('layoff_history (0-no, 1-yes)', [0,1])
employment_gap_months = st.slider('employment_gap_months')
relocation_willingness = st.selectbox('relocation_willingness (0-no, 1-yes)', [0,1])           


# derived metrics

interview_skills_score = (technical_score + aptitude_score + communication_score) /3/100
academic_score = (ssc_percentage + hsc_percentage + degree_percentage)/3/100
experience_score = np.minimum(years_of_experience / 5, 1)
certification_score = np.minimum(certifications_count / 5, 1)
skills_match_score = skills_match_percentage/100
notice_period_score = np.where(notice_period_days <= 15, 1, np.where(notice_period_days <= 30, 0.7, 0.4))
employment_gap_penalty = np.maximum(1 - (employment_gap_months / 24), 0)
job_readiness_score = (skills_match_percentage / 100 * 0.7 + job_role_match * 0.3)
total_experience_score = (internship_experience * 0.2 + experience_score * 0.4 +
    relevant_experience * 0.3 +
    certification_score * 0.1
)
flexibility_score = (relocation_willingness * 0.3 + notice_period_score * 0.4 + employment_gap_penalty * 0.3)

placement_probability_score = (
    academic_score * 0.20 +
    interview_skills_score * 0.30 +
    job_readiness_score * 0.20 +
    total_experience_score * 0.15 +
    flexibility_score * 0.15
)*100 

input_info = np.array([[age_years, ssc_percentage, hsc_percentage, degree_percentage, technical_score, 
                        aptitude_score, 
                        communication_score, skills_match_percentage, certifications_count, 
                        internship_experience, years_of_experience, career_switch_willingness,
                        relevant_experience, previous_ctc_lpa, expected_ctc_lpa, company_tier, 
                        job_role_match, competition_level,bond_requirement, notice_period_days,
                        layoff_history, employment_gap_months, relocation_willingness, academic_score, 
                        interview_skills_score, experience_score, certification_score, skills_match_score, notice_period_score, 
                        employment_gap_penalty, job_readiness_score, total_experience_score,
                        flexibility_score, placement_probability_score]])

col1, col2, col3 = st.columns(3, gap="large")

col1.metric("academic score:", academic_score)
col2.metric("interview skills score:", interview_skills_score)
col3.metric("experience score (years of experience):", experience_score)

col4, col5, col6 = st.columns(3, gap="large")

col4.metric("certification score:", certification_score)
col5.metric("skills match score:", skills_match_score)
col6.metric(label="notice period score:", value= notice_period_score.item())

col7, col8, col9 = st.columns(3, gap="large")

col7.metric("employment_gap_penalty", employment_gap_penalty)
col8.metric("job_readiness_score:", job_readiness_score)
col9.metric("total_experience_score:", total_experience_score)

col10, col11 = st.columns(2, gap="large")

col10.metric("flexibility_score:", flexibility_score)
col11.metric("placement_probability_score:", placement_probability_score)


if(st.button("predict placement")):
    prediction = model.predict(input_info)

    if prediction[0] ==  1:
        st.write("placed")
        st.write("placement probability :", model.predict_proba(input_info)[:, 1])

    else:
        st.write("not placed")
        st.write("placement probability :", model.predict_proba(input_info)[:, 1])


