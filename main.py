from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup as regsetup, compare_models as regcompare, pull as regpull, save_model as regsave
from pycaret.classification import setup as cfsetup,compare_models as cfcompare, pull as cfpull, save_model as cfsave
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import os   

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.title("Machine learning model recommendation")
    choice2 = st.radio("Choose method",["Regression","Classification","Clustering"])

if choice2 == "Regression":
    with st.sidebar: 
        choice = st.radio("Navigation", ["Upload","Visualize and Analyse","Modelling"])
        st.info("Visualize and Analyse your data and find a suitable model for your dataset.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Visualize and Analyse": 
        st.title("Visualization and Analysis:")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling": 
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'): 
            regsetup(df, target=chosen_target)
            setup_df = regpull()
            st.dataframe(setup_df)
            best_model = regcompare()
            compare_df = regpull()
            st.dataframe(compare_df)
            regsave(best_model, 'best_model')



if choice2 == "Classification":
    with st.sidebar: 
        choice = st.radio("Navigation", ["Upload","Visualize and Analyse","Modelling"])
        st.info("Visualize and Analyse your data and find a suitable model for your dataset.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Visualize and Analyse": 
        st.title("Visualization and Analysis:")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling": 
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'): 
            cfsetup(df, target=chosen_target)
            setup_df = cfpull()
            st.dataframe(setup_df)
            best_model = cfcompare()
            compare_df = cfpull()
            st.dataframe(compare_df)
            cfsave(best_model, 'best_model')


if choice2 == "Clustering":
    with st.sidebar: 
        choice = st.radio("Navigation", ["Upload","Visualize and Analyse"])
        st.info("Visualize and Analyse your data and find a suitable model for your dataset.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Visualize and Analyse": 
        st.title("Visualization and Analysis:")
        profile_df = df.profile_report()
        st_profile_report(profile_df)
