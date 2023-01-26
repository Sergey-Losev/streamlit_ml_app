import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model


def main():
    global df
    with st.sidebar:
        st.image("/home/serge/PycharmProjects/streamlit_ml_app/inno2.png")
        st.title("AutoStreamML")
        choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
        st.info("This application allows you to build an automated ML pipeline using Streamlit, "
                "Pandas Profiling and PyCaret. And it damnright magic!")
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)

    if choice == "Upload":
        st.title("Upload Your Data for Modelling")
        file = st.file_uploader("Upload your dataset Here")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Automated Exploratory Data Analysis")
        profile_report = df.profile_report()
        st_profile_report(profile_report)

    if choice == "ML":
        st.title("Machine Learning")
        target = st.selectbox("Select your target", df.columns)
        if st.button("Train Model"):
            setup(df, target=target, silent=True)
            setup_df = pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, "best_model")

    if choice == "Download":
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download the Model", f, "trained_model.pkl")


if __name__ == "__main__":
    main()
