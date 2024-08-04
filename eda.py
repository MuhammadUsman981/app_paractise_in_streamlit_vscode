import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns 
from ydata_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report

# Web app ka title
st.markdown('''
# **Exploratory Data Analysis Web Application**
This app is developed by codanics called **EDA App**
''')

# How to upload a file from PC
with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    

df= sns.load_dataset('titanic')
st.sidebar.markdown("Example CSV file")

if uploaded_file is not None:
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv

    df = load_csv()
    pr = profile_report(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('----')
    st.header('**Profiling report with Pandas**')
    st_profile_report(pr)
else:
    st.info('Awaiting CSV file upload. Please upload a file.')
    if st.button('Press to use the example'):
        # Example data set
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5),
                             columns=['age', 'banana', 'usman', 'pukhtunistan', 'nose'])
            return a

        df = load_data()
        pr = profile_report(df, explorative=True)
        st.header('**Input DF**')
        st.write(df)
        st.write('----')
        st.header('**Profiling report with Pandas**')
        st_profile_report(pr)
