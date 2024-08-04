import streamlit as st 
import seaborn as sns 

st.header("this code is brought to by Muhammad Usman")

st.text("han g hamain maza Aaa raha hai sekhnai mai")


st.header("i don't know what to write")

df= sns.load_dataset('iris')
 

st.write(df[['species','sepal_length','petal_length']].head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])