# import libraries
import pandas as pd
import streamlit as st 
import plotly.express as px 

# import dataset

st.title("plotly or streamlit ko mila kai app banana")
df = px.data.gapminder()
st.write(df)
# st.write(df.head())
st.write(df.columns)

# Summury stat
st.write(df.describe())

# Data managment 
year_option = df['year'].unique().tolist() 

year = st.selectbox("which year should we plot?",year_option,0)
df = df[df['year']== year]


# using plotly for plotting

fig = px.scatter(x=df['gdpPercap'],y=df['lifeExp'],size=df['pop'],color=df['country'],hover_name=df['country'],
                  log_x=True,size_max=55,range_x=[100,10000],range_y=[20,90],
                 animation_frame=df['year'],animation_group=df['country'])
fig.update_layout(width=1000)
st.write(fig)

