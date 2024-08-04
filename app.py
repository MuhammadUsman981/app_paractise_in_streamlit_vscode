import streamlit as st 
import seaborn as sns 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# Make containers
header = st.container()
data_sets = st.container()
feature = st.container()
model_training = st.container()

with header:
    st.title("kashti ki app")
    st.text("in this project we will work on kashti data set")
    
with data_sets:
        st.title("kashti doob gai")
        st.text("aww! kashti doob gaii")
        # morting datasets 
        df = sns.load_dataset("titanic")
        df=df.dropna()
        st.write(df.head(10))
        st.subheader("aray OOooo kitnai admi thai")
        st.bar_chart(df['sex'].value_counts())
        # other charts/giving heading
        st.subheader("class kai hisaab sa faraq")
        st.bar_chart(df['class'].value_counts())


with feature:
    st.title("these are our app features")
    st.text("titanic walai boht badnaseeb thai")
    st.markdown('1. **Feature 1**This will tell us something that we dont know')
with model_training:
    st.title("kashti walo ka kia bana")
    st.text("mujay nahi pata is mai kia likhna hai")
    #making coloumns
    input,display = st.columns(2)
    # pehlay colum mai ap kai selection features ho
    max_depth = input.slider("how many people do you know",min_value=10,max_value=100,value=20,step=5) 

#n_estimator
n_estimators = input.selectbox("how many people should e there in RF?",options=[50,100,200,300,'NO limit'])

# adding list of features 
input.write(df.columns)

# Input feature from users

input_features = input.text_input("which feature should we use")


# Machine learning model
model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

# defining x & y

x = df[[input_features]]
y = df[['fare']]

# fit our model

model.fit(x,y)

pred = model.predict(y)


# display metrics 

display.subheader("mean absolute error of the model is:")
display.write(mean_absolute_error(y,pred))
display.subheader("mean squared error of the model is:")
display.write(mean_squared_error(y,pred))
display.subheader("R squared score of the model is:")
display.write(r2_score(y,pred))