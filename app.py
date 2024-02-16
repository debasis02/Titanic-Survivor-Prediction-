
import pandas as pd
import streamlit as st
import sklearn
import pickle
df=pd.read_csv('df_clean.csv')
model=pickle.load(open('Titanic_Survivour.pkl','rb'))
st.title('Titanic Survivor Prediction')
col1, col2= st.columns(2)

with col1:
    pclass = st.selectbox(
        'Entre the Pclass',
        (df['Pclass'].unique()))
    embarked = st.selectbox(
        'Entre the Embarkment',
        (df['Embarked'].unique()))
    sex= st.selectbox(
        'Entre the gender',
        (df['Sex'].unique()))
with col2:
    age= st.number_input('Entre the age of passenger',value=0)
    fare= st.number_input('Entre the fare of his ticket',value=0)
    family = st.number_input('Entre his family size',value=0)
if st.button('Show Predictions'):
    df=pd.DataFrame({'Pclass':[pclass],'Embarked':[embarked],'Sex':[sex],
                     'Age':[age],'Fare':[fare],'Family':family})
    price=model.predict(df)
    st.write('Prediction is',price)

