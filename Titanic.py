"""
Creating a Streamlit App that uses an Neural Network

Here our project focuses on "Whether a person will survive the Titanic incident or not

"""

# Importing the dependencies
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import Dense 

import streamlit as st
import streamlit.components.v1 as comp


## Loading the dataset
def load_data():
    titanic = pd.read_csv(r'titanic.csv')
    return titanic

def preprocess_data(titanic):
    titanic = titanic.drop(columns='Name')
    titanic.loc[titanic['Sex']=='male','Sex'] = 1
    titanic.loc[titanic['Sex']=='female','Sex'] = 0

    X = titanic.drop('Survived', axis = 1)
    y = titanic['Survived'].values.astype(np.float32)
    X = X.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    return X_train, X_test, y_train, y_test

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6,)),
        keras.layers.Dense(6, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def save_model(model):
    model.save("Titanic_model.h5")

def app():
    st.title('Will you survive the Titanic ?')

    #Load the dataset
    titanic = load_data()

    #Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(titanic)

    #Creating and training the model
    dl_model = build_model()
    dl_model.fit(X_train,y_train, epochs=50, verbose=0)

    #Save the trained model
    save_model(dl_model)

    #Evaluating the model
    _, accuracy = dl_model.evaluate(X_test,y_test)
    st.write(f"Accuracy of the model : {accuracy*100:.2f}%")

    #Asking for user input so that we can make a prediction
    st.sidebar.header("Make a Prediction")

    Pclass = st.slider('Choose Passenger Class',1,3)
    sex = st.selectbox('Choose Sex',('Male','Female'))

    if sex =='Male':
        sex = 1
    else:
        sex = 0
    
    age = st.slider("Choose Age of the passenger",0,100)
    sibsp = st.slider("No of Siblings/Spouses",0,3)
    parch = st.slider("No of Parents/Children",0,3)
    fair = st.slider("Select the Fair",0,200)

    data = [[Pclass, sex, age, sibsp, parch, fair]]
    data = tf.constant(data)
    return data

def predict(data,model):
    prediction =  model.predict(data)
    if prediction[0] == 1:
        st.success("Passenger Survived :thumbsup:")
    else:
        st.error("Passenger Did Not Survive :thumbsdown:")

data = app()
model = load_model('Titanic_model.h5')
trigger = st.button('Predict', on_click=predict(data,model))
    












