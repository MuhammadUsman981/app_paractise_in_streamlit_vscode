import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 

# app ki heading 
st.write('''
    # Explore different ML models dataset
    Dekhtay hain kon sa model best hai
    ''')

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)


def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x, y = data.data, data.target
    return x, y

x,y  = get_dataset(dataset_name)

st.write('Shape of dataset:', x.shape)
st.write('number of classes: ', len(np.unique(y)))

params = dict()  # Create an empty dictionary

def add_parameter_ui(classifier_name):

    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C  # Degree of correct classification

    elif classifier_name == 'KNN':
        n_neighbors = st.sidebar.slider('n_neighbors', 1, 15)
        print(f"n_neighbors: {n_neighbors}")
        params['n_neighbors'] = n_neighbors  # Number of nearest neighbors

    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth  # Depth of trees in Random Forest

        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators  # Number of trees in Random Forest

    return params

# Example usage:
selected_classifier = st.sidebar.selectbox("Select Classifier", ("SVM", "KNN", "Random Forest"))
classifier_params = add_parameter_ui(selected_classifier)

def get_classifier(classifier_name, params):
    clf = None

    if classifier_name == 'SVM':
        clf = SVC(C=params['C'], kernel='rbf', gamma=params['gamma'])

    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'])

    else:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=1234
        )

    return clf


clf = get_classifier(classifier_name,  params)


x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.2,random_state=1234)


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Check the model's accuracy score and print it
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')

