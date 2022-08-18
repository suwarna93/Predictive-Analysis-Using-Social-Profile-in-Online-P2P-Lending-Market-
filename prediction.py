import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

pickle_in = open('NB_model.pkl', 'rb') 
NB_model = pickle.load(pickle_in)

pickle_in = open('LG_model.pkl', 'rb') 
LG_model = pickle.load(pickle_in)

pickle_in = open('DT_model.pkl', 'rb') 
DT_model = pickle.load(pickle_in)





def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)