import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle

pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)




def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)