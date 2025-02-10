from src.preprocessing import DataProcessing
import pickle
from data_prep import *

def predict(X):
    # Load the model and scaler from the saved file
    with open("models/model.pickle", 'rb') as f:
        print("Model Imported")
        model, label_encoders, scaler, ordinal_encoders = pickle.load(f)

        X = pd.read_csv('./data/raw/test.csv')

        X = data_prep(X, '', ordinal_encoders, label_encoders, scaler=None)

        pred = model.predict(X)  # Predict time of delivery
        return pred