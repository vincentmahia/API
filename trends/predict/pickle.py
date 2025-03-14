import pickle

#Load trained model
with open("predict/models/predict_model.pkl", "rb") as f:
    model = pickle.load(f)