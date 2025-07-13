
import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['features']

def predict_cart(model, features, input_items):
    input_df = pd.DataFrame([{item: (item in input_items) for item in features}])
    prediction = model.predict(input_df)
    predicted_items = [features[i] for i, val in enumerate(prediction[0]) if val]
    return predicted_items
