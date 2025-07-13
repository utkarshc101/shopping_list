
import streamlit as st
import pickle
import pandas as pd
import gdown
import os
# Load model
model_path = "../models/shopping_model.pkl"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, model_path, quiet=False)

# Now load as usual
with open(model_path, "rb") as f:
    saved = pickle.load(f)

model = saved['model']
features = saved['features']

st.title("🛒 Shopping Cart Suggestion App")

items = st.multiselect("Select items you usually buy or need:", features)

input_df = pd.DataFrame([{item: (item in items) for item in features}])

predicted = model.predict(input_df)[0]
suggested_items = [features[i] for i, val in enumerate(predicted) if val and features[i] not in items]

st.subheader("Suggested items to add to your cart:")
if suggested_items:
    for item in suggested_items:
        st.write("-", item)
else:
    st.write("No new suggestions. Try selecting more common items!")
