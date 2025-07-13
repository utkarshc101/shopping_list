
import streamlit as st
import pickle
import pandas as pd

# Load model
with open('../models/shopping_model.pkl', 'rb') as f:
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
