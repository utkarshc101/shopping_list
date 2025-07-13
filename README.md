
# 🛒 Shopping Cart AI

This project uses a real-world shopping dataset to train an AI that suggests items you may want to buy based on your current shopping behavior.

## Structure
- `shopping_cart_ai/`: Core logic for training, preprocessing, prediction.
- `notebooks/`: Jupyter Notebook to train the model.
- `streamlit_app/`: Streamlit app for user-friendly predictions.
- `data/`: Raw data (like groceries.csv).
- `models/`: Trained model in pickle format.

## Getting Started
1. Run `train_model.ipynb` to train the model.
2. Launch the app:
```bash
cd streamlit_app
streamlit run app.py
```

## Dataset Source
- [Groceries Market Basket Dataset](https://www.kaggle.com/datasets/irfanasrullah/groceries)
