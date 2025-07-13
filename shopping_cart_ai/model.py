
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from mlxtend.preprocessing import TransactionEncoder
import pickle
import os

def train_model(data_path, model_path):
    df = pd.read_csv(data_path, header=None)
    transactions = df.values.tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    data = pd.DataFrame(te_ary, columns=te.columns_)

    X = data
    y = data

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': list(data.columns)}, f)

    print("✅ Model trained and saved at:", model_path)
