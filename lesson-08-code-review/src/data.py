import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_data(data):
    # Удаляем строки с пропущенными значениями, чтобы модель не падала с ошибкой NaN
    data = data.dropna()
    
    features = data.drop('fare_amount', axis=1)
    # Оставляем только числовые колонки
    features = features.select_dtypes(include=['number'])
    target = data['fare_amount']
    return train_test_split(features, target, test_size=0.2, random_state=42)