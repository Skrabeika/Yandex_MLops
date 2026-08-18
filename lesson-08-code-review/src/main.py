import os
import pandas as pd
from src.data import load_data, split_data
from src.features import add_time_features
from src.model import TaxiFareModel

# 1. Путь к данным
DATA_PATH = "data/uber.csv"

# 2. Загрузка данных
raw_data = load_data(DATA_PATH)

# 3. Сначала применяем feature engineering (превращаем дату в числа и удаляем строку с датой)
processed_data = add_time_features(raw_data)

# 4. Делим уже обработанные данные на train/test (избегаем Data Leakage)
X_train, X_test, y_train, y_test = split_data(processed_data)

# 5. Обучение модели
model = TaxiFareModel()
model.fit(X_train, y_train)

# 6. Оценка качества
score = model.score(X_test, y_test)
print(f"R²: {score:.2f}")