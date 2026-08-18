from sklearn.ensemble import GradientBoostingRegressor


class TaxiFareModel:
    def __init__(self):
        # Фиксируем random_state для воспроизводимости экспериментов в MLOps
        self.model = GradientBoostingRegressor(random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        # Инкапсулируем метод оценки качества
        return self.model.score(X, y)