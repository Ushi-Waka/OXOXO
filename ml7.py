import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

class StartupProfitAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.regressor = LinearRegression()
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных о стартапах"""
        self.dataset = pd.read_csv(self.data_path)
        
        # Выделение признаков и целевой переменной
        self.X = self.dataset.iloc[:, :-1].values  # Все столбцы кроме последнего
        self.y = self.dataset.iloc[:, -1].values   # Последний столбец - прибыль
        
        print("Первые 5 строк признаков:")
        print(self.X[:5])
        print("\nПервые 5 значений прибыли:")
        print(self.y[:5])
        
    def encode_categorical_features(self):
        """Кодирование категориальных признаков"""
        # Определяем столбец с категориями (3-й столбец, индекс 3)
        ct = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(), [3])  # Кодируем 3-й столбец
            ],
            remainder='passthrough'  # Остальные столбцы оставляем как есть
        )
        
        self.X = ct.fit_transform(self.X)
        print("\nМатрица после OneHot кодирования:")
        print(self.X[:4, :])
        
        # Удаляем одну фиктивную переменную (избегаем дамми-ловушку)
        self.X = self.X[:, 1:]
        print("\nМатрица после удаления одной фиктивной переменной:")
        print(self.X[:4, :])
        
    def split_data(self):
        """Разделение данных на обучающую и тестовую выборки"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=42
        )
        print(f"\nРазмер обучающей выборки: {len(self.X_train)}")
        print(f"Размер тестовой выборки: {len(self.X_test)}")
        
    def train_linear_model(self):
        """Обучение модели линейной регрессии"""
        self.regressor.fit(self.X_train, self.y_train)
        print("\nМодель линейной регрессии обучена")
        
        # Предсказание на тестовых данных
        y_pred = self.regressor.predict(self.X_test)
        print("\nПредсказанные значения прибыли:")
        print(y_pred)
        
    def optimize_model(self):
        """Оптимизация модели с помощью метода наименьших квадратов"""
        # Добавляем столбец единиц для константы
        X_with_const = np.append(
            arr=np.ones((len(self.X), 1)).astype(int), 
            values=self.X, 
            axis=1
        )
        
        # Первоначальная модель со всеми признаками
        print("\nРезультаты первоначальной модели:")
        initial_features = [0, 1, 2, 3, 4, 5]
        self._fit_and_print_ols(X_with_const, initial_features)
        
        # Оптимизированная модель (удален наименее значимый признак)
        print("\nРезультаты оптимизированной модели:")
        optimized_features = [0, 1, 3, 4, 5]
        self._fit_and_print_ols(X_with_const, optimized_features)
        
    def _fit_and_print_ols(self, X, feature_indices):
        """Вспомогательный метод для подбора модели OLS"""
        X_opt = X[:, feature_indices].astype(float)
        model = sm.OLS(self.y, X_opt).fit()
        print(model.summary())
        
    def run_analysis(self):
        """Запуск полного анализа"""
        self.load_and_prepare_data()
        self.encode_categorical_features()
        self.split_data()
        self.train_linear_model()
        self.optimize_model()

# Использование класса
if __name__ == "__main__":
    analyzer = StartupProfitAnalyzer('50_Startups.csv')
    analyzer.run_analysis()