import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class SalaryAnalyzer:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.lin_model = LinearRegression()
        self.poly_model = LinearRegression()
        self.poly_transformer = None
        
    def load_data(self):
        """Загрузка и подготовка данных"""
        self.df = pd.read_csv('Position_Salaries.csv')
        
        # Выбираем уровень позиции как признак и зарплату как целевую переменную
        self.X = self.df.iloc[:, 1:2].values  # Берем как матрицу (n, 1)
        self.y = self.df.iloc[:, 2].values    # Вектор зарплат
        
        print("Первые 5 уровней позиций:")
        print(self.X[:5])
        print("\nПервые 5 значений зарплат:")
        print(self.y[:5])
    
    def train_models(self):
        """Обучение линейной и полиномиальной моделей"""
        # Линейная регрессия
        self.lin_model.fit(self.X, self.y)
        
        # Полиномиальная регрессия 10-й степени
        self.poly_transformer = PolynomialFeatures(degree=10)
        X_poly = self.poly_transformer.fit_transform(self.X)
        self.poly_model.fit(X_poly, self.y)
        
        print("\nЛинейная модель обучена")
        print(f"Коэффициент: {self.lin_model.coef_[0]:.2f}")
        print(f"Пересечение: {self.lin_model.intercept_:.2f}")
    
    def predict_salary(self, level):
        """Предсказание зарплаты для заданного уровня"""
        # Линейное предсказание
        lin_pred = self.lin_model.predict([[level]])[0]
        
        # Полиномиальное предсказание
        poly_pred = self.poly_model.predict(
            self.poly_transformer.transform([[level]])
        )[0]
        
        print(f"\nПрогноз для уровня {level}:")
        print(f"Линейная модель: ${lin_pred:,.2f}")
        print(f"Полиномиальная модель: ${poly_pred:,.2f}")
        
        return lin_pred, poly_pred
    
    def visualize_results(self):
        """Визуализация результатов"""
        plt.figure(figsize=(15, 5))
        
        # График линейной регрессии
        plt.subplot(1, 3, 1)
        plt.scatter(self.X, self.y, color='crimson', label='Реальные данные')
        plt.plot(self.X, self.lin_model.predict(self.X), 
                color='navy', label='Линейная модель')
        plt.title('Линейная регрессия', pad=15)
        plt.xlabel('Уровень позиции', labelpad=10)
        plt.ylabel('Зарплата ($)', labelpad=10)
        plt.legend()
        plt.grid(alpha=0.2)
        
        # График полиномиальной регрессии (дискретный)
        plt.subplot(1, 3, 2)
        plt.scatter(self.X, self.y, color='crimson', label='Реальные данные')
        plt.plot(self.X, self.poly_model.predict(
            self.poly_transformer.transform(self.X)), 
            color='darkgreen', label='Полиномиальная модель')
        plt.title('Полиномиальная регрессия (10 степень)', pad=15)
        plt.xlabel('Уровень позиции', labelpad=10)
        plt.ylabel('Зарплата ($)', labelpad=10)
        plt.legend()
        plt.grid(alpha=0.2)
        
        # График полиномиальной регрессии (сглаженный)
        X_smooth = np.arange(min(self.X), max(self.X), 0.1).reshape(-1, 1)
        plt.subplot(1, 3, 3)
        plt.scatter(self.X, self.y, color='crimson', label='Реальные данные')
        plt.plot(X_smooth, self.poly_model.predict(
            self.poly_transformer.transform(X_smooth)), 
            color='darkgreen', label='Сглаженная модель')
        plt.title('Сглаженная полиномиальная регрессия', pad=15)
        plt.xlabel('Уровень позиции', labelpad=10)
        plt.ylabel('Зарплата ($)', labelpad=10)
        plt.legend()
        plt.grid(alpha=0.2)
        
        plt.tight_layout()
        plt.show()

# Основной блок выполнения
if __name__ == "__main__":
    analyzer = SalaryAnalyzer()
    
    # Шаг 1: Загрузка данных
    analyzer.load_data()
    
    # Шаг 2: Обучение моделей
    analyzer.train_models()
    
    # Шаг 3: Прогнозирование для уровня 6.5
    analyzer.predict_salary(6.5)
    
    # Шаг 4: Визуализация результатов
    analyzer.visualize_results()