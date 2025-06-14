import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class SalaryPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.X = None
        self.y = None
        self.regressor = LinearRegression()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        self.dataset = pd.read_csv(self.data_path)
        self.X = self.dataset.iloc[:, :-1].values  # Признак: опыт работы
        self.y = self.dataset.iloc[:, 1].values    # Целевая переменная: зарплата
        
        print("Первые 5 записей признаков:")
        print(self.X[:5])
        print("\nПервые 5 значений зарплат:")
        print(self.y[:5])
        
    def split_data(self):
        """Разделение данных на обучающую и тестовую выборки"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.25, 
            random_state=42
        )
        print(f"\nРазмер обучающей выборки: {len(self.X_train)}")
        print(f"Размер тестовой выборки: {len(self.X_test)}")
        
    def train_model(self):
        """Обучение модели линейной регрессии"""
        self.regressor.fit(self.X_train, self.y_train)
        print("\nМодель успешно обучена")
        print(f"Коэффициент: {self.regressor.coef_[0]:.2f}")
        print(f"Пересечение: {self.regressor.intercept_:.2f}")
        
    def predict_and_evaluate(self):
        """Предсказание и оценка результатов"""
        y_pred = self.regressor.predict(self.X_test)
        print("\nПредсказанные значения для тестовой выборки:")
        print(y_pred)
        return y_pred
        
    def visualize_results(self):
        """Визуализация результатов"""
        # Создаем фигуру с двумя графиками
        plt.figure(figsize=(14, 6))
        
        # График для обучающей выборки
        plt.subplot(1, 2, 1)
        plt.scatter(self.X_train, self.y_train, color='coral', label='Фактические данные')
        plt.plot(self.X_train, self.regressor.predict(self.X_train), 
                color='navy', linewidth=2, label='Прогноз модели')
        plt.title('Зарплата vs Опыт (Обучающая выборка)', pad=15)
        plt.xlabel('Годы опыта', labelpad=10)
        plt.ylabel('Зарплата', labelpad=10)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # График для тестовой выборки
        plt.subplot(1, 2, 2)
        plt.scatter(self.X_test, self.y_test, color='coral', label='Фактические данные')
        plt.plot(self.X_train, self.regressor.predict(self.X_train), 
                color='navy', linewidth=2, label='Прогноз модели')
        plt.title('Зарплата vs Опыт (Тестовая выборка)', pad=15)
        plt.xlabel('Годы опыта', labelpad=10)
        plt.ylabel('Зарплата', labelpad=10)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def run_analysis(self):
        """Запуск полного анализа"""
        self.load_and_prepare_data()
        self.split_data()
        self.train_model()
        self.predict_and_evaluate()
        self.visualize_results()

# Использование класса
if __name__ == "__main__":
    analyzer = SalaryPredictor('Salary_Data.csv')
    analyzer.run_analysis()
