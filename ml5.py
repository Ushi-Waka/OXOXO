import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.X = None
        self.y = None
        self.X_processed = None
        
    def load_and_split_data(self):
        """Загрузка данных и разделение на признаки и целевую переменную"""
        self.dataset = pd.read_csv(self.data_path)
        features = self.dataset.iloc[:, :-1].values
        target = self.dataset.iloc[:, 3].values
        return features, target
    
    def handle_missing_values(self, data):
        """Обработка пропущенных значений"""
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        numeric_data = data[:, 1:3].copy()
        imputer.fit(numeric_data)
        numeric_data_imputed = imputer.transform(numeric_data)
        
        data_imputed = data.copy()
        data_imputed[:, 1:3] = numeric_data_imputed
        return data_imputed
    
    def encode_categorical_target(self, target):
        """Кодирование категориальной целевой переменной"""
        print("\nЦелевая переменная до кодирования:")
        print(target)
        
        encoder = LabelEncoder()
        encoded_target = encoder.fit_transform(target)
        
        print("\nЦелевая переменная после кодирования:")
        print(encoded_target)
        return encoded_target
    
    def create_preprocessing_pipeline(self):
        """Создание комплексного пайплайна для обработки данных"""
        transformers = [
            ('categorical', OneHotEncoder(), [0]),  # OneHot для категориальных признаков
            ('numeric', SimpleImputer(strategy='mean'), [1, 2])  # Импутация для числовых
        ]
        return ColumnTransformer(transformers)
    
    def run_pipeline(self):
        """Запуск всего процесса предобработки"""
        # Шаг 1: Загрузка данных
        self.X, self.y = self.load_and_split_data()
        
        print("Исходная матрица признаков:")
        print(self.X[:5])
        print("\nИсходная целевая переменная:")
        print(self.y[:5])
        
        # Шаг 2: Обработка пропущенных значений (альтернативный способ)
        X_imputed = self.handle_missing_values(self.X)
        print("\nМатрица после обработки пропусков:")
        print(X_imputed[:5])
        
        # Шаг 3: Кодирование целевой переменной
        self.y = self.encode_categorical_target(self.y)
        
        # Шаг 4: Комплексная обработка признаков
        preprocessor = self.create_preprocessing_pipeline()
        self.X_processed = preprocessor.fit_transform(self.X)
        
        print("\nРазмерность обработанной матрицы признаков:", self.X_processed.shape)
        print("\nОбработанная матрица:")
        print(self.X_processed[:5])
        
        # Шаг 5: Преобразование в DataFrame
        processed_df = pd.DataFrame(
            self.X_processed,
            columns=['Category_1', 'Category_2', 'Category_3', 'Age', 'Salary']
        )
        
        print("\nИтоговый DataFrame:")
        print(processed_df.head())
        
        return processed_df, self.y

# Использование класса
if __name__ == "__main__":
    processor = DataPreprocessor('Data5.csv')
    X_final, y_final = processor.run_pipeline()