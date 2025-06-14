import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

class IrisAnalyzer:
    def __init__(self, data_path):
        self.feature_names = [
            'sepal_length', 
            'sepal_width', 
            'petal_length', 
            'petal_width', 
            'species'
        ]
        self.dataset = self.load_data(data_path)
        self.plot_settings = {
            'style': 'seaborn-v0_8-talk',
            'palette': 'husl',
            'markers': ['o', 's', 'D']
        }
        
    def load_data(self, path):
        """Загрузка данных о цветах ириса"""
        iris_data = pd.read_csv(path, names=self.feature_names)
        print(f"Данные успешно загружены. Образцов: {iris_data.shape[0]}")
        return iris_data
        
    def explore_data(self):
        """Визуализация распределения признаков"""
        plt.figure(figsize=(12, 8))
        sns.pairplot(
            self.dataset, 
            hue='species',
            markers=self.plot_settings['markers'],
            palette=self.plot_settings['palette'],
            plot_kws={'alpha': 0.8, 'edgecolor': 'black'}
        )
        plt.suptitle('Парные распределения признаков ирисов', y=1.02)
        plt.show()
        
    def train_model(self):
        """Обучение и оценка модели KNN"""
        features = self.dataset.drop('species', axis=1)
        target = self.dataset['species']
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, 
            test_size=0.3, 
            random_state=42,
            stratify=target
        )
        
        # Обучение модели
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        # Оценка точности
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'\nТочность модели: {accuracy:.2f}')
        
        return features, target
        
    def optimize_parameters(self, X, y):
        """Подбор оптимального числа соседей"""
        k_range = range(1, 50)
        cv_results = []
        
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
            cv_results.append(scores.mean())
            
        # Нахождение оптимального K
        optimal_k = k_range[np.argmax(cv_results)]
        print(f'Оптимальное число соседей: {optimal_k}')
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, [1 - x for x in cv_results], 'b-', linewidth=2)
        plt.xlabel('Число соседей (K)', fontsize=12)
        plt.ylabel('Ошибка классификации', fontsize=12)
        plt.title('Зависимость ошибки от числа соседей', pad=20)
        plt.grid(alpha=0.3)
        plt.show()
        
        return optimal_k
        
    def visualize_decision_boundaries(self, X, y, optimal_k):
        """Визуализация решающих границ"""
        feature_pairs = [(i, j) for i in range(4) for j in range(4)]
        species_types = y.unique()
        colors = ['red', 'green', 'blue']
        
        plt.figure(figsize=(18, 18))
        
        for idx, (f1, f2) in enumerate(feature_pairs):
            ax = plt.subplot(4, 4, idx + 1)
            
            if f1 != f2:
                # Подготовка сетки
                x_min, x_max = X.iloc[:, f1].min() - 1, X.iloc[:, f1].max() + 1
                y_min, y_max = X.iloc[:, f2].min() - 1, X.iloc[:, f2].max() + 1
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, 0.02),
                    np.arange(y_min, y_max, 0.02)
                
                # Обучение на двух признаках
                model = KNeighborsClassifier(n_neighbors=optimal_k)
                model.fit(X.iloc[:, [f1, f2]], y)
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = np.array([np.where(species_types == z)[0][0] for z in Z])
                Z = Z.reshape(xx.shape)
                
                # Визуализация границ
                plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            
            # Визуализация точек данных
            for i, species in enumerate(species_types):
                if f1 == f2:
                    # Гистограмма для диагональных графиков
                    plt.hist(
                        X[y == species].iloc[:, f1], 
                        color=colors[i],
                        alpha=0.5,
                        bins=15
                    )
                else:
                    # Точечный график
                    plt.scatter(
                        X[y == species].iloc[:, f1],
                        X[y == species].iloc[:, f2],
                        c=colors[i],
                        label=species,
                        edgecolor='black',
                        s=50
                    )
            
            # Настройка подписей
            if idx % 4 == 0:
                plt.ylabel(self.feature_names[f2])
            if idx >= 12:
                plt.xlabel(self.feature_names[f1])
                
        plt.tight_layout()
        plt.suptitle('Решающие границы классификатора KNN', y=1.02, fontsize=16)
        plt.show()

# Основной блок выполнения
if __name__ == "__main__":
    analyzer = IrisAnalyzer('iris.data')
    analyzer.explore_data()
    
    features, target = analyzer.train_model()
    best_k = analyzer.optimize_parameters(features, target)
    analyzer.visualize_decision_boundaries(features, target, best_k)
