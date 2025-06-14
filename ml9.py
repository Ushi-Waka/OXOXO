import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Настройки отображения графиков
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Загрузка данных о покупателях
customer_data = pd.read_csv('Mall_Customers.csv')
spending_data = customer_data.iloc[:, [3, 4]].values  # Доход и рейтинг трат

# Анализ оптимального числа кластеров методом локтя
def find_optimal_clusters(data, max_clusters=10):
    cluster_errors = []
    for n in range(1, max_clusters+1):
        model = KMeans(n_clusters=n, init='k-means++', random_state=42)
        model.fit(data)
        cluster_errors.append(model.inertia_)
    
    # Визуализация метода локтя
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters+1), cluster_errors, marker='o', 
             color='royalblue', linewidth=2)
    plt.axvline(x=5, color='r', linestyle='--', alpha=0.5)
    plt.title('Определение оптимального числа кластеров', pad=20)
    plt.xlabel('Количество групп', labelpad=10)
    plt.ylabel('Сумма квадратов расстояний', labelpad=10)
    plt.grid(alpha=0.3)
    plt.show()
    
    return cluster_errors

# Выполняем анализ
wcss = find_optimal_clusters(spending_data)

# Кластеризация данных
def perform_clustering(data, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = model.fit_predict(data)
    return model, clusters

# Создаем модель с 5 кластерами (по результатам метода локтя)
kmeans_model, cluster_labels = perform_clustering(spending_data)

# Визуализация результатов
def plot_clusters(data, labels, centers):
    # Цвета для каждого кластера
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    plt.figure(figsize=(12, 8))
    
    # Отображаем точки каждого кластера
    for i in range(5):
        cluster_data = data[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                    s=100, c=palette[i], 
                    label=f'Группа {i+1}', 
                    edgecolor='white', alpha=0.8)
    
    # Отображаем центроиды
    plt.scatter(centers[:, 0], centers[:, 1], 
                s=250, c='gold', 
                marker='*', label='Центры', 
                edgecolor='black', linewidth=1)
    
    # Настройки графика
    plt.title('Сегментация покупателей по доходу и расходам', pad=20)
    plt.xlabel('Годовой доход (тыс. $)', labelpad=10)
    plt.ylabel('Уровень расходов (1-100)', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# Отображаем результаты
plot_clusters(spending_data, cluster_labels, kmeans_model.cluster_centers_)