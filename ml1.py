import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pip._internal.utils.misc import tabulate

# Загрузка и подготовка данных
def load_iris_data(filepath):
    iris_dtype = np.dtype([
        ('sepal_length', 'f8'),
        ('sepal_width', 'f8'),
        ('petal_length', 'f8'),
        ('petal_width', 'f8'),
        ('species', 'U30')
    ])
    return np.genfromtxt(filepath, delimiter=",", dtype=iris_dtype)

iris_data = load_iris_data("iris.data")

# Проверка данных
print(f"Общее количество образцов: {iris_data.shape[0]}")
print(f"Тип данных: {type(iris_data)}")
print(f"Тип метки вида: {type(iris_data[0]['species'])}")
print("\nПервые 5 записей:")
for i in range(5):
    print(iris_data[i])

# Извлечение признаков
def extract_features(data):
    features = {
        'sepal_length': data['sepal_length'],
        'sepal_width': data['sepal_width'],
        'petal_length': data['petal_length'],
        'petal_width': data['petal_width']
    }
    return features

all_features = extract_features(iris_data)

# Разделение по видам
species = {
    'setosa': slice(0, 50),
    'versicolor': slice(50, 100),
    'virginica': slice(100, 150)
}

# Настройка графиков
mpl.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Функция для создания графиков
def create_scatter_plot(fig_num, x_data, y_data, xlabel, ylabel, title):
    plt.figure(fig_num)
    for sp_name, sp_slice in species.items():
        plt.scatter(
            x_data[sp_slice], 
            y_data[sp_slice], 
            label=sp_name.capitalize(),
            alpha=0.7
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Создание графиков
create_scatter_plot(
    1, all_features['sepal_length'], all_features['sepal_width'],
    'Длина чашелистика (см)', 'Ширина чашелистика (см)',
    'Соотношение длины и ширины чашелистика'
)

create_scatter_plot(
    2, all_features['sepal_length'], all_features['petal_length'],
    'Длина чашелистика (см)', 'Длина лепестка (см)',
    'Зависимость длины лепестка от длины чашелистика'
)

create_scatter_plot(
    3, all_features['sepal_length'], all_features['petal_width'],
    'Длина чашелистика (см)', 'Ширина лепестка (см)',
    'Зависимость ширины лепестка от длины чашелистика'
)

plt.figure(4)
for sp_name, sp_slice in species.items():
    plt.scatter(
        all_features['petal_length'][sp_slice],
        all_features['petal_width'][sp_slice],
        label=sp_name.capitalize(),
        alpha=0.7
    )
plt.xlabel('Длина лепестка (см)')
plt.ylabel('Ширина лепестка (см)')
plt.title('Соотношение длины и ширины лепестка')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# Статистический анализ
def calculate_statistics(data, prefix=""):
    return {
        'max': np.max(data),
        'min': np.min(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'prefix': prefix
    }

# Сбор статистики
stats = []

# Общая статистика по всем признакам
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    stats.append(calculate_statistics(all_features[feature], f"Общая {feature.replace('_', ' ')}"))

# Статистика по видам
for sp_name, sp_slice in species.items():
    for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        stats.append(calculate_statistics(
            all_features[feature][sp_slice],
            f"{sp_name.capitalize()} {feature.replace('_', ' ')}"
        ))

# Форматирование таблицы
table_header = ["Характеристика", "Максимум", "Минимум", "Среднее", "Стандартное отклонение"]
table_data = [table_header]

for stat in stats:
    table_data.append([
        stat['prefix'],
        f"{stat['max']:.2f}",
        f"{stat['min']:.2f}",
        f"{stat['mean']:.2f}",
        f"{stat['std']:.2f}"
    ])

# Сохранение результатов
with open('iris_statistics.txt', 'w', encoding='utf-8') as f:
    f.write(tabulate(table_data, headers='firstrow', tablefmt='grid'))
