import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Конфигурация отображения
plt.style.use('seaborn')
pd.set_option('display.max_columns', 15)
sns.set_palette('husl')

def load_telecom_data():
    """Загрузка и предобработка данных о клиентах телеком компании"""
    telecom_df = pd.read_csv("telecom_churn.csv")
    print(f"Данные успешно загружены. Размерность: {telecom_df.shape}")
    return telecom_df

# Основной анализ данных
def perform_eda(df):
    """Анализ и визуализация данных"""
    
    # 1. Анализ обращений в поддержку
    plt.figure(figsize=(10, 6))
    df['Customer service calls'].hist(bins=15, edgecolor='black')
    plt.title('Распределение обращений в клиентскую поддержку', pad=20)
    plt.xlabel('Количество обращений', labelpad=10)
    plt.ylabel('Частота', labelpad=10)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 2. Анализ дневных звонков
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Total day minutes'], width=0.4)
    plt.title('Распределение времени дневных разговоров', pad=20)
    plt.ylabel('Минуты разговора', labelpad=10)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 3. Топ-3 штата по времени разговоров
    top_states = df.groupby('State')['Total day minutes'].sum().nlargest(3).index
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='Total day minutes', 
        y='State', 
        data=df[df['State'].isin(top_states)], 
        hue='State',
        palette='Set2'
    )
    plt.title('Топ-3 штата по времени дневных разговоров', pad=20)
    plt.xlabel('Минуты разговора', labelpad=10)
    plt.ylabel('Штат', labelpad=10)
    plt.legend().remove()
    plt.grid(axis='x', alpha=0.3)
    plt.show()

    # 4. Распределение клиентов по штатам
    plt.figure(figsize=(14, 7))
    state_counts = df['State'].value_counts().sort_values(ascending=True)
    state_counts.plot(kind='barh')
    plt.title('Географическое распределение клиентов', pad=20)
    plt.xlabel('Количество клиентов', labelpad=10)
    plt.ylabel('Штат', labelpad=10)
    plt.grid(axis='x', alpha=0.3)
    plt.show()

    # 5. Парные зависимости
    numerical_features = ['Total day charge', 'Total intl charge', 'Customer service calls']
    pair_grid = sns.pairplot(
        df[numerical_features + ['Churn']], 
        hue='Churn',
        plot_kws={'alpha': 0.7},
        height=3
    )
    pair_grid.fig.suptitle('Анализ взаимосвязей показателей с оттоком', y=1.05)
    plt.show()

    # 6. Сравнение тарифов
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Total day charge', 
        y='Total intl charge', 
        hue='Churn',
        data=df,
        palette='coolwarm',
        alpha=0.8
    )
    plt.title('Сравнение дневных и международных тарифов', pad=20)
    plt.xlabel('Дневные начисления ($)', labelpad=10)
    plt.ylabel('Международные начисления ($)', labelpad=10)
    plt.grid(alpha=0.3)
    plt.show()

    # 7. Корреляционный анализ
    processed_df = df.iloc[:, :df.columns.get_loc("Churn")]
    processed_df = processed_df.drop(['International plan', 'Voice mail plan', 'State'], axis=1)
    processed_df = pd.get_dummies(processed_df)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        processed_df.corr(),
        cmap='Blues',
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Матрица корреляций между признаками', pad=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Основной блок выполнения
if __name__ == "__main__":
    telecom_data = load_telecom_data()
    perform_eda(telecom_data)



