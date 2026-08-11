import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def correlation_matrix(df):
    """
    Строит тепловую карту корреляций на очищенном от пропусков df.
    """

    df = df.copy()
    df_not_nan = df.dropna()

    # mapping
    df_not_nan['состояние_здоровья'] = df_not_nan['состояние_здоровья'].map({'fit': 0, 'at-risk': 1, 'unhealthy': 2})
    df_not_nan['тип_диеты'] = df_not_nan['тип_диеты'].map({'balanced': 0, 'veg': 1, 'non-veg': 2})
    df_not_nan['уровень_стресса'] = df_not_nan['уровень_стресса'].map({'low': 0, 'medium': 1, 'high': 2})
    df_not_nan['качество_сна'] = df_not_nan['качество_сна'].map({'good': 0, 'average': 1, 'poor': 2})
    df_not_nan['уровень_физ_активности'] = df_not_nan['уровень_физ_активности'].map(
        {'active': 0, 'moderate': 1, 'sedentary': 2})
    df_not_nan['курение_алкоголь'] = df_not_nan['курение_алкоголь'].map({'no': 0, 'occasional': 1, 'yes': 2})
    df_not_nan['пол'] = df_not_nan['пол'].map({'male': 0, 'other': 1, 'female': 2})

    # matrix
    correl = df_not_nan.corr()
    plt.figure(figsize=(22, 20))
    sns.heatmap(correl,
                annot=True,  # показывать числа
                cmap='coolwarm',  # цветовая схема (красный = 1, синий = -1)
                center=0,  # центр палитры (0 = белый)
                vmin=-1, vmax=1,  # диапазон значений
                square=True,  # квадратные клетки
                fmt='.2f',  # два знака после запятой
                linewidths=0.5,  # тонкие линии между клетками
                cbar_kws={'shrink': 0.8})  # размер цветовой шкалы
    plt.title('Матрица корреляции признаков', fontsize=14, fontweight='bold')
    plt.show()
    return correl
