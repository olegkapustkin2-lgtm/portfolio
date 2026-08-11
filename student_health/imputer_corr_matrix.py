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

# СОЗДАЕМ ИМПЬЮТЕР ДЛЯ ЗАПОЛНЕНИЯ ПРОПУСКОВ (ПОСЛЕДОВАТЕЛЬНЫЙ)
from catboost import CatBoostRegressor, CatBoostClassifier

class CustomImputer:
    def __init__(self, categorical_cols=None, order=None, min_samples=200, random_state=42):
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.order = order
        self.min_samples = min_samples
        self.models = {}
        self.random_state = random_state
        self.used_features = {}


    def fit_transform(self, X, y=None):
        X_copy = X.copy()

        # 1.

        # 2. Определяем порядок заполнения
        if self.order is None:
            cols_to_fill = [col for col in X_copy.columns if X_copy[col].isna().sum() > 0]
        else:
            cols_to_fill = [col for col in self.order if col in X_copy.columns and X_copy[col].isna().sum() > 0]

        print(f" Порядок заполнения: {cols_to_fill}")

        for col in cols_to_fill:
            print(f"\n Заполняем колонку: {col}")

            # 3. Фильтрация для таргета (работает с NaN)
            if col in self.categorical_cols:
                train_mask = X_copy[col].notna()  # <-- Используем notna() вместо != 'None'
            else:
                train_mask = X_copy[col].notna()

            feature_cols = [f for f in X_copy.columns if f != col and f in X_copy.columns and f != 'состояние_здоровья']
            self.used_features[col] = feature_cols

            X_train_known = X_copy[train_mask][feature_cols]
            y_train_known = X_copy[train_mask][col]

            # 4. Проверяем, достаточно ли данных
            if len(X_train_known) < 10:
                print(f"   ⚠️ Мало данных ({len(X_train_known)}). Заполняем константой.")
                if col in self.categorical_cols:
                    fill_val = 'Unknown'
                else:
                    fill_val = X_copy[col].median()
                missing_mask = X_copy[col].isna()
                X_copy.loc[missing_mask, col] = fill_val
                continue

            # 5. Определяем индексы категориальных колонок
            cat_feature_indices = [i for i, f in enumerate(feature_cols) if f in self.categorical_cols]

            # 5.1. Создаём валидацию
            from sklearn.model_selection import train_test_split

            if len(X_train_known) > 500:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_known, y_train_known,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_train_known if col in self.categorical_cols else None
                )

                # ЗАМЕНЯЕМ NaN НА 'None' В X_tr И X_val (для CatBoost)
                for cat_col in self.categorical_cols:
                    if cat_col in X_tr.columns and X_tr[cat_col].isna().any():
                        X_tr[cat_col] = X_tr[cat_col].fillna('None')
                    if cat_col in X_val.columns and X_val[cat_col].isna().any():
                        X_val[cat_col] = X_val[cat_col].fillna('None')

                eval_set = (X_val, y_val)
            else:
                X_tr, y_tr = X_train_known, y_train_known
                eval_set = None

            # 6. Обучаем CatBoost
            if col in self.categorical_cols:
                model = CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.03,
                    verbose=0,
                    random_state=self.random_state,
                    #auto_class_weights='Balanced',
                    nan_mode='Min',
                    cat_features=cat_feature_indices,
                    task_type="GPU",
                    early_stopping_rounds=50
                )
            else:
                model = CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.03,
                    verbose=0,
                    random_state=self.random_state,
                    nan_mode='Min',
                    cat_features=cat_feature_indices,
                    task_type="GPU",
                    early_stopping_rounds=50
                )

            model.fit(X_tr, y_tr, eval_set=eval_set, early_stopping_rounds=30, verbose=0)
            self.models[col] = model
            print(f"Модель обучена на {len(X_train_known)} строках.")

            # 7. Заполняем пропуски
            missing_mask = X_copy[col].isna()

            if missing_mask.sum() > 0:
                #X_missing = X_copy[missing_mask][feature_cols]
                X_missing = X_copy.loc[missing_mask, feature_cols].copy()
                # Заменяем NaN на 'None' в категориальных колонках перед предсказанием
                for cat_col in self.categorical_cols:
                    if cat_col in X_missing.columns and X_missing[cat_col].isna().any():
                        X_missing[cat_col] = X_missing[cat_col].fillna('None')

                predictions = model.predict(X_missing)

                # 8.ПРЕОБРАЗУЕМ ПРЕДСКАЗАНИЯ В ПРАВИЛЬНЫЙ ТИП
                if hasattr(model, 'classes_'):
                    # Категориальная колонка → строки
                    if len(predictions.shape) > 1:
                        predictions = predictions.ravel()
                    # reverse_mapping = {i: cls for i, cls in enumerate(model.classes_)}
                    # predictions = pd.Series(predictions).map(reverse_mapping).values
                else:
                    # Числовая колонка → float
                    predictions = predictions.astype(float)

                # 9. Записываем
                X_copy.loc[missing_mask, col] = predictions
                print('первые 5 предсказаний',predictions[:5])
                print('первые 5 предсказаний loc',X_copy.loc[missing_mask, col].head())
                print(f"   ✅ Заполнено {missing_mask.sum()} пропусков.")

            # Проверяем, что колонка заполнена
            print(f"   NaN в колонке ПОСЛЕ заполнения: {X_copy[col].isna().sum()}")

        return X_copy


    def transform(self, X):
        X_copy = X.copy()


        for col, model in self.models.items():
            if col not in X_copy.columns:
                continue

            missing_mask = X_copy[col].isna()  # <-- Ищем NaN, а не 'None'

            if missing_mask.sum() == 0:
                continue

            if not hasattr(model, 'predict'):
                X_copy[col] = X_copy[col].fillna(model)
                continue

            feature_cols = self.used_features.get(col, [f for f in X_copy.columns if f != col])
            X_missing = X_copy[missing_mask][feature_cols]

            #ЗАМЕНЯЕМ NaN НА 'None' В X_missing
            for cat_col in self.categorical_cols:
                if cat_col in X_missing.columns and X_missing[cat_col].isna().any():
                    X_missing[cat_col] = X_missing[cat_col].fillna('None')


            predictions = model.predict(X_missing)

            if hasattr(model, 'classes_'):
                if len(predictions.shape) > 1:
                    predictions = predictions.ravel()
            #     reverse_mapping = {i: cls for i, cls in enumerate(model.classes_)}
            #     predictions = pd.Series(predictions).map(reverse_mapping).values
            # else:
            #     predictions = predictions.astype(float)

            X_copy.loc[missing_mask, col] = predictions

        return X_copy

# НЕ ПОСЛЕДОВАТЕЛЬНОЕ ЗАПОЛНЕНИЕ (ОТСУТСТВИЕ МУЛЬТИПЛИЦИРОВАНИЯ ОШИБОК ПОСЛЕД. ЗАПОЛНЕНИЯ)
# СОЗДАЕМ ИМПЬЮТЕР ДЛЯ ЗАПОЛНЕНИЯ ПРОПУСКОВ
from catboost import CatBoostRegressor, CatBoostClassifier

class CustomImputer_2:
    def __init__(self, categorical_cols=None, order=None, min_samples=200, random_state=42):
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.order = order
        self.min_samples = min_samples
        self.models = {}
        self.random_state = random_state
        self.used_features = {}


    def fit_transform(self, X, y=None):
        X_original = X.copy()
        X_copy = X.copy()
        # 1.

        # 2. Определяем порядок заполнения
        if self.order is None:
            cols_to_fill = [col for col in X_copy.columns if X_copy[col].isna().sum() > 0]
        else:
            cols_to_fill = [col for col in self.order if col in X_copy.columns and X_copy[col].isna().sum() > 0]

        print(f" Порядок заполнения: {cols_to_fill}")

        for col in cols_to_fill:
            print(f"\n Заполняем колонку: {col}")

            # 3. Фильтрация для таргета (работает с NaN)
            if col in self.categorical_cols:
                train_mask = X_copy[col].notna()
            else:
                train_mask = X_copy[col].notna()

            feature_cols = [f for f in X_copy.columns if f != col and f in X_copy.columns and f != 'состояние_здоровья']
            self.used_features[col] = feature_cols

            # X_train_known = X_copy[train_mask][feature_cols]
            y_train_known = X_copy[train_mask][col]
            X_train_known = X_original.loc[train_mask, feature_cols]
            #
            # # 4. Проверяем, достаточно ли данных
            # if len(X_train_known) < 10:
            #     print(f"    Мало данных ({len(X_train_known)}). Заполняем константой.")
            #     if col in self.categorical_cols:
            #         fill_val = 'Unknown'
            #     else:
            #         fill_val = X_copy[col].median()
            #     missing_mask = X_copy[col].isna()
            #     X_copy.loc[missing_mask, col] = fill_val
            #     continue

            # 5. Определяем индексы категориальных колонок
            cat_feature_indices = [i for i, f in enumerate(feature_cols) if f in self.categorical_cols]

            # 5.1. Создаём валидацию
            from sklearn.model_selection import train_test_split

            if len(X_train_known) > 500:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_known, y_train_known,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_train_known if col in self.categorical_cols else None
                )

                #  ЗАМЕНЯЕМ NaN НА 'None' В X_tr И X_val (для CatBoost)
                for cat_col in self.categorical_cols:
                    if cat_col in X_tr.columns and X_tr[cat_col].isna().any():
                        X_tr[cat_col] = X_tr[cat_col].fillna('None')
                    if cat_col in X_val.columns and X_val[cat_col].isna().any():
                        X_val[cat_col] = X_val[cat_col].fillna('None')

                eval_set = (X_val, y_val)
            else:
                X_tr, y_tr = X_train_known, y_train_known
                eval_set = None

            # 6. Обучаем CatBoost
            if col in self.categorical_cols:
                model = CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.03,
                    verbose=0,
                    random_state=self.random_state,
                    #auto_class_weights='Balanced',
                    nan_mode='Min',
                    cat_features=cat_feature_indices,
                    task_type="GPU",
                    early_stopping_rounds=50
                )
            else:
                model = CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.03,
                    verbose=0,
                    random_state=self.random_state,
                    nan_mode='Min',
                    cat_features=cat_feature_indices,
                    task_type="GPU",
                    early_stopping_rounds=50
                )

            model.fit(X_tr, y_tr, eval_set=eval_set, early_stopping_rounds=30, verbose=0)
            self.models[col] = model
            print(f"    Модель обучена на {len(X_train_known)} строках.")

            # 7. Заполняем пропуски
            missing_mask = X_copy[col].isna()

            if missing_mask.sum() > 0:
                #X_missing = X_copy[missing_mask][feature_cols]
                X_missing = X_original.loc[missing_mask, feature_cols].copy()
                #X_missing = X_copy.loc[missing_mask, feature_cols].copy()

                # Заменяем NaN на 'None' в категориальных колонках перед предсказанием
                for cat_col in self.categorical_cols:
                    if cat_col in X_missing.columns and X_missing[cat_col].isna().any():
                        X_missing[cat_col] = X_missing[cat_col].fillna('None')

                predictions = model.predict(X_missing)

                # 8. ПРЕОБРАЗУЕМ ПРЕДСКАЗАНИЯ В ПРАВИЛЬНЫЙ ТИП
                if hasattr(model, 'classes_'):
                    # Категориальная колонка → строки
                    if len(predictions.shape) > 1:
                        predictions = predictions.ravel()
                    # reverse_mapping = {i: cls for i, cls in enumerate(model.classes_)}
                    # predictions = pd.Series(predictions).map(reverse_mapping).values
                else:
                    # Числовая колонка → float
                    predictions = predictions.astype(float)

                # 9. Записываем
                X_copy.loc[missing_mask, col] = predictions
                print('первые 5 предсказаний',predictions[:5])
                print('первые 5 предсказаний loc',X_copy.loc[missing_mask, col].head())
                print(f" Заполнено {missing_mask.sum()} пропусков.")

            # Проверяем, что колонка заполнена
            print(f"   NaN в колонке ПОСЛЕ заполнения: {X_copy[col].isna().sum()}")

        return X_copy


    def transform(self, X):
        X_original = X.copy()
        X_copy = X.copy()

        for col, model in self.models.items():
            if col not in X_copy.columns:
                continue

            # Теперь ищем NaN (как в fit_transform)
            missing_mask = X_copy[col].isna()

            if missing_mask.sum() == 0:
                continue

            if not hasattr(model, 'predict'):
                X_copy[col] = X_copy[col].fillna(model)
                continue

            feature_cols = self.used_features.get(col, [f for f in X_copy.columns if f != col])
            #X_missing = X_copy[missing_mask][feature_cols]
            X_missing = X_original.loc[missing_mask, feature_cols].copy()

            # ЗАМЕНЯЕМ NaN НА 'None' В X_missing (для CatBoost)
            for cat_col in self.categorical_cols:
                if cat_col in X_missing.columns and X_missing[cat_col].isna().any():
                    X_missing[cat_col] = X_missing[cat_col].fillna('None')


            # ПРИВОДИМ К ОДНОМЕРНОМУ МАССИВУ (ЕСЛИ НУЖНО)
            predictions = model.predict(X_missing)

            if hasattr(model, 'classes_'):
                if len(predictions.shape) > 1:
                    predictions = predictions.ravel()

            X_copy.loc[missing_mask, col] = predictions

        return X_copy

