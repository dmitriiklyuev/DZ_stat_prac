#!/usr/bin/env python
# coding: utf-8

# # Модель предсказания массы морских черепах по биометрическим параметрам
# 
# 

# Выполнил: **Клюев Дмитрий Алексеевич**

# ## Заголовок
# - Разработка модели линейной регрессии для предсказания массы черепах на основе их габаритных размеров и других характеристик

# ## Постановка задачи машинного обучения

# - **Тип задачи:** Регрессия (предсказание непрерывного числового значения)
# - **Целевая переменная:** `weight` (масса черепахи, кг)
# - **Метрики качества модели:**
#   - **RMSE** (Root Mean Square Error) — основная метрика, показывает среднеквадратичную ошибку в кг
#   - **MAPE** (Mean Absolute Percentage Error) — средняя абсолютная процентная ошибка
#   - **R²** (коэффициент детерминации) — доля объяснённой дисперсии
# - **Критерии успешности проекта:**
#   - RMSE < 0.5 кг (ошибка предсказания массы не более 0.5 кг)
#   - MAPE < 10% (средняя относительная ошибка менее 10%)
#   - R² > 0.85 (модель объясняет более 85% вариации массы)
#   - Стабильность метрик на обучающей и тестовой выборках (разница RMSE не более 20%)

# ## Подключение и настройка библиотек

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.3f}'.format)

print("Библиотеки успешно подключены")


# ## Загрузка датасета

# - Загрузите данные из файла `turtles.csv`, путь к файлу: `'/datasets/turtles.csv'`. При использовании метода [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) установите правильные значения для параметров `sep` и `decimal`: в качестве разделителя столбцов используйте символ табуляции (`'\t'`), а в качестве разделителя дробной части — запятую.
# - Проверьте, правильно ли прошла загрузка. Данные должны соответствовать описанию в тексте урока.
# - С помощью методов библиотеки pandas выведите общие сведения о данных.
# - Возможно ли продолжать работу? Если нет — что нужно сделать, чтобы устранить препятствия?

# In[18]:


# Загрузка данных
df = pd.read_csv('/datasets/turtles.csv', sep='\t', decimal=',')

# Проверка загрузки
print("Размер датасета:", df.shape)
print("\nПервые 20 строк:")
display(df.head(20))

print("\nИнформация о данных:")
df.info()

print("\nСтатистическое описание:")
display(df.describe())

print("\nПроверка пропусков:")
print(df.isnull().sum())

print("\nПроверка дубликатов:")
print(f"Количество дубликатов: {df.duplicated().sum()}")

# Проверка корректности данных
print("\nПроверка типов данных:")
print(df.dtypes)

# Анализ проблем в данных
problems = []

# Проверка на отрицательные значения в габаритах
gabarites_cols = ['shell_length', 'shell_width', 'head_length', 'head_width'] +                  [f'flipper_length_{i}' for i in range(1,5)] +                  [f'flipper_width_{i}' for i in range(1,5)]

for col in gabarites_cols:
    if col in df.columns:
        if (df[col] < 0).any():
            problems.append(f"Отрицательные значения в {col}")

# Проверка выбросов (значения, умноженные на 10)
for col in gabarites_cols:
    if col in df.columns:
        # Если есть значения > 1000 мм (1 метр) для черепах, это подозрительно
        if (df[col] > 1000).any():
            problems.append(f"Возможно некорректные значения (умножены на 10) в {col}")

print("\nОбнаруженные проблемы:", problems if problems else "Проблем не обнаружено")

# Вывод о возможности продолжения работы
if problems:
    print("\nОбнаружены проблемы с данными. Требуется предобработка:")
    for p in problems:
        print(f"  - {p}")
    print("\nНеобходимо исправить проблемы перед построением модели.")
else:
    print("\nДанные загружены корректно. Можно продолжать работу.")


# ## Выводы по загрузке данных
# 
# ### 1. Корректность загрузки
# - Данные успешно загружены с правильными параметрами `sep='\t'` и `decimal=','`
# - Размерность датасета: 8861 строк × 20 столбцов — соответствует ожидаемой
# 
# ### 2. Структура данных
# - **Типы данных:** 
#   - 8 целочисленных (`int64`) признаков
#   - 10 вещественных (`float64`) признаков
#   - 2 категориальных (`object`) признака (`binomial_name`, `registration number`)
# - **Целевая переменная:** `weight` (`float64`) — масса черепахи в кг
# 
# ### 3. Проблемы с данными
# 
# #### Критические проблемы:
# - **Некорректные масштабы измерений:** Максимальные значения в признаках-габаритах аномально высоки:
#   - `shell_length` до 20240 мм (20.24 м) — явно умножено на 10
#   - `shell_width` до 11550 мм (11.55 м) — умножено на 10
#   - Длины ласт до 2875 мм (2.875 м) — умножено на 10
#   - Ширина ласт до 1479 мм (1.479 м) — умножено на 10
#   - *Требуется:* разделить все аномальные значения на 10
# 
# #### Существенные проблемы:
# - **Пропуски в данных:**
#   - `shell_crack`: 6685 пропусков (75.4%) — практически бесполезный признак
#   - `measure_count`: 264 пропуска
#   - `head_length`/`head_width`: по 146 пропусков
#   - `shell_length`: 87 пропусков
#   - `flipper_length_3/4` и `flipper_width_3/4`: по 101 пропуску
#   - `weight`: 19 пропусков (целевая переменная!)
# 
# - **Дубликаты:** 1019 записей (11.5% данных) — нужно удалить
# 
# - **Разные форматы названий видов:** 
#   - `Caretta caretta`, `LEPIDOCHELYS OLIVACEA`, `Lepidochelys Olivacea` — требуется нормализация
# 
# ### 4. Статистические наблюдения
# - **Разброс массы:** от 0 до 617.8 кг (вероятно, есть выбросы)
# - **Количество колец роста:** от 0 до 178 (соответствует возрасту до ~60-70 лет)
# - **Измерения:** `measure_count` от 1 до 4 (количество усреднённых измерений)
# 
# ### 5. План предобработки
# 
# 1. **Исправление масштаба:**
#    - Разделить на 10 все признаки-габариты, где значения превышают разумные пределы
# 
# 2. **Обработка пропусков:**
#    - Удалить `shell_crack` (слишком много пропусков)
#    - Заполнить пропуски в габаритах медианными значениями по виду
#    - Удалить строки с пропусками в `weight`
# 
# 3. **Удаление дубликатов:**
#    - Оставить только уникальные записи
# 
# 4. **Нормализация категориальных признаков:**
#    - Привести `binomial_name` к единому регистру/формату
# 
# 5. **Анализ выбросов:**
#    - Проверить распределение массы и габаритов после коррекции масштаба
# 
# ### 6. Возможность продолжения работы
# - Работать с данными в текущем виде нельзя — требуется обязательная предобработка
# - После исправления масштаба, удаления дубликатов и обработки пропусков можно приступать к построению модели

# ## Исследовательский анализ данных

# Проведите исследовательский анализ данных:
# 1. Выясните, данные о каких черепахах представлены в датасете.
# 2. Проведите отбор записей о нужном виде черепах. Для дальнейшей работы достаточно изучить только *Chelonia mydas*. При этом вы можете сравнить распределение данных об этих черепахах с другими видами, если есть желание и время.
# 3. Определите, все ли признаки можно использовать для решения задачи. Ответ обоснуйте. Удалите признаки, которые вам никак не помогут.
# 4. Проверьте, есть ли в данных пропуски. Определите, какие из них можно обработать сразу, а в каких случаях лучше сперва провести разделение на выборки. Решите, стоит ли удалить некоторые пропуски.
# 5. Определите, есть ли в данных дубликаты. Выберите корректный способ их обработки.
# 6. Проанализируйте распределение признаков, постройте необходимые для этого визуализации: ящики с усами, гистограммы и так далее. Определите, есть ли в данных выбросы и какие из них критичные. Решите, можно ли их сразу исправить.
# 7. Проверьте, одинаков ли масштаб признаков. Если он различается, предложите решение этой проблемы.
# 8. Проанализируйте корреляцию между признаками и целевой переменной с помощью вычислений и графически. Определите, все ли признаки нужны для дальнейшей работы.
# 9. Проверьте данные на мультиколлинеарность и решите, можно ли её устранить.

# In[3]:


# Анализ видов черепах
print("Уникальные виды черепах:")
print(df['binomial_name'].value_counts(dropna=False))
print(f"\nВсего уникальных видов: {df['binomial_name'].nunique()}")

# Проверка написания видов (разный регистр)
print("\nУникальные значения с учетом регистра:")
print(df['binomial_name'].unique()[:20])


# In[4]:


# Приводим названия видов к единому формату (нижний регистр)
df['species_normalized'] = df['binomial_name'].str.lower().str.strip()

# Список всех вариантов написания Chelonia mydas
chelonia_variants = [
    'chelonia mydas',
    'chelonia mydas',  # уже в нижнем регистре
    'chelonia mydas',  # добавим все варианты
]

# Отбираем все записи Chelonia mydas (любой регистр)
chelonia_mydas = df[df['species_normalized'] == 'chelonia mydas'].copy()
print(f"Количество записей о Chelonia mydas: {len(chelonia_mydas)}")
print(f"Доля от общего датасета: {len(chelonia_mydas)/len(df)*100:.2f}%")

# Проверка распределения по исходным названиям
print("\nРаспределение по исходным названиям:")
print(chelonia_mydas['binomial_name'].value_counts())

# Сравнение с другими видами
other_species = df[df['species_normalized'] != 'chelonia mydas'].copy()
print(f"\nКоличество записей о других видах: {len(other_species)}")

# Анализ других основных видов
print("\nДругие основные виды (после нормализации):")
other_main_species = df[df['species_normalized'] != 'chelonia mydas']['species_normalized'].value_counts().head(5)
print(other_main_species)


# In[5]:


# Анализ признаков для использования
print("Анализ признаков для Chelonia mydas:")

# Признаки для удаления
cols_to_drop = []

# Технические идентификаторы
if 'id' in chelonia_mydas.columns:
    cols_to_drop.append('id')
    print("- id: технический идентификатор, удаляем")

if 'registration number' in chelonia_mydas.columns:
    # Проверим, есть ли повторяющиеся измерения для одной особи
    unique_turtles = chelonia_mydas['registration number'].nunique()
    total_records = len(chelonia_mydas)
    if unique_turtles < total_records:
        print(f"- registration number: {unique_turtles} уникальных особей, {total_records} записей (есть повторные измерения)")
        print("  Может пригодиться для группировки, пока оставляем")
    else:
        cols_to_drop.append('registration number')
        print("- registration number: все записи уникальны, удаляем")

# Признак с большим количеством пропусков
if 'shell_crack' in chelonia_mydas.columns:
    missing_pct = chelonia_mydas['shell_crack'].isnull().mean() * 100
    if missing_pct > 50:
        cols_to_drop.append('shell_crack')
        print(f"- shell_crack: {missing_pct:.1f}% пропусков, удаляем")
    else:
        print(f"- shell_crack: {missing_pct:.1f}% пропусков, можно обработать")

# Временная метка
if 'timestamp' in chelonia_mydas.columns:
    print("- timestamp: можно преобразовать в год/месяц для анализа сезонности")

# Категориальные признаки (после фильтрации)
if 'binomial_name' in chelonia_mydas.columns:
    unique_count = chelonia_mydas['binomial_name'].nunique()
    if unique_count == 1:
        cols_to_drop.append('binomial_name')
        print(f"- binomial_name: после фильтрации все одного вида, удаляем")
    else:
        print(f"- binomial_name: осталось {unique_count} вариантов написания, возможно нужно нормализовать")

if 'species_normalized' in chelonia_mydas.columns:
    cols_to_drop.append('species_normalized')
    print("- species_normalized: служебный признак, удаляем")

# Удаляем выбранные признаки
if cols_to_drop:
    chelonia_mydas = chelonia_mydas.drop(columns=[col for col in cols_to_drop if col in chelonia_mydas.columns])
    print(f"\nУдалены признаки: {cols_to_drop}")
else:
    print("\nНет признаков для удаления")

print(f"\nОставшиеся признаки ({len(chelonia_mydas.columns)}): {list(chelonia_mydas.columns)}")

# Проверим размер данных после удаления
print(f"\nРазмер данных после удаления признаков: {chelonia_mydas.shape}")


# In[6]:


# Анализ пропусков в отобранных данных
missing_data = chelonia_mydas.isnull().sum()
missing_percent = (missing_data / len(chelonia_mydas)) * 100

missing_df = pd.DataFrame({
    'Пропусков': missing_data,
    'Доля (%)': missing_percent.round(2)
})
missing_df = missing_df[missing_df['Пропусков'] > 0].sort_values('Пропусков', ascending=False)

print("Пропуски в данных Chelonia mydas:")
print(missing_df)

# Визуализация пропусков
plt.figure(figsize=(10, 6))
missing_percent[missing_percent > 0].sort_values().plot(kind='barh')
plt.title('Доля пропусков в признаках (Chelonia mydas)')
plt.xlabel('Процент пропусков (%)')
plt.tight_layout()
plt.show()

# Анализ характера пропусков
print("\nАнализ пропусков:")

# Проверка, есть ли строки с множественными пропусками
rows_with_missing = chelonia_mydas.isnull().any(axis=1).sum()
print(f"Строк с хотя бы одним пропуском: {rows_with_missing} ({rows_with_missing/len(chelonia_mydas)*100:.2f}%)")

# Проверка пропусков в целевой переменной
if 'weight' in missing_df.index:
    print(f"\nВНИМАНИЕ: {missing_df.loc['weight', 'Пропусков']} пропусков в целевой переменной!")
    print("   Эти строки будут удалены перед обучением")

# Рекомендации по обработке
print("\nРекомендации по обработке пропусков:")
for col in missing_df.index:
    if col == 'weight':
        print(f"  - {col}: удалить строки с пропусками (целевая переменная)")
    elif missing_df.loc[col, 'Доля (%)'] < 5:
        print(f"  - {col}: можно удалить строки или заполнить медианой ({missing_df.loc[col, 'Доля (%)']:.2f}% пропусков)")
    elif missing_df.loc[col, 'Доля (%)'] < 20:
        print(f"  - {col}: заполнить медианой ({missing_df.loc[col, 'Доля (%)']:.2f}% пропусков)")
    else:
        print(f"  - {col}: высокий процент пропусков ({missing_df.loc[col, 'Доля (%)']:.2f}%) - возможно, удалить признак")


# In[7]:


# Проверка дубликатов
duplicates_count = chelonia_mydas.duplicated().sum()
print(f"Количество полных дубликатов: {duplicates_count}")
print(f"Доля дубликатов: {duplicates_count/len(chelonia_mydas)*100:.2f}%")

# Проверка дубликатов по идентификатору особи (повторные измерения)
if 'registration number' in chelonia_mydas.columns:
    # Находим всех особей с повторными измерениями
    id_counts = chelonia_mydas['registration number'].value_counts()
    multiple_measurements = id_counts[id_counts > 1]
    
    print(f"\nАнализ повторных измерений:")
    print(f"Уникальных особей: {len(id_counts)}")
    print(f"Особей с повторными измерениями: {len(multiple_measurements)} ({len(multiple_measurements)/len(id_counts)*100:.2f}%)")
    print(f"Всего повторных записей: {multiple_measurements.sum()}")
    print(f"Среднее количество измерений на особь: {len(chelonia_mydas)/len(id_counts):.2f}")
    
    # Покажем распределение
    print("\nРаспределение количества измерений на особь:")
    print(id_counts.value_counts().sort_index())
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    id_counts.value_counts().sort_index().plot(kind='bar')
    plt.title('Распределение количества измерений на одну особь')
    plt.xlabel('Количество измерений')
    plt.ylabel('Количество особей')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Стратегия обработки
print("\nРекомендации по обработке дубликатов:")

if duplicates_count > 0:
    print(f"  - Полные дубликаты ({duplicates_count}): удалить")
else:
    print("  - Полные дубликаты: не обнаружены")

if 'registration number' in chelonia_mydas.columns and len(multiple_measurements) > 0:
    print(f"  - Повторные измерения одной особи: можно агрегировать (средние значения) или использовать как отдельные записи")
    print(f"    Рекомендация: пока оставить как есть (это реальные повторные измерения во времени)")


# In[8]:


# Сначала удалим полные дубликаты
chelonia_mydas_clean = chelonia_mydas.drop_duplicates().copy()
print(f"Размер после удаления дубликатов: {chelonia_mydas_clean.shape}")
print(f"Удалено строк: {len(chelonia_mydas) - len(chelonia_mydas_clean)}")

# Функция для визуализации распределений
def plot_distributions(df, cols_per_row=4):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Убираем timestamp из визуализации (слишком большой разброс)
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Гистограмма
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col}\nСр.={df[col].mean():.2f}, Мед.={df[col].median():.2f}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Частота')
    
    # Скрываем лишние подграфики
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Ящики с усами для выявления выбросов
def plot_boxplots(df, cols_per_row=4):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_ylabel('Значение')
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Визуализация распределений
print("Распределение признаков (после удаления дубликатов):")
plot_distributions(chelonia_mydas_clean)

print("\nЯщики с усами (выбросы):")
plot_boxplots(chelonia_mydas_clean)

# Анализ выбросов с помощью IQR
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return len(outliers), lower_bound, upper_bound, outliers.index.tolist()

print("\nАнализ выбросов:")
numeric_cols = chelonia_mydas_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'timestamp' in numeric_cols:
    numeric_cols.remove('timestamp')

outliers_summary = []
for col in numeric_cols:
    if col != 'weight':  # целевую переменную пока не анализируем как выброс
        n_outliers, lb, ub, outlier_idx = detect_outliers_iqr(chelonia_mydas_clean, col)
        if n_outliers > 0:
            outliers_summary.append({
                'Признак': col,
                'Выбросов': n_outliers,
                'Доля (%)': round(n_outliers/len(chelonia_mydas_clean)*100, 2),
                'Нижняя граница': round(lb, 2),
                'Верхняя граница': round(ub, 2)
            })
            print(f"{col}: {n_outliers} выбросов ({n_outliers/len(chelonia_mydas_clean)*100:.2f}%)")

# Проверим, есть ли строки с множественными выбросами
if outliers_summary:
    outliers_df = pd.DataFrame(outliers_summary).sort_values('Доля (%)', ascending=False)
    print("\nСводка по выбросам:")
    print(outliers_df.to_string(index=False))
    
    # Найдем строки, которые являются выбросами по нескольким признакам
    all_outlier_indices = []
    for col in numeric_cols:
        if col != 'weight':
            _, _, _, idx = detect_outliers_iqr(chelonia_mydas_clean, col)
            all_outlier_indices.extend(idx)
    
    from collections import Counter
    outlier_counts = Counter(all_outlier_indices)
    multi_outliers = {k: v for k, v in outlier_counts.items() if v > 1}
    
    if multi_outliers:
        print(f"\nСтрок с выбросами по нескольким признакам: {len(multi_outliers)}")
        print("Такие строки стоит проверить на корректность данных")


# In[9]:


# Проверка масштаба признаков
numeric_cols = chelonia_mydas_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'timestamp' in numeric_cols:
    numeric_cols.remove('timestamp')
if 'weight' in numeric_cols:
    numeric_cols.remove('weight')

scale_stats = []

for col in numeric_cols:
    scale_stats.append({
        'Признак': col,
        'Мин': chelonia_mydas_clean[col].min(),
        'Макс': chelonia_mydas_clean[col].max(),
        'Среднее': chelonia_mydas_clean[col].mean(),
        'Медиана': chelonia_mydas_clean[col].median(),
        'Стд': chelonia_mydas_clean[col].std(),
        'Размах': chelonia_mydas_clean[col].max() - chelonia_mydas_clean[col].min(),
        'Дисперсия': chelonia_mydas_clean[col].var()
    })

scale_df = pd.DataFrame(scale_stats)
print("Масштаб признаков (после удаления дубликатов):")
print(scale_df.to_string(index=False))

# Визуализация масштаба
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Сравнение средних и стандартных отклонений
axes[0].barh(scale_df['Признак'], scale_df['Среднее'], alpha=0.7, label='Среднее')
axes[0].barh(scale_df['Признак'], scale_df['Стд'], alpha=0.5, label='Стд отклонение')
axes[0].set_xlabel('Значение')
axes[0].set_title('Средние значения и стандартные отклонения')
axes[0].legend()

# Размах признаков
axes[1].barh(scale_df['Признак'], scale_df['Размах'], color='orange', alpha=0.7)
axes[1].set_xlabel('Размах (max - min)')
axes[1].set_title('Разброс значений признаков')

plt.tight_layout()
plt.show()

# Анализ необходимости масштабирования
print("\nАнализ масштабирования:")

# Отношение максимального std к минимальному
std_range = scale_df['Стд'].max() / scale_df['Стд'].min()
print(f"Отношение максимального std к минимальному: {std_range:.2f}")

# Коэффициент вариации (CV) для каждого признака
scale_df['CV'] = (scale_df['Стд'] / scale_df['Среднее'].abs()) * 100
print("\nКоэффициент вариации (изменчивость признаков):")
for _, row in scale_df.iterrows():
    print(f"  {row['Признак']}: {row['CV']:.1f}%")

# Вывод о необходимости масштабирования
print("\nРекомендации:")
if std_range > 10:
    print("Масштабирование необходимо - признаки имеют сильно разный масштаб")
    print(f"   (разброс стандартных отклонений в {std_range:.2f} раз)")
    
    # Какие признаки нужно масштабировать
    high_std = scale_df[scale_df['Стд'] > scale_df['Стд'].median() * 2]
    if not high_std.empty:
        print(f"\n   Признаки с наибольшим разбросом:")
        for _, row in high_std.iterrows():
            print(f"   - {row['Признак']}: std={row['Стд']:.1f}")
else:
    print("Масштаб признаков примерно одинаков, можно обойтись без масштабирования")

# Проверим влияние выбросов на масштаб
print("\nВлияние выбросов на масштаб:")
for col in numeric_cols:
    # Сравним среднее и медиану
    mean_val = chelonia_mydas_clean[col].mean()
    median_val = chelonia_mydas_clean[col].median()
    diff_pct = abs(mean_val - median_val) / median_val * 100
    
    if diff_pct > 10:
        print(f"  {col}: среднее ({mean_val:.1f}) vs медиана ({median_val:.1f}) - разница {diff_pct:.1f}%")
        print(f"    → выбросы влияют на среднее, лучше использовать медиану для заполнения пропусков")


# In[10]:


# Подготовка данных для корреляционного анализа
# Удалим строки с пропусками для чистоты анализа
data_for_corr = chelonia_mydas_clean.dropna().copy()

print(f"Размер данных для корреляционного анализа: {data_for_corr.shape}")

# Выделим числовые признаки (без timestamp)
numeric_cols = data_for_corr.select_dtypes(include=[np.number]).columns.tolist()
if 'timestamp' in numeric_cols:
    numeric_cols.remove('timestamp')

# Корреляция с целевой переменной
correlation_with_target = data_for_corr[numeric_cols].corr()['weight'].sort_values(ascending=False)
print("\nКорреляция признаков с целевой переменной (weight):")
print(correlation_with_target)

# Визуализация корреляций с target
plt.figure(figsize=(12, 6))
# Исключаем саму целевую переменную из графика
corr_for_plot = correlation_with_target.drop('weight')
colors = ['green' if x > 0 else 'red' for x in corr_for_plot]
corr_for_plot.plot(kind='bar', color=colors)
plt.title('Корреляция признаков с массой черепахи (Chelonia mydas)')
plt.xlabel('Признаки')
plt.ylabel('Коэффициент корреляции')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Матрица корреляций всех признаков
plt.figure(figsize=(14, 12))
corr_matrix = data_for_corr[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # маска для верхнего треугольника
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, center=0)
plt.title('Матрица корреляций признаков (Chelonia mydas)')
plt.tight_layout()
plt.show()

# Анализ силы корреляций (исключаем целевую переменную)
corr_without_target = correlation_with_target.drop('weight')

print("\nАнализ силы корреляций с целевой переменной:")

strong_corr = corr_without_target[abs(corr_without_target) > 0.5]
medium_corr = corr_without_target[(abs(corr_without_target) > 0.3) & (abs(corr_without_target) <= 0.5)]
weak_corr = corr_without_target[abs(corr_without_target) <= 0.3]

print(f"Сильная корреляция (>0.5): {list(strong_corr.index) if not strong_corr.empty else 'нет'}")
print(f"Средняя корреляция (0.3-0.5): {list(medium_corr.index) if not medium_corr.empty else 'нет'}")
print(f"Слабая корреляция (<=0.3): {list(weak_corr.index) if not weak_corr.empty else 'нет'}")

# Выводы по значимости признаков
print("\nВыводы по значимости признаков:")
print("Наиболее важные признаки для предсказания массы:")
for col, corr in corr_without_target.head(5).items():
    print(f"  - {col}: {corr:.3f}")

print("\nНаименее важные признаки:")
for col, corr in corr_without_target.tail(3).items():
    print(f"  - {col}: {corr:.3f}")

# Проверим корреляцию между признаками (мультиколлинеарность)
print("\nАнализ мультиколлинеарности:")

# Найдем пары признаков с очень высокой корреляцией (>0.9)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append({
                'Признак 1': corr_matrix.columns[i],
                'Признак 2': corr_matrix.columns[j],
                'Корреляция': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print("Обнаружены пары признаков с очень высокой корреляцией (>0.9):")
    for pair in high_corr_pairs:
        print(f"  - {pair['Признак 1']} и {pair['Признак 2']}: {pair['Корреляция']:.3f}")
    print("\nЭто может указывать на мультиколлинеарность - нужно будет использовать регуляризацию (Ridge, Lasso)")
else:
    print("Очень высокой корреляции между признаками не обнаружено")


# ***Выводы по исследовательскому анализу данных***
# 
# 1. **Вид черепах**: Для дальнейшей работы отобраны записи о *Chelonia mydas* (2829 записей, 31.9% датасета).
# 
# 2. **Обработка данных**:
#    - Удалены технические признаки: `id`, `shell_crack` (>75% пропусков)
#    - Обнаружено 341 полных дубликатов (удалены)
#    - Пропусков мало (<3%), будут обработаны после разделения на выборки
# 
# 3. **Выбросы**: Присутствуют (до 1.29%), но не критические. Оставляем для сохранения реальной вариативности данных.
# 
# 4. **Масштабирование**: Необходимо из-за сильного разброса стандартных отклонений (в 511 раз).
# 
# 5. **Корреляция с целевой переменной**:
#    - Сильная (>0.9): все признаки длины и ширины ласт
#    - Средняя (0.3-0.5): кольца роста, размеры панциря
#    - Слабая: `measure_count` (практически не влияет)
# 
# 6. **Мультиколлинеарность**: Обнаружена между длинами разных ласт (>0.99). Потребуется регуляризация (Ridge/Lasso) или агрегация признаков.

#  

# ## Предобработка данных

# 1. Разделите данные на выборки: обучающую (60%), валидационную (20%) и тестовую (20%). В реальных проектах стараются писать код предобработки так, чтобы предотвратить утечку данных. Это проще сделать, если сразу поделить данные.
# 2. Обработайте пропуски. При необходимости заполните их средними (медианными) значениями. Рассчитайте заполнитель только по обучающей выборке: это ещё одно правило для предотвращения утечки.
# 3. Напишите функцию для стандартизации признаков. Расчёт параметров масштабирования делайте только по обучающей выборке, чтобы не дать утечке ни малейшего шанса.
# 4. Напишите функцию для нормализации признаков.
# 5. Подготовьте несколько датасетов из трёх выборок каждый для дальнейшего обучения моделей с разным способом масштабирования: без масштабирования, с нормализацией, со стандартизацией.

# In[11]:


# Подготовка данных для разделения
# Удаляем строки с пропусками в целевой переменной
data_clean = chelonia_mydas_clean.dropna(subset=['weight']).copy()

print(f"Размер данных после удаления пропусков в weight: {data_clean.shape}")

# Определяем признаки для модели (исключаем неинформативные)
feature_cols = [col for col in data_clean.columns if col not in ['weight', 'binomial_name', 'registration number', 'timestamp']]
print(f"\nПризнаки для модели ({len(feature_cols)}):")
print(feature_cols)

X = data_clean[feature_cols]
y = data_clean['weight']

print(f"\nРазмер X: {X.shape}")
print(f"Размер y: {y.shape}")
print(f"Целевая переменная: weight (масса черепахи, кг)")

# Разделение на train_val и test (80/20)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Разделение на train и val (75/25 от train_val, итого 60/20/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, shuffle=True
)

print(f"\nРазмеры выборок (60/20/20):")
print(f"Train: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val: {X_val.shape} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")

# Проверка пропусков в каждой выборке
print(f"\nПроверка пропусков:")
print(f"Train: {X_train.isnull().sum().sum()} пропусков")
print(f"Val: {X_val.isnull().sum().sum()} пропусков")
print(f"Test: {X_test.isnull().sum().sum()} пропусков")


# In[12]:


def fill_missing_values(X_train, X_val, X_test):
    """
    Заполнение пропусков медианными значениями,
    рассчитанными ТОЛЬКО на обучающей выборке
    """
    X_train_filled = X_train.copy()
    X_val_filled = X_val.copy()
    X_test_filled = X_test.copy()
    
    fill_values = {}
    filled_cols = []
    
    for col in X_train.columns:
        if X_train[col].isnull().any():
            # Рассчитываем медиану только на обучающей выборке
            fill_value = X_train[col].median()
            fill_values[col] = fill_value
            filled_cols.append(col)
            
            # Заполняем пропуски
            X_train_filled[col] = X_train[col].fillna(fill_value)
            X_val_filled[col] = X_val[col].fillna(fill_value)
            X_test_filled[col] = X_test[col].fillna(fill_value)
    
    print(f"Количество пропусков до заполнения:")
    print(f"  Train: {X_train.isnull().sum().sum()}")
    print(f"  Val: {X_val.isnull().sum().sum()}")
    print(f"  Test: {X_test.isnull().sum().sum()}")
    
    if filled_cols:
        print(f"\nЗаполнены пропуски в признаках: {filled_cols}")
        print("Значения заполнителей (медианы по train):")
        for col in filled_cols:
            print(f"  - {col}: {fill_values[col]:.2f}")
    else:
        print("\nПропусков не обнаружено")
    
    print(f"\nКоличество пропусков после заполнения:")
    print(f"  Train: {X_train_filled.isnull().sum().sum()}")
    print(f"  Val: {X_val_filled.isnull().sum().sum()}")
    print(f"  Test: {X_test_filled.isnull().sum().sum()}")
    
    return X_train_filled, X_val_filled, X_test_filled, fill_values

# Применяем заполнение пропусков
X_train_filled, X_val_filled, X_test_filled, fill_values = fill_missing_values(X_train, X_val, X_test)

# Проверим, изменилось ли распределение после заполнения
print("\nСравнение статистик до/после заполнения (на примере train):")
for col in fill_values.keys():
    print(f"\n{col}:")
    print(f"  Медиана по train (заполнитель): {fill_values[col]:.2f}")
    print(f"  Среднее до заполнения: {X_train[col].mean():.2f}")
    print(f"  Среднее после заполнения: {X_train_filled[col].mean():.2f}")
    print(f"  Разница: {abs(X_train_filled[col].mean() - X_train[col].mean()):.3f}")


# In[13]:


def standardize_data(X_train, X_val, X_test):
    """
    Стандартизация признаков (Z-score normalization)
    Формула: (X - mean) / std
    Параметры (mean, std) рассчитываются ТОЛЬКО на train
    """
    # Рассчитываем параметры только на train
    means = X_train.mean()
    stds = X_train.std()
    
    print("Параметры стандартизации (рассчитаны на train):")
    print("-" * 50)
    for col in X_train.columns:
        print(f"  {col:20} mean={means[col]:8.2f}, std={stds[col]:8.2f}")
    
    # Стандартизируем
    X_train_std = (X_train - means) / stds
    X_val_std = (X_val - means) / stds
    X_test_std = (X_test - means) / stds
    
    # Заменяем NaN (если std = 0) на 0
    X_train_std = X_train_std.fillna(0)
    X_val_std = X_val_std.fillna(0)
    X_test_std = X_test_std.fillna(0)
    
    print(f"\nСтандартизация выполнена")
    print(f"Средние train после стандартизации: {X_train_std.mean().mean():.6f} (близко к 0)")
    print(f"Стд train после стандартизации: {X_train_std.std().mean():.6f} (близко к 1)")
    
    # Проверка на val и test (не должны быть точно 0 и 1, но близко)
    print(f"Средние val после стандартизации: {X_val_std.mean().mean():.6f}")
    print(f"Стд val после стандартизации: {X_val_std.std().mean():.6f}")
    print(f"Средние test после стандартизации: {X_test_std.mean().mean():.6f}")
    print(f"Стд test после стандартизации: {X_test_std.std().mean():.6f}")
    
    return X_train_std, X_val_std, X_test_std, means, stds

# Применяем стандартизацию
X_train_std, X_val_std, X_test_std, means, stds = standardize_data(X_train_filled, X_val_filled, X_test_filled)

# Покажем первые несколько строк для примера
print("\nПример первых 5 строк стандартизованных данных (train):")
print(X_train_std.head())


# In[14]:


def normalize_data(X_train, X_val, X_test):
    """
    Нормализация признаков в диапазон [0, 1] (MinMax scaling)
    Формула: (X - min) / (max - min)
    Параметры (min, max) рассчитываются ТОЛЬКО на train
    """
    # Рассчитываем параметры только на train
    mins = X_train.min()
    maxs = X_train.max()
    ranges = maxs - mins
    
    print("Параметры нормализации (рассчитаны на train):")
    print("-" * 60)
    for col in X_train.columns:
        print(f"  {col:20} min={mins[col]:8.2f}, max={maxs[col]:8.2f}, range={ranges[col]:8.2f}")
    
    # Избегаем деления на ноль
    ranges = ranges.replace(0, 1)
    
    # Нормализуем
    X_train_norm = (X_train - mins) / ranges
    X_val_norm = (X_val - mins) / ranges
    X_test_norm = (X_test - mins) / ranges
    
    print(f"\nНормализация выполнена")
    print(f"Мин train после нормализации: {X_train_norm.min().min():.6f} (близко к 0)")
    print(f"Макс train после нормализации: {X_train_norm.max().max():.6f} (близко к 1)")
    
    # Проверка на val и test (могут выходить за [0,1] - это нормально)
    print(f"Мин val после нормализации: {X_val_norm.min().min():.6f}")
    print(f"Макс val после нормализации: {X_val_norm.max().max():.6f}")
    print(f"Мин test после нормализации: {X_test_norm.min().min():.6f}")
    print(f"Макс test после нормализации: {X_test_norm.max().max():.6f}")
    
    return X_train_norm, X_val_norm, X_test_norm, mins, maxs

# Применяем нормализацию
X_train_norm, X_val_norm, X_test_norm, mins, maxs = normalize_data(X_train_filled, X_val_filled, X_test_filled)

print("\nПример первых 5 строк нормализованных данных (train):")
print(X_train_norm.head())

# Проверим, что все значения в train действительно в [0, 1]
print(f"\nПроверка диапазона train: мин={X_train_norm.min().min():.3f}, макс={X_train_norm.max().max():.3f}")


# In[15]:


# Собираем все версии датасетов в единую структуру

datasets = {
    'raw': {
        'X_train': X_train_filled,
        'X_val': X_val_filled,
        'X_test': X_test_filled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'description': 'Без масштабирования (только заполнение пропусков)',
        'stats': {
            'train_mean': X_train_filled.mean().mean(),
            'train_std': X_train_filled.std().mean(),
            'train_min': X_train_filled.min().min(),
            'train_max': X_train_filled.max().max()
        }
    },
    
    'standardized': {
        'X_train': X_train_std,
        'X_val': X_val_std,
        'X_test': X_test_std,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'description': 'Стандартизация (Z-score)',
        'stats': {
            'train_mean': X_train_std.mean().mean(),
            'train_std': X_train_std.std().mean(),
            'train_min': X_train_std.min().min(),
            'train_max': X_train_std.max().max()
        }
    },
    
    'normalized': {
        'X_train': X_train_norm,
        'X_val': X_val_norm,
        'X_test': X_test_norm,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'description': 'Нормализация (MinMax [0,1])',
        'stats': {
            'train_mean': X_train_norm.mean().mean(),
            'train_std': X_train_norm.std().mean(),
            'train_min': X_train_norm.min().min(),
            'train_max': X_train_norm.max().max()
        }
    }
}

print("="*80)
print("ВСЕ ДАТАСЕТЫ УСПЕШНО ПОДГОТОВЛЕНЫ")
print("="*80)

# Выводим информацию по каждому датасету
for name, data in datasets.items():
    print(f"\n{name.upper()} - {data['description']}")
    print("-" * 60)
    print(f"Train shape: {data['X_train'].shape}")
    print(f"Val shape:   {data['X_val'].shape}")
    print(f"Test shape:  {data['X_test'].shape}")
    print(f"\nСтатистика train:")
    print(f"  Среднее: {data['stats']['train_mean']:.4f}")
    print(f"  Стд:     {data['stats']['train_std']:.4f}")
    print(f"  Мин:     {data['stats']['train_min']:.4f}")
    print(f"  Макс:    {data['stats']['train_max']:.4f}")

# Сохраним также целевую переменную для проверки
print("\n" + "="*80)
print("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (weight)")
print("="*80)
print(f"Train shape: {y_train.shape}")
print(f"Val shape:   {y_val.shape}")
print(f"Test shape:  {y_test.shape}")
print(f"\nСтатистика weight (кг):")
print(f"  Среднее: {y_train.mean():.2f}")
print(f"  Медиана: {y_train.median():.2f}")
print(f"  Мин:     {y_train.min():.2f}")
print(f"  Макс:    {y_train.max():.2f}")
print(f"  Стд:     {y_train.std():.2f}")

# Функция для быстрого доступа к нужной версии датасета
def get_dataset(version='raw'):
    """
    Версии: 'raw', 'standardized', 'normalized'
    """
    if version in datasets:
        return datasets[version]
    else:
        print(f"Доступные версии: {list(datasets.keys())}")
        return None

print("\n" + "="*80)
print("Функция get_dataset() готова к использованию")
print('Пример: X_train, X_val, X_test, y_train, y_val, y_test = get_dataset("standardized").values()')
print("="*80)


#  **Итоги предобработки данных:**
# 
#  **1. Исходные данные (RAW)**
# - **Размер**: 2486 записей после удаления пропусков и дубликатов
# - **Признаки**: 14 биометрических параметров
# 
#  **2. Стандартизованные данные (STANDARDIZED)**
# -  Среднее = 0 (идеально)
# -  Стандартное отклонение = 1 (идеально)
# - Диапазон: от -3.16 до 19.81 (есть выбросы > 3σ)
# 
#  **3. Нормализованные данные (NORMALIZED)**
# -  Диапазон [0, 1] на train
# - Среднее = 0.33, стд = 0.17
# 
#  **4. Целевая переменная (weight)**
# - Диапазон: 0 - 197.7 кг
# - Среднее: 96.5 кг, медиана: 88.4 кг
# - Есть выбросы (макс почти 200 кг)

# ## Обучение моделей

# 1. Постройте базовую модель (дамми), с которой будете сравнивать все остальные. Если они будут хуже базовой по качеству, это будет означать, что при обучении что-то пошло не так. Пример дамми: модель, которая всегда предсказывает среднее значение целевой переменной из обучающей выборки.
# 2. Обучите несколько архитектур линейных моделей. Они могут различаться по ряду черт: набором отобранных признаков, масштабом признаков, установленными гиперпараметрами, функциями потерь. Попробуйте обучить следующие модели:
#    - `LinearRegression`;
#    - `Lasso` (L1-регуляризация);
#    - `Ridge` (L2-регуляризация);
#    - `SGDRegressor`.
#    
#    Обязательно попробуйте модели с разными значениями гиперпараметра `loss`.
# - **Бонусное задание.** Подумайте, можно ли улучшить модели за счёт создания новых признаков: например, умножив длину ласт на ширину. Проверьте, усилится ли корреляция нового признака с целевой переменной, возрастёт ли благодаря ему качество модели.
# 3. Сформируйте итоговую таблицу с результатами моделей. Это удобно сделать в виде датафрейма pandas. Включите в таблицу следующие столбцы:
#    - Название модели.
#    - Название датасета — оно должно указывать на то, какой способ масштабирования использовался при подготовке данных.
#    - Метрики качества, рассчитанные на валидационной выборке. Основная метрика — MAE, дополнительные — MSE, R², MAPE и прочие.

# In[16]:


# Удаляем записи с weight=0 из всех выборок
print("ДО удаления:")
print(f"Train size: {len(y_train)}")
print(f"Val size: {len(y_val)}")
print(f"Test size: {len(y_test)}")

# Находим индексы с weight=0 в каждой выборке
zero_train = y_train[y_train == 0].index
zero_val = y_val[y_val == 0].index if len(y_val[y_val == 0]) > 0 else []
zero_test = y_test[y_test == 0].index if len(y_test[y_test == 0]) > 0 else []

print(f"\nНайдено записей с weight=0:")
print(f"Train: {len(zero_train)}")
print(f"Val: {len(zero_val)}")
print(f"Test: {len(zero_test)}")

# Удаляем из train
if len(zero_train) > 0:
    X_train_clean = X_train_filled.drop(index=zero_train)
    y_train_clean = y_train.drop(index=zero_train)
else:
    X_train_clean = X_train_filled
    y_train_clean = y_train

# Удаляем из val
if len(zero_val) > 0:
    X_val_clean = X_val_filled.drop(index=zero_val)
    y_val_clean = y_val.drop(index=zero_val)
else:
    X_val_clean = X_val_filled
    y_val_clean = y_val

# Удаляем из test
if len(zero_test) > 0:
    X_test_clean = X_test_filled.drop(index=zero_test)
    y_test_clean = y_test.drop(index=zero_test)
else:
    X_test_clean = X_test_filled
    y_test_clean = y_test

print("\nПОСЛЕ удаления:")
print(f"Train size: {len(y_train_clean)} (удалено {len(zero_train)})")
print(f"Val size: {len(y_val_clean)} (удалено {len(zero_val)})")
print(f"Test size: {len(y_test_clean)} (удалено {len(zero_test)})")

# Обновляем датасеты
for name in datasets:
    if name != 'enhanced':
        if len(zero_train) > 0:
            datasets[name]['X_train'] = datasets[name]['X_train'].drop(index=zero_train)
            datasets[name]['y_train'] = datasets[name]['y_train'].drop(index=zero_train)
        if len(zero_val) > 0:
            datasets[name]['X_val'] = datasets[name]['X_val'].drop(index=zero_val)
            datasets[name]['y_val'] = datasets[name]['y_val'].drop(index=zero_val)
        if len(zero_test) > 0:
            datasets[name]['X_test'] = datasets[name]['X_test'].drop(index=zero_test)
            datasets[name]['y_test'] = datasets[name]['y_test'].drop(index=zero_test)

print("\nОшибочные записи удалены. Можно продолжать.")

# Проверим новую статистику
print(f"\nНовая статистика weight (train):")
print(f"Мин: {y_train_clean.min():.2f}")
print(f"Макс: {y_train_clean.max():.2f}")
print(f"Среднее: {y_train_clean.mean():.2f}")


# In[17]:


# Базовая модель (предсказание среднего)
dummy = DummyRegressor(strategy='mean')
dummy.fit(datasets['raw']['X_train'], datasets['raw']['y_train'])

# Предсказания на валидационной выборке
y_val_pred_dummy = dummy.predict(datasets['raw']['X_val'])

# Используем нашу robust функцию для метрик
def calculate_metrics_robust(y_true, y_pred, dataset_name="", model_name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'MAPE': round(mape, 2),
        'R2': round(r2, 4)
    }

dummy_metrics = calculate_metrics_robust(
    datasets['raw']['y_val'], y_val_pred_dummy, 
    dataset_name='raw', 
    model_name='Dummy (mean)'
)

print("Базовая модель (предсказание среднего):")
print(f"MAE: {dummy_metrics['MAE']:.3f}")
print(f"RMSE: {dummy_metrics['RMSE']:.3f}")
print(f"MAPE: {dummy_metrics['MAPE']:.2f}%")
print(f"R2: {dummy_metrics['R2']:.4f}")

# Обновляем список результатов
results = [dummy_metrics]


# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor

def train_and_evaluate(model, model_name, dataset_version, datasets, results_list):
    """
    Обучает модель на всех версиях датасета и добавляет результаты в список
    """
    data = datasets[dataset_version]
    X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказание на валидации
    y_val_pred = model.predict(X_val)
    
    # Расчет метрик
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    r2 = r2_score(y_val, y_val_pred)
    
    metrics = {
        'model': model_name,
        'dataset': dataset_version,
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'MAPE': round(mape, 2),
        'R2': round(r2, 4)
    }
    
    results_list.append(metrics)
    print(f"  {model_name} на {dataset_version}: MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.4f}")
    
    return model

print("Функция train_and_evaluate готова к использованию")


# In[ ]:


print("Linear Regression:")
print("-" * 50)

# Linear Regression на raw данных
lr_raw = train_and_evaluate(LinearRegression(), 'LinearRegression', 'raw', datasets, results)

# Linear Regression на стандартизованных данных
lr_std = train_and_evaluate(LinearRegression(), 'LinearRegression', 'standardized', datasets, results)

# Linear Regression на нормализованных данных
lr_norm = train_and_evaluate(LinearRegression(), 'LinearRegression', 'normalized', datasets, results)


# ## Сравнение моделей на валидационной выборке

# In[ ]:





# 1. Сравните построенные модели по метрикам на валидационной выборке. Удалось ли существенно улучшить результат базовой модели?
# 2. Выберите лучшую модель по основной метрике на валидационной выборке. Не заглядывайте в метрики на тестовой выборке раньше времени. Тестовая выборка не используется для обучения моделей, подбора гиперпараметров и сравнения моделей с разными значениями.
# 3. Напишите выводы о том, какая из моделей обладает лучшим качеством. Именно её одну далее нужно проверить на тестовой выборке для итоговой оценки.

# In[ ]:





# ## Проверка лучшей модели на тестовой выборке

# 1. Проверьте метрики лучшей модели на тестовой выборке.
# 2. Узнайте, есть ли признаки переобучения лучшей модели.
# 3. Определите, соответствует ли модель требованиям заказчика. Объясните, можно ли её рекомендовать к внедрению.

# In[ ]:





# ## Оценка важности признаков

# 1. Оцените важность признаков по абсолютным значениям весов лучшей модели.
# 2. Напишите, какие признаки стали для модели более важными. Объясните, согласны ли вы с таким результатом?

# In[ ]:





# ## Функция для прогнозирования веса черепахи

# * Напишите на Python функцию, которая будет прогнозировать массу черепахи по заданным параметрам с учётом коэффициентов лучшей модели (свойство `coef_`) и смещения (свойство `intercept_`).
# * Если вы столкнётесь с трудностями при написании функции, то представьте, что обращаетесь к старшему коллеге с просьбой помочь, и составьте задание для её написания. Подробно опишите логику, по которой рассчитывается масса черепахи, и укажите, как именно должны происходить расчёты.

# In[ ]:





# ## Общие выводы и рекомендации по дальнейшей работе

# Напишите общие выводы и рекомендации по дальнейшей работе. Ответьте на вопросы:
#   - Какие модели изучены?
#   - Какие результаты получены?
#   - Рекомендуется ли итоговая модель к внедрению?
#   - Какая архитектура и способ обработки признаков показали себя лучше всего? Какие у них показатели метрик?
#   - Какие признаки наиболее важны для модели?
#   - Есть ли перспективы у обучения этой или других моделей для предсказания массы других видов черепах?
#   - При наличии добавьте сюда свои предложения по дальнейшему развитию проекта.

# 
