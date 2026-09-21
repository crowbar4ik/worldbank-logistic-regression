import os
import pandas as pd
import pycountry

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. ЗЧИТУВАННЯ ДАНИХ
# ---------------------------------------------------------

# os.path.abspath(__file__) отримує абсолютний шлях до поточного Python-файлу.
# os.path.dirname(...) залишає лише папку, у якій знаходиться цей файл.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# os.path.join(...) безпечно формує шлях до файлу worldbank.xlsx
# незалежно від того, який роздільник шляхів використовує операційна система.
FILE_PATH = os.path.join(
    BASE_DIR,
    "worldbank.xlsx"
)

# pd.read_excel(...) читає Excel-файл і повертає pandas DataFrame.
# sheet_name='Data' вказує, з якого аркуша Excel потрібно зчитати дані.
df = pd.read_excel(
    FILE_PATH,
    sheet_name="Data"
)

print("\n=== RAW DATA ===")
print("Shape:", df.shape)
print(df.head())


# ---------------------------------------------------------
# 2. ЗАЛИШАЄМО ЛИШЕ ПОТРІБНІ ПОКАЗНИКИ
# ---------------------------------------------------------

indicators = [
    "IT.NET.USER.ZS",       # Користувачі Інтернету, %
    "NY.GDP.PCAP.PP.KD",    # ВВП на душу населення за ПКС
    "SP.URB.TOTL.IN.ZS",    # Міське населення, %
    "EG.ELC.ACCS.ZS"        # Доступ до електроенергії, %
]

df = df[
    # .isin(indicators) для кожного рядка перевіряє, чи входить Series Code
    # до списку потрібних нам показників і повертає True / False.
    df["Series Code"].isin(indicators)
# .copy() створює незалежну копію відібраних даних,
# щоб подальші зміни не впливали на початковий DataFrame.
].copy()

print("\n=== AFTER SELECTING INDICATORS ===")
print("Shape:", df.shape)


# ---------------------------------------------------------
# 3. ПЕРЕТВОРЮЄМО ЗНАЧЕННЯ НА ЧИСЛА
#
# ".." та інші нечислові значення перетворюються на NaN
# ---------------------------------------------------------

# pd.to_numeric(...) намагається перетворити значення стовпця на числа.
# errors='coerce' замінює значення, які не можна перетворити, на NaN.
df["2021 [YR2021]"] = pd.to_numeric(
    df["2021 [YR2021]"],
    errors="coerce"
)


print("\n=== MISSING VALUES BY INDICATOR ===")

missing_by_indicator = (
    # .groupby('Series Code') групує рядки за кодом показника.
    # Після цього статистику можна рахувати окремо для кожного показника.
    df.groupby("Series Code")["2021 [YR2021]"]
      # .agg(...) дозволяє порахувати кілька агрегованих показників одночасно:
      # count() рахує доступні значення, а isna().sum() — кількість пропусків.
      .agg(
          available="count",
          missing=lambda x: x.isna().sum()
      )
)

print(missing_by_indicator)


# ---------------------------------------------------------
# 4. ПЕРЕТВОРЕННЯ PIVOT
#
# ДО:
#
# Україна | Інтернет
# Україна | ВВП
# Україна | Міське населення
# Україна | Електроенергія
#
# ПІСЛЯ:
#
# Україна | Інтернет | ВВП | Міське населення | Електроенергія
# ---------------------------------------------------------

# .pivot(...) перетворює дані з 'довгого' формату у 'широкий'.
# Після pivot кожна країна стає одним рядком, а показники — окремими стовпцями.
df = df.pivot(
    index=[
        "Country Name",
        "Country Code"
    ],
    columns="Series Code",
    values="2021 [YR2021]"
# .reset_index() повертає Country Name та Country Code зі складу індексу
# назад у звичайні стовпці DataFrame.
).reset_index()


# .rename(columns={...}) перейменовує технічні коди World Bank
# у короткі та зрозумілі назви стовпців.
df = df.rename(
    columns={
        "IT.NET.USER.ZS":
            "internet",

        "NY.GDP.PCAP.PP.KD":
            "gdp_per_capita",

        "SP.URB.TOTL.IN.ZS":
            "urban_population",

        "EG.ELC.ACCS.ZS":
            "electricity_access"
    }
)


print("\n=== AFTER PIVOT ===")
print("Entities:", len(df))

print(
    df[
        [
            "Country Name",
            "Country Code",
            "internet",
            "gdp_per_capita",
            "urban_population",
            "electricity_access"
        ]
    ].head()
)


# ---------------------------------------------------------
# 5. ВИДАЛЯЄМО АГРЕГОВАНІ ПОКАЗНИКИ WORLD BANK
#
# Видаляємо:
# Світ
# Європейський Союз
# Країни з високим рівнем доходу
# Країни — члени OECD
# регіони
# тощо
# ---------------------------------------------------------

# Власна функція is_country(code) перевіряє,
# чи відповідає код реальній країні/території, а не агрегату World Bank.
def is_country(code):

    # Косово у World Bank позначається кодом XKX
    if code == "XK":
        return True

    if code == "XKX":
        return True

    return (
        # pycountry.countries.get(alpha_3=code) шукає країну за ISO alpha-3 кодом.
        # Якщо країну знайдено — повертається об'єкт країни, інакше None.
        pycountry.countries.get(
            alpha_3=code
        )
        is not None
    )


df["is_country"] = (
    df["Country Code"]
    # .apply(is_country) викликає нашу функцію is_country
    # для кожного значення у стовпці Country Code.
    .apply(is_country)
)


print("\n=== COUNTRY VS AGGREGATES ===")
print(
    df["is_country"]
    # .value_counts() рахує, скільки разів зустрічається кожне значення.
    .value_counts()
)


df = df[
    df["is_country"]
# .copy() створює незалежну копію відібраних даних,
# щоб подальші зміни не впливали на початковий DataFrame.
].copy()


print(
    "\nReal countries / territories:",
    len(df)
)


# ---------------------------------------------------------
# 6. ПЕРЕВІРЯЄМО ПРОПУЩЕНІ ЗНАЧЕННЯ
# ---------------------------------------------------------

features = [
    "internet",
    "gdp_per_capita",
    "urban_population",
    "electricity_access"
]


print(
    "\n=== MISSING VALUES FOR REAL COUNTRIES ==="
)

print(
    # df[features] вибирає лише потрібні числові стовпці зі списку features.
    df[features]
    # .isna() позначає пропущені значення як True.
    .isna()
    # .sum() для булевих значень рахує кількість True,
    # тобто кількість пропусків у кожному стовпці.
    .sum()
)


# ---------------------------------------------------------
# 7. ВИДАЛЯЄМО РЯДКИ З ПРОПУЩЕНИМИ ЗНАЧЕННЯМИ
# ---------------------------------------------------------

before = len(df)


# .dropna(subset=features) видаляє рядки,
# у яких відсутнє хоча б одне значення з потрібних ознак.
df = df.dropna(
    subset=features
).copy()


after = len(df)


print("\n=== CLEANING ===")

print(
    "Before dropna:",
    before
)

print(
    "After dropna:",
    after
)

print(
    "Removed:",
    before - after
)


# ---------------------------------------------------------
# 8. СТВОРЮЄМО ЦІЛЬОВУ ЗМІННУ
#
# high_internet = 1:
# Інтернетом користуються щонайменше 70%
#
# high_internet = 0:
# менше 70%
# ---------------------------------------------------------

df["high_internet"] = (
    df["internet"] >= 70
# .astype(int) перетворює True / False у 1 / 0.
).astype(int)


print(
    "\n=== TARGET DISTRIBUTION ==="
)

print(
    df["high_internet"]
    # .value_counts() рахує, скільки разів зустрічається кожне значення.
    .value_counts()
)


print("\nPercentage:")

print(
    df["high_internet"]
      # normalize=True повертає частки класів замість абсолютної кількості.
      .value_counts(normalize=True)
      # .mul(100) переводить частки у відсотки.
      .mul(100)
      # .round(2) округлює числові значення до двох знаків після коми.
      .round(2)
)


# ---------------------------------------------------------
# 9. ПОКАЗУЄМО ОЧИЩЕНІ ДАНІ
# ---------------------------------------------------------

print(
    "\n=== CLEAN DATA ==="
)

print(
    df[
        [
            "Country Name",
            "Country Code",
            "internet",
            "gdp_per_capita",
            "urban_population",
            "electricity_access",
            "high_internet"
        ]
    ]
    .head(10)
    # .round(2) округлює числові значення до двох знаків після коми.
    .round(2)
)


# ---------------------------------------------------------
# 10. ВИБИРАЄМО ОЗНАКИ X ТА ЦІЛЬОВУ ЗМІННУ y
# ---------------------------------------------------------

feature_names = [
    "gdp_per_capita",
    "urban_population",
    "electricity_access"
]


# X — матриця ознак, тобто дані, на основі яких модель буде робити прогноз.
X = df[
    feature_names
# .copy() створює незалежну копію відібраних даних,
# щоб подальші зміни не впливали на початковий DataFrame.
].copy()


# y — цільова змінна, правильна відповідь, яку модель вчиться передбачати.
y = df[
    "high_internet"
# .copy() створює незалежну копію відібраних даних,
# щоб подальші зміни не впливали на початковий DataFrame.
].copy()


# ---------------------------------------------------------
# 11. ПЕРЕВІРЯЄМО КОРЕЛЯЦІЮ МІЖ ОЗНАКАМИ
#
# Ми не хочемо, щоб предиктори майже дублювали
# один одного.
#
# Кореляція, близька до:
#
# 0    -> слабкий зв'язок
# +/-1 -> сильний зв'язок
#
# > 0.8 або < -0.8 може бути сигналом проблеми
# ---------------------------------------------------------

correlation_matrix = (
    # X.corr() обчислює попарну кореляцію між числовими ознаками.
    X.corr()
    .round(3)
)


print(
    "\n=== FEATURE CORRELATION MATRIX ==="
)

print(
    correlation_matrix
)


# Шукаємо сильні кореляції
strong_correlations = []

for i in range(
    len(feature_names)
):

    for j in range(
        i + 1,
        len(feature_names)
    ):

        feature_1 = (
            feature_names[i]
        )

        feature_2 = (
            feature_names[j]
        )

        correlation = (
            correlation_matrix.loc[
                feature_1,
                feature_2
            ]
        )

        # abs(...) бере модуль кореляції, щоб однаково враховувати
        # сильний позитивний і сильний негативний зв'язок.
        if abs(correlation) >= 0.8:

            strong_correlations.append(
                (
                    feature_1,
                    feature_2,
                    correlation
                )
            )


print(
    "\n=== STRONG CORRELATIONS >= 0.8 ==="
)

if strong_correlations:

    for (
        feature_1,
        feature_2,
        correlation
    ) in strong_correlations:

        print(
            feature_1,
            "<->",
            feature_2,
            ":",
            correlation
        )

else:

    print(
        "No strong correlations detected."
    )


# ---------------------------------------------------------
# 12. РОЗПОДІЛ НА TRAIN / TEST
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = (
    # train_test_split(...) ділить дані на навчальну та тестову частини.
    # test_size=0.20 означає 20% даних для тестування.
    # random_state=42 робить розподіл відтворюваним.
    # stratify=y зберігає приблизно однакову частку класів у train і test.
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    "\n=== TRAIN / TEST ==="
)

print(
    "Train observations:",
    len(X_train)
)

print(
    "Test observations:",
    len(X_test)
)


# ---------------------------------------------------------
# 13. СТАНДАРТИЗАЦІЯ / НОРМАЛІЗАЦІЯ
#
# Ознаки мають дуже різні масштаби:
#
# ВВП:
# 2,000 - 100,000+
#
# Міське населення:
# 10 - 100
#
# Доступ до електроенергії:
# 20 - 100
#
# StandardScaler виконує перетворення:
#
# x_scaled =
# (x - середнє) / стандартне відхилення
#
# ВАЖЛИВО:
# scaler.fit() виконується ЛИШЕ на train-даних.
# ---------------------------------------------------------

# StandardScaler() створює об'єкт для стандартизації ознак.
# Після стандартизації кожна ознака матиме приблизно середнє 0
# та стандартне відхилення 1.
scaler = StandardScaler()


X_train_scaled = (
    # fit_transform(X_train) спочатку обчислює параметри масштабування
    # лише на train-даних, а потім одразу масштабує train.
    scaler.fit_transform(
        X_train
    )
)


X_test_scaled = (
    # transform(X_test) масштабує test-дані за параметрами,
    # які були отримані ТІЛЬКИ з train-даних.
    scaler.transform(
        X_test
    )
)


print(
    "\n=== STANDARDIZATION ==="
)

print(
    "Mean values used by scaler:"
)

for (
    feature,
    mean
) in zip(
    feature_names,
    scaler.mean_
):

    print(
        feature,
        ":",
        round(mean, 2)
    )


print(
    "\nStandard deviations used by scaler:"
)

for (
    feature,
    scale
) in zip(
    feature_names,
    scaler.scale_
):

    print(
        feature,
        ":",
        round(scale, 2)
    )


# Необов'язково:
# показуємо перші рядки після масштабування

# pd.DataFrame(...) перетворює масив після StandardScaler
# назад у табличний формат з назвами стовпців.
scaled_example = pd.DataFrame(
    X_train_scaled,
    columns=feature_names
)


print(
    "\nFirst 5 observations after standardization:"
)

print(
    scaled_example
    .head()
    .round(3)
)


# ---------------------------------------------------------
# 14. ЛОГІСТИЧНА РЕГРЕСІЯ
# ---------------------------------------------------------

# LogisticRegression(...) створює модель логістичної регресії.
# max_iter=1000 дозволяє алгоритму зробити до 1000 ітерацій
# для пошуку оптимальних коефіцієнтів.
model = LogisticRegression(
    max_iter=1000
)


# Тут модель навчається
# model.fit(X_train_scaled, y_train) НАВЧАЄ модель.
# Під час fit модель підбирає коефіцієнти для ознак та intercept,
# щоб найкраще відрізняти клас 0 від класу 1 на train-даних.
model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# 15. ПОКАЗУЄМО КОЕФІЦІЄНТИ МОДЕЛІ
#
# Оскільки змінні були стандартизовані,
# коефіцієнти легше порівнювати.
# ---------------------------------------------------------

coefficients = pd.DataFrame(
    {
        "feature":
            feature_names,

        "coefficient":
            # model.coef_ містить навчені коефіцієнти при кожній ознаці.
            # Знак коефіцієнта показує напрямок зв'язку з імовірністю класу 1.
            model.coef_[0]
    }
)


coefficients["coefficient"] = (
    coefficients["coefficient"]
    .round(3)
)


print(
    "\n=== MODEL COEFFICIENTS ==="
)

print(
    coefficients
    .sort_values(
        "coefficient",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print(
    "\nIntercept:",
    round(
        # model.intercept_ — вільний член моделі, тобто базове зміщення
        # до врахування конкретних значень ознак.
        model.intercept_[0],
        3
    )
)


# ---------------------------------------------------------
# 16. ПРОГНОЗИ
# ---------------------------------------------------------

# model.predict(...) повертає готовий прогноз класу 0 або 1
# для кожного об'єкта з тестового набору.
y_pred = model.predict(
    X_test_scaled
)


# Ймовірність того, що:
# high_internet = 1

y_probability = (
    # model.predict_proba(...) повертає ймовірності належності до кожного класу.
    # [:, 1] вибирає ймовірність саме класу high_internet = 1.
    model.predict_proba(
        X_test_scaled
    )[:, 1]
)


# ---------------------------------------------------------
# 17. ЯКІСТЬ МОДЕЛІ
# ---------------------------------------------------------

# accuracy_score(y_test, y_pred) рахує частку правильних прогнозів.
accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n=== MODEL QUALITY ==="
)


print(
    "Accuracy:",
    round(
        accuracy,
        3
    )
)


print(
    "\nConfusion Matrix:"
)


print(
    # confusion_matrix(...) показує кількість TN, FP, FN та TP —
    # тобто де модель класифікувала правильно, а де помилилася.
    confusion_matrix(
        y_test,
        y_pred
    )
)


print(
    "\nClassification Report:"
)


print(
    # classification_report(...) виводить Precision, Recall, F1-score
    # та support окремо для кожного класу.
    classification_report(
        y_test,
        y_pred,
        digits=3
    )
)


# ---------------------------------------------------------
# 18. ПОКАЗУЄМО ТЕСТОВІ КРАЇНИ ТА ПРОГНОЗИ
# ---------------------------------------------------------

# df.loc[X_test.index, ...] вибирає саме ті країни,
# які потрапили до тестової вибірки.
results = df.loc[
    X_test.index,
    [
        "Country Name",
        "internet"
    ]
# .copy() створює незалежну копію відібраних даних,
# щоб подальші зміни не впливали на початковий DataFrame.
].copy()


results["actual"] = (
    y_test
)


results["predicted"] = (
    y_pred
)


results[
    "probability_high_internet"
] = (
    y_probability * 100
)


results[
    "probability_high_internet"
] = (
    results[
        "probability_high_internet"
    ]
    # .round(2) округлює числові значення до двох знаків після коми.
    .round(2)
)


# Відстань від нашого порогового значення 70%

results[
    "distance_from_70"
] = (
    results["internet"] - 70
).round(2)


results = (
    # .sort_values(...) сортує таблицю за ймовірністю класу 1.
    # ascending=False означає: від найбільшої ймовірності до найменшої.
    results.sort_values(
        "probability_high_internet",
        ascending=False
    )
)


print(
    "\n=== TEST RESULTS ==="
)


print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 19. ПОКАЗУЄМО ПОМИЛКИ МОДЕЛІ
# ---------------------------------------------------------

# Тут залишаємо лише рядки, де actual != predicted,
# тобто країни, на яких модель помилилася.
mistakes = results[
    results["actual"]
    !=
    results["predicted"]
# .copy() створює незалежну копію відібраних даних,
# щоб подальші зміни не впливали на початковий DataFrame.
].copy()


print(
    "\n=== MODEL MISTAKES ==="
)


print(
    mistakes[
        [
            "Country Name",
            "internet",
            "distance_from_70",
            "actual",
            "predicted",
            "probability_high_internet"
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 20. ПОКАЗУЄМО УКРАЇНУ
# ---------------------------------------------------------

# Фільтруємо DataFrame за кодом UKR, щоб окремо показати Україну.
ukraine = df[
    df["Country Code"] == "UKR"
]


print(
    "\n=== UKRAINE ==="
)


print(
    ukraine[
        [
            "Country Name",
            "internet",
            "gdp_per_capita",
            "urban_population",
            "electricity_access",
            "high_internet"
        ]
    ].round(2)
)
