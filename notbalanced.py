import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/Users/anastasiavaznikova/Desktop/University/Kursova_code/Dataset of Diabetes .csv'
df = pd.read_csv(file_path)


sns.set(style="whitegrid")

missing_values = df.isnull().sum()
print("Пропущені значення у датасеті:\n", missing_values)

df['CLASS'] = df['CLASS'].str.strip().str.upper()

class_distribution = df['CLASS'].value_counts()
print("\nРозподіл класів:\n", class_distribution)

# Перевірка збалансованості
class_balance_ratio = class_distribution / len(df)
print("\nЗбалансованість класів:\n", class_balance_ratio)


plt.figure(figsize=(8, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='pastel')
plt.title('Розподіл класів пацієнтів (CLASS)')
plt.xlabel('Клас')
plt.ylabel('Кількість записів')
plt.show()


plt.figure(figsize=(12, 10))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Кореляційна матриця між змінними')
plt.show()


