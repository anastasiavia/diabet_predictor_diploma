# Завантаження необхідних бібліотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Завантаження датасету
file_path = '/Users/anastasiavaznikova/Desktop/University/Kursova_code/binary_class_dataset.csv'
df = pd.read_csv(file_path)


if df["CLASS"].dtype == 'object':
    df["CLASS"] = df["CLASS"].str.strip()


print("Унікальні значення у колонці CLASS:")
print(df['CLASS'].value_counts())


if 'Gender' in df.columns:
    label_encoder = LabelEncoder()
    df["Gender"] = label_encoder.fit_transform(df["Gender"])


for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])


X = df.drop(columns=['CLASS'], errors='ignore')  
y = df['CLASS'].astype(int)  


plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="Set2")
plt.title("Розподіл класів до SMOTE")
plt.xlabel("Клас (Діабет)")
plt.ylabel("Кількість записів")
plt.show()


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

df_resampled = pd.DataFrame(X_res, columns=X.columns)
df_resampled['CLASS'] = y_res


output_file_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/balanced1_dataset.csv"
df_resampled.to_csv(output_file_path, index=False)



plt.figure(figsize=(6, 4))
sns.countplot(x=y_res, palette="Set2")
plt.title("Розподіл класів після SMOTE")
plt.xlabel("Клас (Діабет)")
plt.ylabel("Кількість записів")
plt.show()


plt.figure(figsize=(12, 10))
correlation_matrix = df_resampled.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Кореляційна матриця після SMOTE')
plt.show()
