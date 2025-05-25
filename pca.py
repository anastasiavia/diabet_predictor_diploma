import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


file_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/balanced1_dataset.csv"
df = pd.read_csv(file_path)


features = df.drop(columns=["CLASS"])

print("Порядок ознак для scaler + PCA:")
print(features.columns.tolist())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
df_pca["CLASS"] = df["CLASS"]  
df_pca.to_csv("/Users/anastasiavaznikova/Desktop/University/Kursova_code/PCA_Balanced_Dataset_of_Diabetes.csv", index=False)

# Побудова кореляційної матриці після PCA
correlation_matrix_pca = df_pca.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_pca, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Кореляційна матриця після PCA")
plt.show()

joblib.dump(scaler, "/Users/anastasiavaznikova/Desktop/University/Kursova_code/scaler_pca.pkl")
joblib.dump(pca, "/Users/anastasiavaznikova/Desktop/University/Kursova_code/pca.pkl")
