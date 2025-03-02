import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score

unbalanced_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/Dataset of Diabetes .csv"
balanced_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/PCA_Balanced_Dataset_of_Diabetes.csv"
model_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/knn_model.pkl"
scaler_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/scaler.pkl"

df_unbalanced = pd.read_csv(unbalanced_path)
df_balanced = pd.read_csv(balanced_path)

if 'Gender' in df_unbalanced.columns:
    df_unbalanced['Gender'] = df_unbalanced['Gender'].map({'M': 0, 'F': 1})

if df_unbalanced['CLASS'].dtype == 'object':
    df_unbalanced['CLASS'] = df_unbalanced['CLASS'].astype('category').cat.codes

df_unbalanced.fillna(df_unbalanced.mean(), inplace=True)
df_balanced.fillna(df_balanced.mean(), inplace=True)

X_balanced = df_balanced.iloc[:, :-1]
y_balanced = df_balanced.iloc[:, -1]

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler_bal = StandardScaler()
X_train_bal = scaler_bal.fit_transform(X_train_bal)
X_test_bal = scaler_bal.transform(X_test_bal)

# Save the scaler
joblib.dump(scaler_bal, scaler_path)

# KNN класифікатор
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_bal, y_train_bal)


joblib.dump(knn, model_path)

# Оцінка на збалансованому датасеті
y_pred_bal = knn.predict(X_test_bal)
balanced_acc_bal = balanced_accuracy_score(y_test_bal, y_pred_bal)
accuracy_bal = accuracy_score(y_test_bal, y_pred_bal)
precision_bal = precision_score(y_test_bal, y_pred_bal, average='weighted')
recall_bal = recall_score(y_test_bal, y_pred_bal, average='weighted')
f1_bal = f1_score(y_test_bal, y_pred_bal, average='weighted')
conf_matrix_bal = confusion_matrix(y_test_bal, y_pred_bal)

print(f'Balanced Accuracy (Balanced): {balanced_acc_bal:.4f}')
print(f'Accuracy (Balanced): {accuracy_bal:.4f}')
print(f'Precision (Balanced): {precision_bal:.4f}')
print(f'Recall (Balanced): {recall_bal:.4f}')
print(f'F1 Score (Balanced): {f1_bal:.4f}')
print('\nConfusion Matrix (Balanced):\n', conf_matrix_bal)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_bal, annot=True, cmap='Blues', fmt='d', xticklabels=['Non-diabetic', 'Diabetic'], yticklabels=['Non-diabetic', 'Diabetic'])
plt.title('Confusion Matrix (Balanced Dataset)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
