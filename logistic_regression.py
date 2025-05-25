import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score


unbalanced_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/Dataset of Diabetes .csv"
balanced_path = "/Users/anastasiavaznikova/Desktop/University/Kursova_code/PCA_Balanced_Dataset_of_Diabetes.csv"

df_unbalanced = pd.read_csv(unbalanced_path)
df_balanced = pd.read_csv(balanced_path)


if 'Gender' in df_unbalanced.columns:
    df_unbalanced['Gender'] = df_unbalanced['Gender'].map({'M': 0, 'F': 1})

if df_unbalanced['CLASS'].dtype == 'object':
    df_unbalanced['CLASS'] = df_unbalanced['CLASS'].astype('category').cat.codes

df_unbalanced.fillna(df_unbalanced.mean(), inplace=True)
df_balanced.fillna(df_balanced.mean(), inplace=True)


X_unbalanced = df_unbalanced.iloc[:, :-1]
y_unbalanced = df_unbalanced.iloc[:, -1]

X_balanced = df_balanced.iloc[:, :-1]
y_balanced = df_balanced.iloc[:, -1]


X_train_unb, X_test_unb, y_train_unb, y_test_unb = train_test_split(X_unbalanced, y_unbalanced, test_size=0.2, random_state=42)
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)


scaler_unb = StandardScaler()
X_train_unb = scaler_unb.fit_transform(X_train_unb)
X_test_unb = scaler_unb.transform(X_test_unb)

scaler_bal = StandardScaler()
X_train_bal = scaler_bal.fit_transform(X_train_bal)
X_test_bal = scaler_bal.transform(X_test_bal)


clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train_unb, y_train_unb)


y_pred_unb = clf.predict(X_test_unb)
balanced_acc_unb = balanced_accuracy_score(y_test_unb, y_pred_unb)


clf.fit(X_train_bal, y_train_bal)


y_pred_test_bal = clf.predict(X_test_bal)
balanced_acc_test = balanced_accuracy_score(y_test_bal, y_pred_test_bal)
accuracy_test = accuracy_score(y_test_bal, y_pred_test_bal)
precision_test = precision_score(y_test_bal, y_pred_test_bal, average='weighted')
recall_test = recall_score(y_test_bal, y_pred_test_bal, average='weighted')
f1_test = f1_score(y_test_bal, y_pred_test_bal, average='weighted')
conf_matrix_test = confusion_matrix(y_test_bal, y_pred_test_bal)


y_pred_train_bal = clf.predict(X_train_bal)
balanced_acc_train = balanced_accuracy_score(y_train_bal, y_pred_train_bal)
accuracy_train = accuracy_score(y_train_bal, y_pred_train_bal)
precision_train = precision_score(y_train_bal, y_pred_train_bal, average='weighted')
recall_train = recall_score(y_train_bal, y_pred_train_bal, average='weighted')
f1_train = f1_score(y_train_bal, y_pred_train_bal, average='weighted')
conf_matrix_train = confusion_matrix(y_train_bal, y_pred_train_bal)


print(f'Balanced Accuracy (Unbalanced dataset): {balanced_acc_unb:.4f}\n')

print('=== Тестова вибірка ===')
print(f'Balanced Accuracy: {balanced_acc_test:.4f}')
print(f'Accuracy: {accuracy_test:.4f}')
print(f'Precision: {precision_test:.4f}')
print(f'Recall: {recall_test:.4f}')
print(f'F1 Score: {f1_test:.4f}')
print('Confusion Matrix:\n', conf_matrix_test)

print('\n=== Навчальна вибірка ===')
print(f'Balanced Accuracy: {balanced_acc_train:.4f}')
print(f'Accuracy: {accuracy_train:.4f}')
print(f'Precision: {precision_train:.4f}')
print(f'Recall: {recall_train:.4f}')
print(f'F1 Score: {f1_train:.4f}')
print('Confusion Matrix:\n', conf_matrix_train)


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_test, annot=True, cmap='Blues', fmt='d',
            xticklabels=['Non-diabetic', 'Diabetic'],
            yticklabels=['Non-diabetic', 'Diabetic'])
plt.title('Confusion Matrix - Test Set (Balanced)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_train, annot=True, cmap='Greens', fmt='d',
            xticklabels=['Non-diabetic', 'Diabetic'],
            yticklabels=['Non-diabetic', 'Diabetic'])
plt.title('Confusion Matrix - Train Set (Balanced)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()
