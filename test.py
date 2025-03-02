import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/anastasiavaznikova/Desktop/University/Kursova_code/Dataset of Diabetes .csv')


df["CLASS"] = df["CLASS"].str.strip()

label_mapping = {'N': 0, 'P': 1, 'Y': 1}
df["CLASS"] = df["CLASS"].map(label_mapping)

unique_classes = df["CLASS"].unique()
class_counts = df["CLASS"].value_counts()

output_path = '/Users/anastasiavaznikova/Desktop/University/Kursova_code/binary_class_dataset.csv'
df.to_csv(output_path, index=False)

unique_classes, class_counts, output_path
