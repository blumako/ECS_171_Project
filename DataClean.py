import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/data.csv')
target_diseases = ['panic disorder', 'flu', 'heart attack', 'migraine', 'head injury']
filtered_df = df[df['diseases'].isin(target_diseases)].copy()

symptom_cols = filtered_df.columns.drop('diseases')
cols_to_keep = symptom_cols[filtered_df[symptom_cols].sum() > 0]
filtered_df = filtered_df[['diseases'] + list(cols_to_keep)]

print(f"Columns before: 377")
print(f"Columns after: {len(cols_to_keep)}")
print(filtered_df.shape)
filtered_df.to_csv('./dataset/filtered_data.csv', index=False)

proportions = (filtered_df['diseases'].value_counts(normalize=True))
proportions.to_csv('./dataset/dataproportions.csv', index=True)

print(filtered_df.isnull().sum().sum())

filtered_df['diseases'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.savefig('class_dist.png')

symptom_cols = filtered_df.columns[1:]
symptom_means = filtered_df.groupby('diseases')[symptom_cols].mean()

top_symptoms = symptom_means.var().nlargest(20).index
sns.heatmap(symptom_means[top_symptoms], cmap='Blues')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('symptom_heatmap.png')