import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#drop nontarget disease rows
df = pd.read_csv('./dataset/data.csv')
target_diseases = ['panic disorder', 'flu', 'heart attack', 'migraine', 'head injury']
filtered_df = df[df['diseases'].isin(target_diseases)].copy()

symptom_cols = filtered_df.columns.drop('diseases')

cols_to_keep = symptom_cols[filtered_df[symptom_cols].sum() > 0] #checks for all 0 columns by summing
filtered_df = filtered_df[['diseases'] + list(cols_to_keep)]
filtered_df = filtered_df.drop_duplicates()

#number of columns/before after data cleaning (getting rid of 0 columns and duplicates)
print(f"Columns before: 377")
print(f"Columns after: {len(cols_to_keep)}")

#shape of filtered data
print(filtered_df.shape)
filtered_df.to_csv('./dataset/filtered_data.csv', index=False)

#missing values check
print("Missing values:", filtered_df.isnull().sum().sum())

proportions = (filtered_df['diseases'].value_counts(normalize=True))
proportions.to_csv('./dataset/dataproportions.csv', index=True)

plt.figure(figsize=(8, 5))
filtered_df['diseases'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Class Distribution Across Target Diseases')
plt.xlabel('Disease')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('class_dist.png')

#top 20 most prevalent symptoms for all diseases
symptom_cols = filtered_df.columns[1:]
symptom_means = filtered_df.groupby('diseases')[symptom_cols].mean()
plt.figure(figsize=(16, 6))
top_symptoms = symptom_means.var().nlargest(20).index
sns.heatmap(symptom_means[top_symptoms], cmap='Blues')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('symptom_heatmap.png')


""""#correlation heatmap b/w diseases and top symptoms
top_symptom_data = filtered_df[list(top_symptoms)]
plt.figure(figsize=(12, 10))
sns.heatmap(top_symptom_data.corr(), cmap='coolwarm', center=0)
plt.title('Symptom Correlation Matrix')
plt.tight_layout()
plt.savefig('symptom_correlation.png')"""

#top 5 symptoms for each disease, each decimal is for example for panic disorder
#depressive_or_psychotic_symptoms    0.682068 means 68.2% of panic disorder patients have that symptom
for disease in target_diseases:
    disease_df = filtered_df[filtered_df['diseases'] == disease]
    top = disease_df[symptom_cols].mean().nlargest(5)
    print(f"\n{disease.upper()}")
    print(top)
