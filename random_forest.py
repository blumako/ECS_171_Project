import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 0
TEST_SIZE = 0.20

df = pd.read_csv("../dataset/filtered_data.csv")

df = df.drop_duplicates()

print(f"Shape: {df.shape}\n")
print(f"Class Counts:\n{df['diseases'].value_counts()}\n")
print(f"NaNs: {df.isna().sum().sum()}\n")

X = df.drop(columns=["diseases"]).values

label_to_int = {
    "flu": 0, 
    "head injury": 1, 
    "heart attack": 2, 
    "migraine": 3, 
    "panic disorder": 4
}
int_to_label = {i: label for label, i in label_to_int.items()}
y = df["diseases"].map(label_to_int).values
class_names = [int_to_label[i] for i in range(len(label_to_int))]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}\n")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1 
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test ,y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1:       {f1_score(y_test, y_pred, average='macro'):.4f}")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
print("Saved rf_confusion_matrix.png")

