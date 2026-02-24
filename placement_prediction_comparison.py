import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('archive/collegePlace.csv')
df.dropna(inplace=True)

# ─────────────────────────────────────────────────────────────
# 2. ENCODE TEXT COLUMNS
# ─────────────────────────────────────────────────────────────
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Stream'] = df['Stream'].map({
    'Electronics And Communication': 1,
    'Computer Science'             : 2,
    'Information Technology'       : 3,
    'Mechanical'                   : 4,
    'Electrical'                   : 5,
    'Civil'                        : 6
})
df.dropna(inplace=True)

# ─────────────────────────────────────────────────────────────
# 3. FEATURE & TARGET SPLIT
# ─────────────────────────────────────────────────────────────
FEATURES = ['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'HistoryOfBacklogs']
X = df[FEATURES]
y = df['PlacedOrNot']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────
# 4. DEFINE ALL MODELS
# ─────────────────────────────────────────────────────────────
models = {
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost"             : XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM"                 : SVC(probability=True, random_state=42),
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "KNN"                 : KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ─────────────────────────────────────────────────────────────
# 5. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────────────────────
accuracies     = {}
trained_models = {}

print("\n" + "="*60)
print("           MODEL COMPARISON RESULTS")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec  = recall_score(y_test, y_pred) * 100
    f1   = f1_score(y_test, y_pred) * 100

    accuracies[name]     = round(acc, 2)
    trained_models[name] = model

    print(f"\n  {name}")
    print(f"    Accuracy  : {acc:.2f}%")
    print(f"    Precision : {prec:.2f}%")
    print(f"    Recall    : {rec:.2f}%")
    print(f"    F1 Score  : {f1:.2f}%")

print("\n" + "="*60)

# ─────────────────────────────────────────────────────────────
# 6. GRAPH 1 — ACCURACY COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────
colors = ['#2ecc71','#3498db','#e74c3c','#9b59b6','#f39c12','#1abc9c','#e67e22']

plt.figure(figsize=(12, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors, edgecolor='black', width=0.6)

for bar, val in zip(bars, accuracies.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Models', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.ylim(0, 115)
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png', dpi=150)
plt.show()
print("Saved: model_accuracy_comparison.png")

# ─────────────────────────────────────────────────────────────
# 7. GRAPH 2 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
# Use Random Forest for feature importance (tree-based, most reliable)
rf_model      = trained_models["Random Forest"]
importances   = rf_model.feature_importances_
indices       = np.argsort(importances)[::-1]
sorted_features    = [FEATURES[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features[::-1], sorted_importances[::-1],
                color=['#2ecc71','#3498db','#e74c3c','#9b59b6','#f39c12','#1abc9c','#e67e22'],
                edgecolor='black')

for bar, val in zip(bars, sorted_importances[::-1]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

plt.title('Feature Importance — Random Forest', fontsize=15, fontweight='bold')
plt.xlabel('Importance Score', fontsize=13)
plt.ylabel('Features', fontsize=13)
plt.xlim(0, max(sorted_importances) + 0.05)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Saved: feature_importance.png")

# ─────────────────────────────────────────────────────────────
# 8. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────
best_model_name = max(accuracies, key=accuracies.get)
best_model      = trained_models[best_model_name]
pickle.dump(best_model, open('model.pkl', 'wb'))

print(f"\nBest Model : {best_model_name} ({accuracies[best_model_name]}% Accuracy)")
print("Best model saved as 'model.pkl'")

# ─────────────────────────────────────────────────────────────
# 9. USER PREDICTION
# ─────────────────────────────────────────────────────────────
print('\n' + '*'*50 + '\n')

age = int(input("Enter Your Age (min 19 - max 31) : "))
age = max(19, min(31, age))

print("Gender :- Male : 1 , Female : 0")
gender = int(input("Enter Your Gender : "))
gender = min(gender, 1)

print("""Stream :-
  Electronics And Communication : 1
  Computer Science              : 2
  Information Technology        : 3
  Mechanical                    : 4
  Electrical                    : 5
  Civil                         : 6""")
stream = int(input("Enter Your Stream : "))
if stream == 0 or stream > 6:
    stream = 6

internship = int(input("Enter No. Of Internships Done (min 0 - max 3) : "))
internship = max(0, min(3, internship))

cgpa = float(input("Enter Your CGPA : "))
cgpa = int(cgpa)
cgpa = min(cgpa, 10)

print("Hostel :- Live In Hostel : 1 , Not Live in Hostel : 0")
hostel = int(input("Do You Live In Hostel : "))
hostel = min(hostel, 1)

print("History of Backlogs :- Yes : 1 , No : 0")
hbl = int(input("Do You Have Any History Of Backlogs : "))
hbl = min(hbl, 1)

print('\n' + '*'*50 + '\n')

data  = [[age, gender, stream, internship, cgpa, hostel, hbl]]
pred  = best_model.predict(data)
proba = best_model.predict_proba(data)[0][1] * 100

if pred:
    print("You Have High Chances Of Getting Placed")
else:
    print("You Have Low Chances Of Getting Placed")

print(f"Your Chances Of Getting Placed : {proba:.2f}%")
print(f"Prediction made using          : {best_model_name}")
print('\n' + '*'*50 + '\n')
