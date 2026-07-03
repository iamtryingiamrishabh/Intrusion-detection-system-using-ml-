import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# nsl-kdd
print("----- NSL-KDD MODEL -----")

df = pd.read_csv("kddtest+.txt", sep=",", header=None)

# Class Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=df[41])
plt.title("NSL-KDD Class Distribution")
plt.xticks(rotation=90)
plt.show()

# Features & Label
X = df.iloc[:, 0:41]
y = df.iloc[:, 41]

# Encoding
X = pd.get_dummies(X)
X.columns = X.columns.astype(str)
 
# Scaling
scaler1 = StandardScaler()
X_scaled = scaler1.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# Predictions
y_pred = model_dt.predict(X_test)

train_acc_dt = accuracy_score(y_train, model_dt.predict(X_train))
test_acc_dt = accuracy_score(y_test, y_pred)

print(f"NSL-KDD Train Accuracy: {train_acc_dt:.4f}")
print(f"NSL-KDD Test Accuracy: {test_acc_dt:.4f}")

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm_kdd = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_kdd, cmap='Blues')
plt.title("NSL-KDD Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feat_imp_kdd = pd.Series(model_dt.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
feat_imp_kdd.nlargest(15).plot(kind='barh')
plt.title("Top 15 NSL-KDD Features")
plt.show()

# cicids
print("\n----- CICIDS MODEL -----")

df2 = pd.read_csv("cicids.csv")

# Clean column names
df2.columns = df2.columns.str.strip()

# Fix invalid values
df2.replace([float('inf'), -float('inf')], 0, inplace=True)
df2 = df2.dropna()

# Features & Label
y2 = df2["Label"]
X2 = df2.drop("Label", axis=1)

# Encoding
X2 = pd.get_dummies(X2)

# Scaling
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# Split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_scaled, y2, test_size=0.2, random_state=42
)

# Model
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_rf.fit(X2_train, y2_train)

# Predictions
pred_cic = model_rf.predict(X2_test)

train_acc_rf = accuracy_score(y2_train, model_rf.predict(X2_train))
test_acc_rf = accuracy_score(y2_test, pred_cic)

print(f"CICIDS Train Accuracy: {train_acc_rf:.4f}")
print(f"CICIDS Test Accuracy: {test_acc_rf:.4f}")

# Confusion Matrix
plt.figure(figsize=(8,6))
cm_cic = confusion_matrix(y2_test, pred_cic)
sns.heatmap(cm_cic, cmap='Blues')
plt.title("CICIDS Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature
feat_imp_cic = pd.Series(model_rf.feature_importances_, index=X2.columns)
plt.figure(figsize=(10,6))
feat_imp_cic.nlargest(15).plot(kind='barh')
plt.title("Top 15 CICIDS Features")
plt.show()

# graph_compare
plt.figure(figsize=(7,5))

labels = ["KDD Train", "KDD Test", "CIC Train", "CIC Test"]
values = [train_acc_dt, test_acc_dt, train_acc_rf, test_acc_rf]

bars = plt.bar(labels, values)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}",
             ha='center', va='bottom')

plt.title("Overall Model Performance")
plt.ylim(0,1)
plt.show()

# reports
print("\nNSL-KDD Classification Report:\n", classification_report(y_test, y_pred))
print("\nCICIDS Classification Report:\n", classification_report(y2_test, pred_cic))

# models
joblib.dump(model_dt, "model_kdd.pkl")
joblib.dump(model_rf, "model_cic.pkl")

print("\nBoth models saved successfully!")