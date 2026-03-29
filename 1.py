import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("kddtest+.txt", sep=",", header=None)

# 2️⃣ Separate features and label
X = df.iloc[:, 0:41]
y = df.iloc[:, 41]

# 3️⃣ One-Hot Encoding
X = pd.get_dummies(X)

# 4️⃣ Fix column name datatype issue
X.columns = X.columns.astype(str)

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Predict
y_pred = model.predict(X_test)

# 8️⃣ Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9️⃣ Save model
joblib.dump(model, "ids_model.pkl")
print("\nModel saved successfully!")
