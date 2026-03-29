import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

#  Load dataset
df = pd.read_csv("kddtest+.txt", sep=",", header=None)

#  Separate features and label
X = df.iloc[:, 0:41]
y = df.iloc[:, 41]

#  One-Hot Encoding
X = pd.get_dummies(X)

#  Fix column name datatype issue
X.columns = X.columns.astype(str)

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#  Predict
y_pred = model.predict(X_test)

#  Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#  Save model
joblib.dump(model, "ids_model.pkl")
print("\nModel saved successfully!")
