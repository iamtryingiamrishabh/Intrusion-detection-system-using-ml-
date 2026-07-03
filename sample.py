import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load Dataset
df = pd.read_csv("kddtest+.txt", sep=",", header=None)

# 2. Data Cleaning & Visualization of Classes
# Visualizing how many 'Normal' vs 'Attack' labels we have
plt.figure(figsize=(10, 6))
sns.countplot(x=df[41])
plt.title("Class Distribution (Normal vs. Various Attacks)")
plt.xticks(rotation=90)
plt.show()

# 3. Features and Label
X = df.iloc[:, 0:41]
y = df.iloc[:, 41]

# Convert categorical text to numbers (One-Hot Encoding)
X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

# 4. Feature Scaling (NEW: Essential for Network Data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 8. FEATURE IMPORTANCE (NEW: Shows what the model looks for)
plt.figure(figsize=(10, 8))
# Getting the top 15 most important features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh', color='skyblue')
plt.title("Top 15 Most Important Features in Detecting Attacks")
plt.xlabel("Importance Score")
plt.show()

# 9. CONFUSION MATRIX (Enhanced with Heatmap)
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
# Using Seaborn for a prettier heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 10. Accuracy Comparison Bar Chart
plt.figure(figsize=(6, 5))
bars = plt.bar(["Train", "Test"], [train_acc, test_acc], color=['orange', 'green'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2%}", ha='center', va='bottom')
plt.title("Model Overfitting Check (Train vs Test)")
plt.ylim(0, 1.1)
plt.show()

# Final Report & Save
print("\nClassification Report:\n", classification_report(y_test, y_pred))
joblib.dump(model, "ids_model_enhanced.pkl")
joblib.dump(scaler, "scaler.pkl") # Save the scaler too!
print("\nModel and Scaler saved successfully!")