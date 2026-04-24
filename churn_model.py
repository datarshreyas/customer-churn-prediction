import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic cleaning
df = df.dropna()

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convert target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop unnecessary column
df = df.drop('customerID', axis=1)

# Convert categorical variables
df = pd.get_dummies(df)

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Evaluation
print(classification_report(y_test, preds))

# Feature importance
importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

print("\nTop Features Influencing Churn:\n", feat_df)

# Plot
feat_df.plot(kind='barh', x='Feature', y='Importance')
plt.gca().invert_yaxis()
plt.title("Top Features Influencing Churn")
plt.show()
