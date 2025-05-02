import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle

# Load the dataset
data = pd.read_csv(r'F:\Projects\Machine Learning\CODSOFT\Bank_Customer_churn\Churn_Modelling.csv')

# Preview data
print("Train data")
print(data.head())
print("\n")

# Save 'CustomerId' and 'Surname' before dropping
customer_info = data[['CustomerId', 'Surname']]

# Drop unnecessary columns
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# One-hot encode 'Geography' and map 'Gender'
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features and target
X = data.drop(columns=['Exited'])  # Features
y = data['Exited']                # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # Handle class imbalance
)
xgb_model.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=True)

print("Model expects these features:")
print(xgb_model.feature_names_in_)

# Predict
y_pred = xgb_model.predict(X_test)

# Merge CustomerId & Surname with predictions
comparison_df = customer_info.loc[X_test.index].copy()
comparison_df['True_Exited'] = y_test
comparison_df['Predicted_Exited'] = y_pred

# Show sample output
print(comparison_df[['CustomerId', 'Surname', 'True_Exited', 'Predicted_Exited']].head())
print("\n")

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': xgb_model,
        'scaler': scaler
    }, f)

print("XGBoost model saved as xgb_model.pkl")
