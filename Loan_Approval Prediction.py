import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
os.makedirs("Visualizations", exist_ok=True)

df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
#print(df.head())

#print("Data shape:", df.shape)
#print("Data info:", df.info())
#print("Missing Values:\n", df.isnull().sum())
#print("Statistical Summary:\n", df.describe())

categorcal_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in categorcal_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


numeric_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
#print(df.isnull().sum())

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
#sns.countplot(x='Loan_Status',data=df)
plt.title('Loan Approval Status Count')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visualizations/Loan Status Count.png')
plt.close()

plt.figure(figsize=(6,4))
#sns.countplot(x='Gender', data=df)
plt.title('Loan Status by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visualizations/Loan Status by Gender')
plt.close()

plt.figure(figsize=(6,4))
#sns.countplot(x='Married', hue='Loan_Status', data=df)
plt.title('Loan Status by Marrital Status')
plt.xlabel('Married')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visualizations/Loan by Marrital Status')
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x= 'Loan_Status', y= 'ApplicantIncome', hue= 'Education', data=df)
plt.title('ApplicantIncome vs Loan Status by Education')
plt.xlabel('Loan Status')
plt.ylabel('Applicant Income')
plt.tight_layout()
plt.savefig('Visualizations/Income vs Loan Status')
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='Self_Employed', hue='Loan_Status', data=df)
plt.title('Loan Status by Employment Type')
plt.xlabel('Self Employed')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visualizations/self_employed_vs_loan_status.png')
plt.close()

numeric_df = df.select_dtypes(include= ['int64', 'float64'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt= '.2f')
plt.title('Correlation Heatmap- Numeric features Only')
plt.tight_layout()
plt.savefig('Visualizations/correlation_heatmap.png')
plt.close()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categorcal_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorcal_cols:
    df[col] = le.fit_transform(df[col])

#print(df.info())

X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']
from sklearn.preprocessing import StandardScaler
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state= 42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
y_pred_test = log_model.predict(x_test)
#print("Accuracy:", accuracy_score(y_test, y_pred_test))
#print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred_test))
#print("Classification Report:\n",classification_report(y_test, y_pred_test))

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
y_pred_test = rf_model.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))


