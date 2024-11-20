#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load the dataset
df = pd.read_csv("C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER01\MSSQL\Backup\Telco.csv")


# In[3]:


# Handle missing values for TotalCharges by converting to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[4]:


# Encode categorical features
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'OnlineBackup', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})


# In[5]:


# Encode gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})


# In[6]:


# One-hot encoding for columns with multiple categories
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines'], drop_first=True)


# In[7]:


# Drop unnecessary columns
df = df.drop(columns=['customerID', 'StreamingMovies', 'StreamingTV', 'TechSupport', 'OnlineSecurity', 'DeviceProtection'])


# In[8]:


# Feature and target split
X = df.drop(columns='Churn')
y = df['Churn']


# In[9]:


# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[10]:


# Impute missing values in numeric columns with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[11]:


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


# Train Logistic Regression Model with increased max_iter
model = LogisticRegression(max_iter=2000)  # Increased max_iter
model.fit(X_train, y_train)


# In[13]:


# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# In[14]:


# Output model accuracy and classification report
print(f"Accuracy: {accuracy}")
classification_report_output = classification_report(y_test, y_pred, output_dict=True)
classification_report_df = pd.DataFrame(classification_report_output).transpose()


# In[15]:


# Print classification report
print("Classification Report:\n", classification_report_df)


# In[16]:


# Combine predictions with test data for visualization
results = pd.DataFrame(X_test, columns=df.drop(columns='Churn').columns)  # Create DataFrame from X_test
results['Actual_Churn'] = y_test.values  # Use .values to get the underlying data as a NumPy array
results['Predicted_Churn'] = y_pred


# In[17]:


# Return the results (for example, sending to Power BI)
print(results)


# In[ ]:




