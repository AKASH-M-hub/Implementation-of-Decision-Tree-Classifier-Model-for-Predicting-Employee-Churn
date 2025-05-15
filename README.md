# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AKash M
RegisterNumber:  212224230013
from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['Employee.csv']))
data.head()

data.head()
data.info()
print("Null values:\n", data.isnull().sum())
print("Class distribution:\n", data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the updated data
data.head()
# Select input features
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
print(x.head())

# Define target variable
y = data["left"]
from sklearn.model_selection import train_test_split

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

from sklearn import metrics

# Predict on test set
y_pred = dt.predict(x_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new employee data
sample_prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample Prediction:", sample_prediction)

*/
```

## Output:
![image](https://github.com/user-attachments/assets/350c9ff0-ca1b-47fb-9769-8a04928f3ecf)

![image](https://github.com/user-attachments/assets/87b85bcc-6b05-4b82-b3ae-36139da970d5)

![image](https://github.com/user-attachments/assets/5d50f275-3095-45e6-971b-7639f23e4030)

![image](https://github.com/user-attachments/assets/b2112b4a-3f52-4f94-bf8e-d042c47119a6)

![image](https://github.com/user-attachments/assets/76644e8c-0fa0-4d7f-8d77-6d39d19eeca2)

![image](https://github.com/user-attachments/assets/334ce0e2-4c2b-4579-b052-fe47abb679de)

![image](https://github.com/user-attachments/assets/472cde6b-515a-4d1f-ba04-938edb70d548)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
