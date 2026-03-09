# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights and bias
2.Choose a loss function

3.Set hyperparameters (learning rate, regularization, etc.)

4.Shuffle training data

5.For each epoch:

6.For each training sample:

7.Compute gradient of loss

8.Update weights using gradient descent

9.Apply regularization

10.Check for convergence or stop after max iterations

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SAKTHIVEL P
RegisterNumber:  25014510
*/

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Step 5: Create and train SGDClassifier for Logistic Regression
# ------------------------------
sgd_model = SGDClassifier(loss='log_loss',       # 'log' loss → logistic regression
                          max_iter=1000,
                          learning_rate='optimal',
                          random_state=42)
sgd_model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)   # Probability of placement

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict placement for a new student
# ------------------------------
new_student = np.array([[6, 68, 1]])  # Example: 6 hours, 68 prev score, Internship yes
new_student_scaled = scaler.transform(new_student)
placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")
```

## Output:
<img width="620" height="412" alt="image" src="https://github.com/user-attachments/assets/da563e7d-1265-4c94-8e3a-f47430570fd3" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
