# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Here is a 5-step algorithm for implementing Logistic Regression using Gradient Descent:  

1. **Initialize Parameters**:  
   - Set initial weights  ![ql_df3f7d842a925caf3f74624e87223401_l3](https://github.com/user-attachments/assets/7a94a877-f668-43fa-892c-9b0ba807a162) 
   to small random values or zeros.  
   - Define the learning rate  ![ql_3c4d54b3d751a45a4ded101ca0d7d8d5_l3](https://github.com/user-attachments/assets/9630ad98-a142-4069-8454-70059f315a91)


2. **Compute Predictions**:  
   - Use the sigmoid function:
     
        ![ql_94b90ee1b29ef91b1945e865184b3ae8_l3](https://github.com/user-attachments/assets/0accde5b-517f-438d-9e42-eb871875ad0b)

   - This gives the probability that the sample belongs to class 1.  

3. **Compute Loss**:  
   - Use the Binary Cross-Entropy loss function:  
      ![ql_044b481e4e5c32dc25cb62821f2712ef_l3](https://github.com/user-attachments/assets/74876d35-c7d9-4de5-ad44-674bcc03f166)

    
   -  m  is the number of training samples.  

4. **Update Parameters Using Gradient Descent**:  
   - Compute gradients:  
    
   - Update weights and bias:  

      
       ![ql_1337d5988000a7ef79e89b2c2129b91e_l3](https://github.com/user-attachments/assets/e667b63f-c104-4b18-9b61-d6673a666c79)

5. **Repeat Until Convergence**:  
   - Iterate steps 2 to 4 until the loss converges (i.e., minimal change between iterations) or a set number of iterations is reached.  
   - Use the trained model to make predictions by applying the sigmoid function to new data.  


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
import numpy as np
import pandas as pd
dataset= pd.read_csv('Placement_Data.csv')
dataset
dataset["gender"] = dataset["gender"].astype('category')
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
# labelling the columns
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

# display dataset
dataset
# selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# display dependent variables
Y
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, X, Y):
    h = sigmoid(X.dot(theta))
    return -np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))

# Gradient Descent function
def gradient_descent(theta, X, Y, alpha, num_iterations):
    m = len(Y)
    for _ in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - Y) / m
        theta -= alpha * gradient
    return theta

# Initialize theta (random weights)
theta = np.random.randn(X.shape[1])
# Train model using gradient descent
theta = gradient_descent(theta, X, Y, alpha=0.01, num_iterations=1000)
# Prediction function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h > 0.5, 1, 0)
    return y_pred

# Predict values for the dataset
y_pred = predict(theta, X)

# Compute accuracy
accuracy = np.mean(y_pred.flatten() == Y)
print("Accuracy:", accuracy)
# Define new test cases
x_new1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
x_new2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])

# Predict placement status for new inputs
y_pred_new1 = predict(theta, x_new1)
y_pred_new2 = predict(theta, x_new2)

# Display predictions
print("Prediction for x_new1:", y_pred_new1)
print("Prediction for x_new2:", y_pred_new2)


Developed by: A S Siddarth
RegisterNumber: 212224040316 
*/
```

## Output:

![Screenshot 2025-03-30 211822](https://github.com/user-attachments/assets/1edc78a5-c2ba-4b43-a29d-91487c2d599e)

![Screenshot 2025-03-30 211833](https://github.com/user-attachments/assets/1c91be3b-8249-46b4-a8ab-b7e2e5a297b1)

![Screenshot 2025-03-30 211838](https://github.com/user-attachments/assets/485771b5-8982-43c1-880b-ac23e643c380)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

