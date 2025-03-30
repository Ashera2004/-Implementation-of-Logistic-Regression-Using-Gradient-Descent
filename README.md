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

Developed by: 
RegisterNumber:  
*/
```

## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

