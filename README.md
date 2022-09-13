# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## PROBLEM STATEMENT

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## NEURAL NETWORK MODEL

![nn](https://user-images.githubusercontent.com/75234991/189541018-dc233ce7-8d92-483d-aa8c-878d2ac664c4.svg)

## DESIGN STEPS

### Step 1:
We start by reading the dataset using pandas.
### Step 2:
The dataset is then preprocessed, i.e, we remove the features that don't contribute towards the result.
### Step 3:
The null values are removed aswell
### Step 4:
The resulting data values are then encoded. We, ensure that all the features are of the type int, or float, for the model to better process the dataset.
### Step 5:
Once the preprocessing is done, we split the available data into Training and Validation datasets.
### Step 6:
The Sequential model is then build using 1 input, 3 dense layers(hidden) and, output layer.
### Step 7:
The model is then complied and trained with the data. A call back method is also implemented to prevent the model from overfitting.
### Step 8:
Once the model is done training, we validate and use the model to predict values.

## PROGRAM

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')

from google.colab import drive
drive.mount('/content/drive')

customer_df.columns

customer_df.dtypes

customer_df.shape

customer_df.isnull().sum()

customer_df_cleaned = customer_df.dropna(axis=0)

customer_df_cleaned.isnull().sum()

customer_df_cleaned.shape

customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()

customer_df_cleaned['Ever_Married'].unique()

customer_df_cleaned['Graduated'].unique()

customer_df_cleaned['Profession'].unique()

customer_df_cleaned['Spending_Score'].unique()

customer_df_cleaned['Var_1'].unique()

customer_df_cleaned['Segmentation'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)

customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])

customers_1.dtypes

le = LabelEncoder()

customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)

customers_1.dtypes

corr = customers_1.corr()

customers_1['Segmentation'].unique()

X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = customers_1[['Segmentation']].values

one_hot_enc = OneHotEncoder()

one_hot_enc.fit(y1)

y1.shape

y = one_hot_enc.transform(y1).toarray()

y.shape

y1[0]

y[0]

X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=50)

X_train[0]

X_train.shape

scaler_age = MinMaxScaler()

scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

ai_brain = Sequential([
  Dense(8,input_shape=(8,)),
  Dense(16,activation='relu'),
  Dense(16,activation='tanh'),
  Dense(4,activation='softmax'),
])

ai_brain.compile(optimizer='adam',loss= 'categorical_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)

ai_brain.fit(x=X_train_scaled,y=y_train,epochs=2000,batch_size=256,validation_data=(X_test_scaled,y_test),callbacks=[early_stop])

metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

ai_brain.save('customer_classification_model.h5')
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

ai_brain = load_model('customer_classification_model.h5')

with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))
```

## DATASET INFORMATION

![188649793-d5045c93-f721-47fd-b504-021494fcb256](https://user-images.githubusercontent.com/75234991/189541313-1e96b1c9-8c90-440e-923a-8111236f4789.png)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75234991/189541329-8b3cf70f-9314-40d9-a175-c4955aaeef72.png)

### Classification Report

<img width="345" alt="image" src="https://user-images.githubusercontent.com/75234991/189541397-c4e7385b-c084-4079-a9cf-0372d2166eee.png">

### Confusion Matrix

<img width="123" alt="image" src="https://user-images.githubusercontent.com/75234991/189541392-51bb088c-82a7-4f36-9b2b-c061cb541c75.png">

### New Sample Data Prediction

<img width="46" alt="image" src="https://user-images.githubusercontent.com/75234991/189541441-21a1ec42-e113-4b15-977e-f07c51ce71fe.png">

## RESULT

Thus, a Simple Neural Network Classification Model is developed successfully.
