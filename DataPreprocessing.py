import numpy as np
import pandas as pd

df= pd.read_csv('train.csv')


# Drop unnecessary columns

df.drop(['Name', 'Ticket', 'Cabin','Fare'], axis=1, inplace=True)

#Check the missing values in the training dataset

missing_values = df.isnull().sum()
print(missing_values)
missing_values_percentage = (missing_values / len(df)) * 100
print(missing_values_percentage)

#Handling missing values based on survival column values


df['Age'] = df.groupby('Survived')['Age'].transform(lambda x: x.fillna(x.median()))

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#Check the missing values in the training dataset

missing_values = df.isnull().sum()
print(missing_values)
missing_values_percentage = (missing_values / len(df)) * 100
print(missing_values_percentage)

#Convert categorical variables into numerical

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


import matplotlib.pyplot as plt


#Visualize the percent of male and female using pie charts in training dataset

male=(df['Sex']==0).sum()
female=(df['Sex']==1).sum()
p=[male, female]
plt.pie(p,labels=['Male', 'Female'],colors=["Red" ,"Green"],startangle=0)
plt.axis('equal')
plt.show()

#repeat the same for testing dataset

df1= pd.read_csv('test.csv')


# Drop unnecessary columns

df1.drop(['Name', 'Ticket', 'Cabin','Fare'], axis=1, inplace=True)

#Check the missing values in the training dataset

missing_values = df1.isnull().sum()
print(missing_values)
missing_values_percentage = (missing_values / len(df1)) * 100
print(missing_values_percentage)

#Handling missing values

df1['Age'] = df1['Age'].fillna(df1['Age'].sum())

df1['Embarked'] = df1['Embarked'].fillna(df1['Embarked'].mode()[0])

#Check the missing values in the training dataset

missing_values = df1.isnull().sum()
print(missing_values)
missing_values_percentage = (missing_values / len(df1)) * 100
print(missing_values_percentage)

#Convert categorical variables into numerical

df1['Sex'] = df1['Sex'].map({'male': 0, 'female': 1})
df1['Embarked'] = df1['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


import matplotlib.pyplot as plt


#Visualize the percent of male and female using pie charts in test dataset

male=(df1['Sex']==0).sum()
female=(df1['Sex']==1).sum()
p=[male, female]
plt.pie(p,labels=['Male', 'Female'],colors=["Red" ,"Green"],startangle=0)
plt.axis('equal')
plt.show()
