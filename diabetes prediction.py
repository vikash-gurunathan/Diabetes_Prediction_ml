import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('diabetes.csv')

df.head()

df.shape

df.info()

df.isnull()

df.isnull().sum()

print("The names of the features :\n", list(df.columns))

from sklearn.model_selection import train_test_split

x=df.iloc[:,df.columns!='Outcome']
y=df.iloc[:,df.columns=='Outcome']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

xtrain.head()

ytrain.head()


# ## Data Visualization

df.hist("Age")

sns.distplot(df["SkinThickness"])

sns.set(palette='BrBG')
df.hist(figsize=(20,20));


# # Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

model.fit(xtrain,ytrain.values.ravel())

predict_output = model.predict(xtest)
print(predict_output)

from sklearn.metrics import accuracy_score

acc=accuracy_score(predict_output,ytest)
print("The accuracy score for RF:",acc)


