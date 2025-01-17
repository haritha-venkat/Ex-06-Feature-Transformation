# Ex-06-Feature-Transformation
# AIM

To read the given data and perform Feature Transformation process and save the data to a file.
# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
# ALGORITHM
### STEP 1

Read the given Data
### STEP 2

Clean the Data Set using Data Cleaning Process
### STEP 3

Apply Feature Transformation techniques to all the features of the data set
### STEP 4
Save the data to the file
# CODE
```python
Name : HARITHASHREE.V
Register Number : 212222230046
**Feature Transformation - Data_to_Transform.csv**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
```

# OUPUT:
## Feature Transformation - Data_to_Transform.csv:
![image](https://user-images.githubusercontent.com/121285701/233553560-1f7359ee-ce00-4767-a06c-54313acbccf0.png)
![image](https://user-images.githubusercontent.com/121285701/233553596-88dca919-cfc4-439c-bbf2-6b27e4193cea.png)
![image](https://user-images.githubusercontent.com/121285701/233553660-8c3f460a-2fc8-4795-b00f-21250768bd42.png)
![image](https://user-images.githubusercontent.com/121285701/233553704-694fe860-40e6-48a9-a0cd-4438b4dccca2.png)
![image](https://user-images.githubusercontent.com/121285701/233555011-48ad3d8d-5df6-47dd-991b-1d4998875728.png)
## Log Transformation
![image](https://user-images.githubusercontent.com/121285701/233553831-527be46e-b397-4536-9b58-e7c76afaade4.png)
## Reciprocal transformation
![image](https://user-images.githubusercontent.com/121285701/233554018-0402d3f7-e869-4adb-b992-4a035ad14d40.png)
## SquareRoot Transformation
![image](https://user-images.githubusercontent.com/121285701/233554129-8a163add-1e44-4101-a497-e98d45252172.png)
## Power Transformation
![image](https://user-images.githubusercontent.com/121285701/233554205-a448147e-1a58-4e53-bf1a-78604e16c6f2.png)
![image](https://user-images.githubusercontent.com/121285701/233554230-db2b4cb6-6ea0-486a-b4e6-b0746ca40aec.png)
## Quantile Transformation
![image](https://user-images.githubusercontent.com/121285701/233554309-b70139d3-7668-43d9-a400-faa39a1a21fc.png)
## RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully
