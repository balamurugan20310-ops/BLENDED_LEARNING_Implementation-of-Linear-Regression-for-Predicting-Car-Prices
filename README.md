# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn. 

2.Load Dataset: Import the dataset containing car prices along with relevant features.

3.Data Preprocessing: Manage missing data and select key features for the model, if required.

4.Split Data: Divide the dataset into training and testing subsets. 

5.Train Model: Build a linear regression model and train it using the training data.

6.Make Predictions: Apply the model to predict outcomes for the test set. 

7.Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.

8.Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.

9.Output Results: Present the predictions and evaluation metrics.

## Program:

Program to implement linear regression model for predicting car prices and test assumptions.

Developed by: BALAMURUGAN S

RegisterNumber:  212225240020

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

df=pd.read_csv('CarPrice_Assignment.csv')
df.head()
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train_scaled,y_train)

y_pred=model.predict(x_test_scaled)

print('Name: Balamurugan S ')
print('Reg. No: 25017904')
print()
print("MODEL COEFFICIENTS:")
for feature,coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MAE':>12}:{mean_absolute_error(y_test,y_pred):>10.2f}")
print(f"{'MSE':>12}:{mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE':>12}:{np.sqrt(mean_squared_error(y_test,y_pred)):>10.2f}")
print(f"{'RMAE':>12}:{np.sqrt(mean_absolute_error(y_test,y_pred)):>10.2f}")
print(f"{'R-squared':>12}:{r2_score(y_test,y_pred):>10.2f}")

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check : Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}","\n(Values close to 2 indicate no auto correlation)")

plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="1301" height="351" alt="image" src="https://github.com/user-attachments/assets/681c607f-e649-4727-b334-8b8477696ed4" />
<img width="1184" height="751" alt="image" src="https://github.com/user-attachments/assets/9d6e1e57-fc26-4f9b-b961-928a18f01d66" />
<img width="1180" height="519" alt="image" src="https://github.com/user-attachments/assets/ed24c80d-75e9-477b-91b0-8fc400589f38" />
<img width="1170" height="473" alt="image" src="https://github.com/user-attachments/assets/be62c332-457e-4c22-9066-50a7c5bf9e9e" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
