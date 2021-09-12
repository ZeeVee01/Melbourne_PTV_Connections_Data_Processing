import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

wd = os.getcwd()

ptv_income = pd.read_csv("datasets/ToGraph/PTV_AND_SA3_DATA.csv")


ptv_income = ptv_income.fillna(0)
ptv_income['TotalCount'] = ptv_income['BusCount']+ptv_income['TrainCount']+ptv_income['TramCount']

# print(ptv_income)

X1 = pd.DataFrame(ptv_income,columns = ['MEAN_INCOME'])
X2 = pd.DataFrame(ptv_income, columns = ['MEDIAN_INCOME'])
X3 = pd.DataFrame(ptv_income, columns = ['MEAN_INCOME', 'MEDIAN_INCOME', 'POP_DENSITY_KM2', 'LAND_AREA_KM2']) 
X4 = pd.DataFrame(ptv_income, columns = ['MEAN_INCOME', 'POP_DENSITY_KM2'])
X5 = pd.DataFrame(ptv_income, columns = ['MEDIAN_INCOME', 'POP_DENSITY_KM2'])
X6 = pd.DataFrame(ptv_income, columns = ['POP_DENSITY_KM2'])




Y = pd.DataFrame(ptv_income, columns = ['TotalCount'])


X1_train, X1_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=42) ## Splits data into training and testing data 
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.2, random_state=42)
X3_train, X3_test, Y_train, Y_test = train_test_split(X3, Y, test_size=0.2, random_state=42)
X4_train, X4_test, Y_train, Y_test = train_test_split(X4, Y, test_size=0.2, random_state=42)
X5_train, X5_test, Y_train, Y_test = train_test_split(X5, Y, test_size=0.2, random_state=42)
X6_train, X6_test, Y_train, Y_test = train_test_split(X6, Y, test_size=0.2, random_state=42)






Y_train = Y_train['TotalCount'].to_numpy().reshape(-1,1) ## Reshapes array so that it is compatible for plots 
Y_test = Y_test['TotalCount'].to_numpy().reshape(-1,1)

lm1 = linear_model.LinearRegression()
model1 = lm1.fit(X1_train, Y_train) ##Fits target(Y) to the given X data, specifically MEAN_INCOME and an intercept for model 1 


Y1_test_predictions = lm1.predict(X1_test) ##This gives our model test data, and predicts a response. We see the comparison between the predicted and actual values
print(' Y actual values', Y_test)
print('Y1_test_predeictions', Y1_test_predictions)

print('intercept model1:', model1.intercept_) ##Prints model coeficient
print('slope:', model1.coef_) ## Prints model slope 

r2_X1_test = lm1.score(X1_test, Y_test)
r2_X1_train = lm1.score(X1_train, Y_train)

print('r2 X1 test:', r2_X1_test) ##R^2 test for test and training data. This is an indication of how much of variance in the target is explained by variation in our predictor variables(MEAN_INCOME for model1)
print('r2 X1 train:',r2_X1_train) ## We get low R^2 values, and they are not similar for train and test, suggesting our predictor variable is not a good indicator


Y_test_h1 = lm1.predict(X1_test)
Y_train_h1 = lm1.predict(X1_train)


residual_train1 = [Y_train - Y_train_h1]
residual_test1 = [Y_test - Y_test_h1]

m = 'TotalCount = 699.96169965 + -0.00426644*MEAN_INCOME '  

plt.scatter(Y_test_h1, residual_test1,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X1_test)) ##Plots residuals, there does not appear to be any trend, which is good
plt.scatter(Y_train_h1, residual_train1, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X1_train))
plt.plot([min(Y_train_h1), max(Y_train_h1)], [0,0], color= 'C2')
plt.legend()
plt.title("Residule plot\n{}".format(m))
plt.show()








lm2 = linear_model.LinearRegression()
model2 = lm2.fit(X2_train, Y_train)

Y2_test_predictions = lm2.predict(X2_test)
# print(Y2_test_predictions)
# print(Y_test)

print('intercept model2:', model2.intercept_)
print('slope:', model2.coef_)

r2_X2_test = lm2.score(X2_test, Y_test)
r2_X2_train = lm2.score(X2_train, Y_train)

print('r2 X2 test:', r2_X2_test)
print('r2 X2 train:',r2_X2_train)


Y_test_h2 = lm2.predict(X2_test)
Y_train_h2 = lm2.predict(X2_train)


residual_train2 = [Y_train - Y_train_h2]
residual_test2 = [Y_test - Y_test_h2]




m = 'TotalCount = 1210.44448597 + -0.0163578*MEDIAN_INCOME '  

plt.scatter(Y_test_h2, residual_test2,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X2_test))
plt.scatter(Y_train_h2, residual_train2, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X2_train))
plt.plot([min(Y_train_h2), max(Y_train_h2)], [0,0], color= 'C2')
plt.legend()
plt.title("Residule plot\n{}".format(m))
plt.show()







lm3 = linear_model.LinearRegression()
model3 = lm3.fit(X3_train, Y_train)

Y3_test_predictions = lm3.predict(X3_test)

# print(Y3_test_predictions)
# print(Y_test)

print('intercept model3:', model3.intercept_) 
print('coefficients:', model3.coef_)


r2_X3_test = lm3.score(X3_test, Y_test)
r2_X3_train = lm3.score(X3_train, Y_train)

print('r2 X3 test:', r2_X3_test)
print('r2 X3 train:',r2_X3_train)

# m = 'MEDV_h = -3.84 + 5.47*RM -0.63*LSTAT'

Y_test_h3 = lm3.predict(X3_test)
Y_train_h3 = lm3.predict(X3_train)



residual_train3 = [Y_train - Y_train_h3]
residual_test3 = [Y_test - Y_test_h3]



m = 'TotalCount = 1445.40635874 + 0.00179186*MEAN_INCOME - 0.02468666*MEDIAN_INCOME + 0.02168637*POP_DENSITY_KM2 + 0.01701446*LAND_AREA_KM2 '  


plt.scatter(Y_test_h3, residual_test3,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X3_test))
plt.scatter(Y_train_h3, residual_train3, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X3_train))
plt.plot([min(Y_train_h3), max(Y_train_h3)], [0,0], color= 'C2')
plt.legend()
plt.title("Residule plot\n{}".format(m))
plt.show()









lm4 = linear_model.LinearRegression()
model4 = lm4.fit(X4_train, Y_train)

Y4_test_predictions = lm4.predict(X4_test)

# print(Y3_test_predictions)
# print(Y_test)

print('intercept model4:', model4.intercept_) 
print('coefficients:', model4.coef_)


r2_X4_test = lm4.score(X4_test, Y_test)
r2_X4_train = lm4.score(X4_train, Y_train)

print('r2 X4 test:', r2_X4_test)
print('r2 X4 train:',r2_X4_train)

# m = 'MEDV_h = -3.84 + 5.47*RM -0.63*LSTAT'

Y_test_h4 = lm4.predict(X4_test)
Y_train_h4 = lm4.predict(X4_train)



residual_train4 = [Y_train - Y_train_h4]
residual_test4 = [Y_test - Y_test_h4]



m = 'TotalCount = 716.32599338 - 0.00491558*MEAN_INCOME + 0.01134801*POP_DENSITY_KM2'

plt.scatter(Y_test_h4, residual_test4,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X4_test))
plt.scatter(Y_train_h4, residual_train4, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X4_train))
plt.plot([min(Y_train_h4), max(Y_train_h4)], [0,0], color= 'C2')
plt.legend()
plt.title("Residule plot\n{}".format(m))
plt.show()










lm5 = linear_model.LinearRegression()
model5 = lm5.fit(X5_train, Y_train)

Y5_test_predictions = lm5.predict(X5_test)

# print(Y3_test_predictions)
# print(Y_test)

print('intercept model5:', model5.intercept_) 
print('coefficients:', model5.coef_)


r2_X5_test = lm5.score(X5_test, Y_test)
r2_X5_train = lm5.score(X5_train, Y_train)

print('r2 X5 test:', r2_X5_test)
print('r2 X5 train:',r2_X5_train)

# m = 'MEDV_h = -3.84 + 5.47*RM -0.63*LSTAT'

Y_test_h5 = lm5.predict(X5_test)
Y_train_h5 = lm5.predict(X5_train)



residual_train5 = [Y_train - Y_train_h5]
residual_test5 = [Y_test - Y_test_h5]


m = 'TotalCount = 1341.95814101 - 0.02008306*MEDIAN_INCOME + 0.02140924*POP_DENSITY_KM2'

plt.scatter(Y_test_h5, residual_test5,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X5_test))
plt.scatter(Y_train_h4, residual_train5, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X5_train))
plt.plot([min(Y_train_h5), max(Y_train_h5)], [0,0], color= 'C2')
plt.legend()
plt.title("Residule plot\n{}".format(m))
plt.show()








lm6 = linear_model.LinearRegression()
model6 = lm6.fit(X6_train, Y_train)

Y6_test_predictions = lm6.predict(X6_test)
# print(Y2_test_predictions)
# print(Y_test)

print('intercept model6:', model6.intercept_)
print('slope:', model6.coef_)

r2_X6_test = lm6.score(X6_test, Y_test)
r2_X6_train = lm6.score(X6_train, Y_train)

print('r2 X6 test:', r2_X6_test)
print('r2 X6 train:',r2_X6_train)


Y_test_h6 = lm6.predict(X6_test)
Y_train_h6 = lm6.predict(X6_train)


residual_train6 = [Y_train - Y_train_h6]
residual_test6 = [Y_test - Y_test_h6]




m = 'TotalCount = 1210.44448597 + -0.0163578*MEDIAN_INCOME '  

plt.scatter(Y_test_h6, residual_test6,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X6_test))
plt.scatter(Y_train_h6, residual_train6, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X6_train))
plt.plot([min(Y_train_h6), max(Y_train_h6)], [0,0], color= 'C2')
plt.legend()
plt.title("Residule plot\n{}".format(m))
plt.show()
