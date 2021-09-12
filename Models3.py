
import pandas as pd 
import os 
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math 



ptv_income = pd.read_csv("datasets/ToGraph/PTV_AND_SA3_DATA.csv")





ptv_income = ptv_income.fillna(0)
ptv_income['TotalCount'] = ptv_income['BusCount']+ptv_income['TrainCount']+ptv_income['TramCount']


ptv_income = ptv_income[ptv_income['TotalCount']!=0]
# Y['TotalCount'] = np.log10(Y['TotalCount'])
Y  = ptv_income['TotalCount']


# print(ptv_income)

X1 = pd.DataFrame(ptv_income,columns = ['MEAN_INCOME'])
X2 = pd.DataFrame(ptv_income, columns = ['MEDIAN_INCOME'])
X3 = pd.DataFrame(ptv_income, columns = ['MEAN_INCOME', 'MEDIAN_INCOME', 'POP_DENSITY_KM2', 'LAND_AREA_KM2']) 
X4 = pd.DataFrame(ptv_income, columns = ['MEAN_INCOME', 'POP_DENSITY_KM2'])
X5 = pd.DataFrame(ptv_income, columns = ['MEDIAN_INCOME', 'POP_DENSITY_KM2'])
X6 = pd.DataFrame(ptv_income, columns = ['POP_DENSITY_KM2'])

Y = pd.DataFrame(ptv_income, columns = ['TotalCount'])
Y['TotalCount'] = np.log10(Y['TotalCount'])

fig, axs = plt.subplots(2, 2,figsize=(12,12))


axs[0, 0].scatter(ptv_income['MEAN_INCOME'], Y)
axs[0, 0].set_title('MEAN_INCOME VS Log10(TotalCount)')

axs[0, 1].scatter(ptv_income['MEDIAN_INCOME'], Y)
axs[0, 1].set_title('MEDIAN_INCOME VS Log10(TotalCount)')

axs[1, 0].scatter(ptv_income['POP_DENSITY_KM2'], Y)
axs[1, 0].set_title('POP_DENSITY_KM2 VS Log10(TotalCount)')

axs[1, 1].scatter(ptv_income['LAND_AREA_KM2'], Y)
axs[1, 1].set_title('LAND_AREA_KM2 VS Log10(TotalCount)')

plt.tight_layout()

plt.suptitle('Variables vs Log10(TotalCount)')
plt.subplots_adjust(top=0.93)
plt.savefig('Variables_vs_TotalCount_log10')


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

r2_X1_test = lm1.score(X1_test, Y_test)
r2_X1_train = lm1.score(X1_train, Y_train)

Y_test_h1 = lm1.predict(X1_test)
Y_train_h1 = lm1.predict(X1_train)

residual_train1 = [Y_train - Y_train_h1]
residual_test1 = [Y_test - Y_test_h1]

coef = str(model1.coef_[0]).lstrip('[').rstrip(']').split()

m1 = 'Log10(TotalCount) = ' + str(model1.intercept_).lstrip('[').rstrip(']') +' + '+ coef[0] + '*MEAN_INCOME'

print('r2 X1 test:', r2_X1_test) ##R^2 test for test and training data. This is an indication of how much of variance in the target is explained by variation in our predictor variables(MEAN_INCOME for model1)
print('r2 X1 train:',r2_X1_train) ## We get low R^2 values, and they are not similar for train and test, suggesting our predictor variable is not a good indicator

print(m1)





lm2 = linear_model.LinearRegression()
model2 = lm2.fit(X2_train, Y_train)

Y2_test_predictions = lm2.predict(X2_test)

r2_X2_test = lm2.score(X2_test, Y_test)
r2_X2_train = lm2.score(X2_train, Y_train)

Y_test_h2 = lm2.predict(X2_test)
Y_train_h2 = lm2.predict(X2_train)

residual_train2 = [Y_train - Y_train_h2]
residual_test2 = [Y_test - Y_test_h2]

coef = str(model2.coef_[0]).lstrip('[').rstrip(']').split()

m2 = 'Log10(TotalCount) = ' + str(model2.intercept_).lstrip('[').rstrip(']') +' + '+ coef[0] + '*MEDIAN_INCOME' \

print('r^2 X2 test:', r2_X2_test)
print('r^2 X2 train:',r2_X2_train)

print(m2)







lm3 = linear_model.LinearRegression()
model3 = lm3.fit(X3_train, Y_train)

Y3_test_predictions = lm3.predict(X3_test)

r2_X3_test = lm3.score(X3_test, Y_test)
r2_X3_train = lm3.score(X3_train, Y_train)

Y_test_h3 = lm3.predict(X3_test)
Y_train_h3 = lm3.predict(X3_train)

residual_train3 = [Y_train - Y_train_h3]
residual_test3 = [Y_test - Y_test_h3]

coef = str(model3.coef_[0]).lstrip('[').rstrip(']').split()

m3 = 'Log10(TotalCount) = ' + str(model3.intercept_).lstrip('[').rstrip(']') +' + '+ coef[0] + '*MEAN_INCOME' + '\n' \
+ ' + '+ coef[1] + '*MEDIAN_INCOME' \
+ ' + '+ coef[2] + '*POP_DENSITY_KM2' + '\n' \
+ ' + '+ coef[3] + '*LAND_AREA_KM2' 

print('r2 X3 test:', r2_X3_test)
print('r2 X3 train:',r2_X3_train)

print(m3)









lm4 = linear_model.LinearRegression()
model4 = lm4.fit(X4_train, Y_train)

r2_X4_test = lm4.score(X4_test, Y_test)
r2_X4_train = lm4.score(X4_train, Y_train)

Y_test_h4 = lm4.predict(X4_test)
Y_train_h4 = lm4.predict(X4_train)

residual_train4 = [Y_train - Y_train_h4]
residual_test4 = [Y_test - Y_test_h4]

coef = str(model4.coef_[0]).lstrip('[').rstrip(']').split()

m4 = 'Log10(TotalCount) = ' + str(model4.intercept_).lstrip('[').rstrip(']') +' + '+ coef[0] + '*MEAN_INCOME' \
+ ' + '+ coef[1] + '*POP_DENSITY_KM2'

print('r^2 X4 test:', r2_X4_test)
print('r^2 X4 train:',r2_X4_train)

print(m4)






lm5 = linear_model.LinearRegression()
model5 = lm5.fit(X5_train, Y_train)

r2_X5_test = lm5.score(X5_test, Y_test)
r2_X5_train = lm5.score(X5_train, Y_train)

Y_test_h5 = lm5.predict(X5_test)
Y_train_h5 = lm5.predict(X5_train)

residual_train5 = [Y_train - Y_train_h5]
residual_test5 = [Y_test - Y_test_h5]

coef = str(model5.coef_[0]).lstrip('[').rstrip(']').split()

m5 = 'Log10(TotalCount) = ' + str(model5.intercept_).lstrip('[').rstrip(']') +' + '+ coef[0] + '*MEDIAN_INCOME' \
+ ' + '+ coef[1] + '*POP_DENSITY_KM2' \

print('r^2 X5 test:', r2_X5_test)
print('r^2 X5 train:',r2_X5_train)

print(m5)





lm6 = linear_model.LinearRegression()
model6 = lm6.fit(X6_train, Y_train)

r2_X6_test = lm6.score(X6_test, Y_test)
r2_X6_train = lm6.score(X6_train, Y_train)

Y_test_h6 = lm6.predict(X6_test)
Y_train_h6 = lm6.predict(X6_train)

residual_train6 = [Y_train - Y_train_h6]
residual_test6 = [Y_test - Y_test_h6]

coef = str(model6.coef_[0]).lstrip('[').rstrip(']').split()

m6 = 'Log10(TotalCount) = ' + str(model6.intercept_).lstrip('[').rstrip(']') + ' + '+ coef[0] + '*POP_DENSITY_KM2' \

print('r^2 X6 test:', r2_X6_test)
print('r^2 X6 train:',r2_X6_train)


print(m6)


fig, axs = plt.subplots(3, 2,figsize=(15,15))


axs[0,0].scatter(Y_test_h1, residual_test1,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X1_test))
axs[0,0].scatter(Y_train_h1, residual_train1, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X1_train))
axs[0,0].plot([min(Y_train_h1), max(Y_train_h1)], [0,0], color= 'C2')
axs[0,0].legend()
axs[0,0].set_title(m1)

axs[0,1].scatter(Y_test_h2, residual_test2,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X2_test))
axs[0,1].scatter(Y_train_h2, residual_train2, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X2_train))
axs[0,1].plot([min(Y_train_h2), max(Y_train_h2)], [0,0], color= 'C2')
axs[0,1].legend()
axs[0,1].set_title(m2)


axs[1,0].scatter(Y_test_h3, residual_test3,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X3_test))
axs[1,0].scatter(Y_train_h3, residual_train3, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X3_train))
axs[1,0].plot([min(Y_train_h3), max(Y_train_h3)], [0,0], color= 'C2')
axs[1,0].legend()
axs[1,0].set_title(m3)

axs[1,1].scatter(Y_test_h4, residual_test4,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X4_test))
axs[1,1].scatter(Y_train_h4, residual_train4, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X4_train))
axs[1,1].plot([min(Y_train_h4), max(Y_train_h4)], [0,0], color= 'C2')
axs[1,1].legend()
axs[1,1].set_title(m4)

axs[2,0].scatter(Y_test_h5, residual_test5,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X5_test))
axs[2,0].scatter(Y_train_h5, residual_train5, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X5_train))
axs[2,0].plot([min(Y_train_h5), max(Y_train_h5)], [0,0], color= 'C2')
axs[2,0].legend()
axs[2,0].set_title(m5)


axs[2,1].scatter(Y_test_h6, residual_test6,color='C0', label = 'R^2 (test):{0:.2f}'.format(r2_X6_test))
axs[2,1].scatter(Y_train_h6, residual_train6, color='C4', alpha = 0.5, label = 'R^2 (training):{0:.2f}'.format(r2_X6_train))
axs[2,1].plot([min(Y_train_h6), max(Y_train_h6)], [0,0], color= 'C2')
axs[2,1].legend()
axs[2,1].set_title(m6)


plt.tight_layout()
plt.suptitle('Residuals')
plt.subplots_adjust(top=0.93)
plt.savefig('Residuals_graphs_log10')
