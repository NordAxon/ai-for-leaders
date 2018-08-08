############################################
# Simple Linear Regression - Using SKLEARN #
############################################
import numpy as np
from math import sqrt
import sklearn.linear_model as sckit_linearmodel
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import RegressionDatasets as regdata

def execute_algorithm(dataset):
    data_to_plot = np.array(dataset)

    ### Scatter real data in red color ###
    X, Y = data_to_plot.T
    plt.scatter(X, Y, color='red',s=50)

    # Transform horizontal first column values into new vertical array
    X1 = []
    for i in range(len(dataset)):
        X1.append(np.array([dataset[i][0]]))
    X1 = np.array(X1)

    ### Fitting Simple Linear Regression to the Data Set ###
    regressor = sckit_linearmodel.LinearRegression()
    regressor.fit(X1, Y)

    ### Predicting the regression line ###
    Y_pred = regressor.predict(X1)
    plt.plot(X, Y_pred, color='blue')

    mse = mean_squared_error(Y, Y_pred)
    return mse

def main():
    # Inform about and import the dataset
    regdata.dataset_choise_print()
    dataset = regdata.dataset_choise('(Sklearn model - No split)')

    for i in range(len(dataset[0])):
        regdata.str_column_to_float(dataset, i)

    mse = execute_algorithm(dataset)
    print('MSE: %.0f' % (mse))
    print('RMSE: %.2f' % sqrt(mse))

    plt.show()

main()










''' Visualising the Test set results
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title ('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
'''

#Feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


print('X len = ',len(X))
print('Y len = ',len(Y))


print('predicted = ', Y_pred)
print('actual = ', Y)

'''