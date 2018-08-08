############################################
# Simple Linear Regression - Using SKLEARN #
############################################
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import RegressionDatasets as regdata
from math import sqrt

def execute_algorithm(dataset, split):
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1-split), random_state=0)

    # Fitting Simple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Predicting the Test set results
    Y_test_pred = regressor.predict(X_test)

    ### Scatter training data in green color ###
    plt.scatter(X_train, Y_train, color='green', s=50)
    ### Scatter test data in - red color ###
    plt.scatter(X_test, Y_test, color='red', s=50)
    ### Plot the regression line based on test data - blue color ###
    plt.plot(X_test, Y_test_pred, color='blue')

    mse = mean_squared_error(Y_test, Y_test_pred)
    return mse

def main():
    # Inform about and import the dataset
    regdata.dataset_choise_print()
    dataset = regdata.dataset_choise('(Sklearn model - Split)')

    split = 0.5

    mse = execute_algorithm(dataset, split)
    print('MSE: %.0f' % (mse))
    print('RMSE: %.2f' % sqrt(mse))

    plt.show()

main()
