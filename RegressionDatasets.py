from csv import reader
from matplotlib import pyplot as plt
import pandas as pd

# Load CSV file
def load_csv_file(filename):
    dataset_list = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset_list.append(row)
    return dataset_list

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def dataset_choise_print():
    ###### Available datasets ######
    print('Which dataset do you want to use for training & testing this linear regression model? Choose by typing dataset number [1, 2, 3 ...] ' + '\n'
    + '1) Salary vs Experience dataset, predicting salary in [SEK] for Swedish dentists given their years of experience. ' + '\n'
    + '2) Interest rates vs House prices, predicting houses prices [USD] given the interest rate.' + '\n'
    + '3) Claims payment vs Nr of claims, predicting total payment for all swedish auto claims in [SEK] given the total number of claims.' + '\n'
    + '4) Small & simple dataset' + '\n'
    + '--> Type 0 (zero) for Quit!'
    )

def dataset_choise(algorithm):
    filename = 'Salary_Data.csv'
    #plt.figure(figsize=(20, 10))
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.set_window_title(algorithm)

    dataset_choise_nr = input('Enter your choise: ')

    if int(dataset_choise_nr) == 1:
        filename = 'Salary_Data.csv'
        plt.title('SWEDISH DENTISTS - SALARY vs EXPERIENCE', fontsize=22)
        plt.xlabel('Years of experience', fontsize=20)
        plt.ylabel('Salary [SEK]', fontsize=20)
    elif int(dataset_choise_nr) == 2:
        filename = 'InterestRate_HomePrice.csv'
        plt.title('HOME PRICES - INTEREST RATES', fontsize=22)
        plt.xlabel('Interest rate', fontsize=20)
        plt.ylabel('Home price[USD]', fontsize=20)
    elif int(dataset_choise_nr) == 3:
        filename = 'Claims.csv'
        plt.title('AUTO INSURANCE IN SWEDEN - CLAIMS PAYMENT vs NR OF CLAIMS', fontsize=22)
        plt.xlabel('Tot nr of claims', fontsize=20)
        plt.ylabel('Claims payment [SEK]', fontsize=20)
    elif int(dataset_choise_nr) == 4:
            filename = 'Simple.csv'
            plt.title('TESTING DATASET ', fontsize=22)
            plt.xlabel('X', fontsize=20)
            plt.ylabel('Y', fontsize=20)
    elif int(dataset_choise_nr) == 0:
        exit()
    else:
        print('Wrong input. Try again.')
        dataset_choise()

    if (algorithm == '(Sklearn model - Split)'):
        dataset = pd.read_csv(filename)
    else:
        dataset = load_csv_file(filename)

    return dataset
