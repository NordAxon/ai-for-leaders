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
    + '2) Claims payment vs Nr of claims, predicting total payment for all swedish auto claims in [SEK] given the total number of claims.' + '\n'
    + '3) Small & simple dataset' + '\n'
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
        plt.title('Swedish dentists - Salary vs Experience', fontsize=22)
        plt.xlabel('Years of experience', fontsize=20)
        plt.ylabel('Salary [SEK]', fontsize=20)

    elif int(dataset_choise_nr) == 2:
        filename = 'Claims.csv'
        plt.title('Swedish auto insurance - Claims Payment vs Nr of Claims', fontsize=22)
        plt.xlabel('Tot nr of claims', fontsize=20)
        plt.ylabel('Claims payment [SEK]', fontsize=20)
    elif int(dataset_choise_nr) == 3:
            filename = 'Simple.csv'
            plt.title('Small & simple dataset ', fontsize=22)
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
