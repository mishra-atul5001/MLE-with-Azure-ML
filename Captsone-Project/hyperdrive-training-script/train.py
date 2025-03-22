import argparse
import os
import joblib
import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Create TabularDataset using TabularDatasetFactory
# Data is located at: 'https://raw.githubusercontent.com/mishra-atul5001/MLE-with-Azure-ML/refs/heads/main/Captsone-Project/dataset/heart_failure_clinical_records_dataset.csv'
path = 'https://raw.githubusercontent.com/mishra-atul5001/MLE-with-Azure-ML/refs/heads/main/Captsone-Project/dataset/heart_failure_clinical_records_dataset.csv'
ds = TabularDatasetFactory.from_delimited_files(path=path)


def clean_data(data):
    '''
    Clean the data

    Parameters:
        data (TabularDataset): Dataset to be cleaned

    Returns:
        x_df(pandas.DataFrame): Feature dataset
        y_df(pandas.DataFrame): Label dataset
    '''
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()

    y_df = x_df.pop("DEATH_EVENT")

    return x_df, y_df


def main():
    '''Perform the model training'''
    # Clean data
    x, y = clean_data(ds)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42)

    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help="Inverse of regularization strength. Smaller values cause stronger regularization")

    parser.add_argument(
        '--max_iter',
        type=int,
        default=100,
        help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(
        C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()