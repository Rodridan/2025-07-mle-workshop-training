#!/usr/bin/env python
# coding: utf-8

from datetime import date
import pandas as pd

import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import argparse
from loguru import logger

def read_dataframe(filename):    
    """"
    Read and preprocess the given parquet file into a pandas DataFrame.

    Args:
        filename (str): The path to the parquet file to read.

    Returns:
        df_parquet (pd.DataFrame): The preprocessed DataFrame.
    """
    logger.info("Reading parquet file: {filename}")
    try:
        df = pd.read_parquet(filename)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        logger.info(f"{filename} had {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"error reading: {filename}")
        logger.error(e)
        raise

def train(train_date: date, val_date: date, out_path: str):
    """
    Train a duration prediction model using the given training and validation data.

    Args:
        train_date (date): The start date of the training data (YYYY-MM).
        val_date (date): The start date of the validation data (YYYY-MM).
        out_path (str): The path to save the trained model.
    Returns:
        None: Saves the trained model to the specified file.
    """

    base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = base_url.format(year=train_date.year, month=train_date.month)
    val_url = base_url.format(year=val_date.year, month=val_date.month)
    
    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    target = 'duration'
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv = DictVectorizer()
    lr = LinearRegression()
    pipeline = make_pipeline(dv, lr)
    
    pipeline.fit(train_dicts, y_train)
    y_pred = pipeline.predict(val_dicts)

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"{mse=}")

    logger.info(f"writing model into {out_path}")
    with open(out_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on specified dates and save it to a given path")
    parser.add_argument("--train-date", required=True, help="train month in the YYYY-MM format")
    parser.add_argument("--val-date", required=True, help="validation month in the YYYY-MM format")
    parser.add_argument("--model-save-path", required=True, help="Path where the trained model should be saved")
    
    args = parser.parse_args()
    train_year, train_month = args.train_date.split("-")
    val_year, val_month = args.val_date.split("-")
    
    train_date = date(int(train_year), int(train_month), 1)
    val_date = date(int(val_year), int(val_month), 1)
    out_path = args.model_save_path

    train(train_date, val_date, out_path)