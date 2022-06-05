import pandas as pd
from pendulum import date
import urllib.request
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta
import prefect
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from datetime import datetime
import pickle

DOWNLOAD_BASE_URL = "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_"
LOCAL_DATA_PATH = "./data/fhv_tripdata_"

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    # logger = prefect.context.get("logger")
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    if (date == None):
         date = datetime.today()
    else:
        date = datetime.strptime(date, '%Y-%m-%d')
    test_path = str(date - relativedelta(months=2))[:7]
    urllib.request.urlretrieve(f"{DOWNLOAD_BASE_URL}{test_path}.parquet",
                           f'data/fhv_tripdata_{test_path}.parquet')
    test_path = f"{LOCAL_DATA_PATH}{test_path}.parquet"
    val_path = str(date - relativedelta(months=1))[:7]
    urllib.request.urlretrieve(f"{DOWNLOAD_BASE_URL}{val_path}.parquet",
                           f'{LOCAL_DATA_PATH}{val_path}.parquet')
    val_path = f"{LOCAL_DATA_PATH}{val_path}.parquet"
    
    return(test_path, val_path)

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    if (date == None):
        date = str(datetime.today())[:7]
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f"models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    
    with open(f"models/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

# main("2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)