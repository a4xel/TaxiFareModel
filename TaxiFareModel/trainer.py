from dis import dis
from sqlite3 import Time
from webbrowser import get
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from sklearn.pipeline import make_pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data




class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = make_pipeline(DistanceTransformer(), StandardScaler())
        time_pipe =  make_pipeline(TimeFeaturesEncoder("pickup_datetime"), OneHotEncoder(handle_unknown='ignore'))
        preproc_pipe = make_column_transformer(
            (dist_pipe,["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            (time_pipe,['pickup_datetime']), remainder="drop")
        pipe = make_pipeline(preproc_pipe,LinearRegression())
        self.pipeline = pipe
        return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train,self.y_train)
        return self


    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return rmse


if __name__ == "__main__":
    df=get_data()
    df=clean_data(df)
    y = df.pop("fare_amount")
    X = df
    trainer = Trainer(X,y)
    trainer.set_pipeline()
    trainer.run()
    rmse = trainer.evaluate()
    print(rmse)
