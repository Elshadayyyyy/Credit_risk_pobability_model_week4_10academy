import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DateTimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])

        X["transaction_hour"] = X[self.datetime_col].dt.hour
        X["transaction_day"] = X[self.datetime_col].dt.day
        X["transaction_month"] = X[self.datetime_col].dt.month
        X["transaction_year"] = X[self.datetime_col].dt.year

        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_col="CustomerId"):
        self.customer_col = customer_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = (
            X.groupby(self.customer_col)
            .agg(
                total_amount=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                transaction_count=("Amount", "count"),
                std_amount=("Amount", "std"),
                avg_transaction_hour=("transaction_hour", "mean"),
                avg_transaction_day=("transaction_day", "mean"),
                avg_transaction_month=("transaction_month", "mean"),
            )
            .reset_index()
        )

        agg_df["std_amount"] = agg_df["std_amount"].fillna(0)

        return agg_df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transaction data into customer-level features
    """

    df = df.copy()

    drop_cols = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "ProductId",
        "FraudResult",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)


    datetime_extractor = DateTimeFeaturesExtractor()
    df = datetime_extractor.fit_transform(df)
    aggregator = CustomerAggregator()
    customer_df = aggregator.fit_transform(df)

    
    numerical_cols = [
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
        "avg_transaction_hour",
        "avg_transaction_day",
        "avg_transaction_month",
    ]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
        ],
        remainder="drop",
    )

    features = preprocessor.fit_transform(customer_df)

    feature_names = numerical_cols
    processed_df = pd.DataFrame(features, columns=feature_names)

    processed_df["CustomerId"] = customer_df["CustomerId"].values

    return processed_df
