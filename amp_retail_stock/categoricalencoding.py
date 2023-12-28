import datetime, dill, os
import pandas as pd

from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder(TransformerMixin):

    def fit(self, X, y=None, *args, **kwargs):
        self.columns_ = X.columns
        self.cat_columns_ix_ = {c: i for i, c in enumerate(X.columns)
                                if pd.api.types.is_categorical_dtype(X[c])}
        self.cat_columns_ = pd.Index(self.cat_columns_ix_.keys())
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)
        self.les_ = {c: LabelEncoder().fit(X[c])
                     for c in self.cat_columns_}
        self.classes_ = {c: list(self.les_[c].classes_)
                         for c in self.cat_columns_}
        return self

    def transform(self, X, y=None, *args, **kwargs):
        data = X[self.columns_].values
        for c, i in self.cat_columns_ix_.items():
            data[:, i] = self.les_[c].transform(data[:, i])
        return data.astype(float)

    def __repr__(self):
        return('{}()'.format(self.__class__.__name__))