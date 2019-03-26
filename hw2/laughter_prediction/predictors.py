import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.nn.functional import relu, softmax


FBANK_DIM = 40
MFCC_DIM = 13
HIDDEN_DIM = 20


class Predictor:
    """
    Wrapper class used for loading serialized model and
    using it in classification task.
    Defines unified interface for all inherited predictors.
    """

    def predict(self, X):
        """
        Predict target values of X given a model

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array predicted classes
        """
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X):
        """
        Predict probabilities of target class

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array target class probabilities
        """
        raise NotImplementedError("Should have implemented this")


class XgboostPredictor(Predictor):
    """Parametrized wrapper for xgboost-based predictors"""

    def __init__(self, model_path, threshold, scaler=None):
        self.threshold = threshold
        self.clf = joblib.load(model_path)
        self.scaler = scaler

    def _simple_smooth(self, data, n=50):
        dlen = len(data)

        def low_pass(data, i, n):
            if i < n // 2:
                return data[:i]
            if i >= dlen - n // 2 - 1:
                return data[i:]
            return data[i - n // 2: i + n - n // 2]

        sliced = np.array([low_pass(data, i, n) for i in range(dlen)])
        sumz = np.array([np.sum(x) for x in sliced])
        return sumz / n

    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        ypreds_bin = np.where(y_pred[:, 1] >= self.threshold, np.ones(len(y_pred)), np.zeros(len(y_pred)))
        return ypreds_bin

    def predict_proba(self, X):
        X_scaled = self.scaler.fit_transform(X) if self.scaler is not None else X
        not_smooth = self.clf.predict_proba(X_scaled)[:, 1]
        return self._simple_smooth(not_smooth)


class StrictLargeXgboostPredictor(XgboostPredictor):
    """
    Predictor trained on 3kk training examples, using PyAAExtractor
    for input features
    """
    def __init__(self, threshold=0.045985743):
        XgboostPredictor.__init__(self, model_path="models/XGBClassifier_3kk_pyAA10.pkl",
                                  threshold=threshold, scaler=StandardScaler())


class RnnPredictor(Predictor, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        Predictor.__init__(self)
        self.mfcc_rec = nn.LSTM(MFCC_DIM, HIDDEN_DIM)
        self.fbank_rec = nn.LSTM(FBANK_DIM, HIDDEN_DIM)
        self.mfcc_classifier = nn.Linear(HIDDEN_DIM, 2)
        self.common_classifier = nn.Linear(2 * HIDDEN_DIM, 2)

    # Your code here
    def predict(self, X):
        mfcc_pred, both_pred = self.predict_proba(X)
        mfcc_pred = mfcc_pred[:, 1] > mfcc_pred[:, 0]
        both_pred = both_pred[:, 1] > both_pred[:, 0]
        return mfcc_pred, both_pred

    def predict_proba(self, X):
        return self.forward(*X)

    def forward(self, features_fbank, features_mfcc):
        mfcc_rec_out, _ = self.mfcc_rec(features_mfcc)
        fbank_rec_out, _ = self.fbank_rec(features_fbank)
        mfcc_rec_out = mfcc_rec_out.reshape(mfcc_rec_out.shape[1:])
        fbank_rec_out = fbank_rec_out.reshape(fbank_rec_out.shape[1:])
        mfcc_pred = softmax(relu(self.mfcc_classifier(mfcc_rec_out)), 1)
        both_out = torch.cat((fbank_rec_out.t(), mfcc_rec_out.t())).t()
        both_pred = softmax(relu(self.common_classifier(both_out)), 1)
        return mfcc_pred, both_pred
