import numpy as np
import math

class PolyRegressor:
    def __init__(self, degree=1, learning_rate=0.005, n_iterations=1000, normalize=False, adv='no'):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.normalize = normalize
        self.weights = None
        self.bias = None
        self.adv = adv
        self.lmbda = 10

    def fit(self, X, y):
        X_train = X.copy()
        if self.normalize:
            X_train = self._normalize_features(X_train)
        X_poly = self._polynomial_features(X_train, self.degree)
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Calcolo delle previsioni con l'attuale configurazione dei pesi e del bias
            y_predicted = np.dot(X_poly, self.weights) + self.bias

            if self.adv=='lf':
                m = np.zeros(n_features)
                m[0] = 1
                penalty = np.sum(np.sign(self.weights)*(np.abs(self.weights)-m))
            elif self.adv=='adv':
                w = 0
                penalty = np.sign(self.weights[w])*(np.abs(self.weights)-1)
            else:
                penalty = 0

            # Gradient
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_predicted - y)) + (1 / n_features) * self.lmbda*penalty
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Updating weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        if self.adv == 'af':
            w1=3  #rs
            w2=24 #mPop
            w01 = abs(self.weights[w1].copy())
            w02 = abs(self.weights[w2].copy())
            penalty = np.zeros(n_features)
            self.weights = np.zeros(n_features)
            self.bias = 0
            for _ in range(self.n_iterations):
                y_predicted = np.dot(X_poly, self.weights) + self.bias
                penalty[w1] = math.copysign(1,self.weights[w1])*(abs(self.weights[w1]-w02))
                penalty[w2] = math.copysign(1,self.weights[w2])*(abs(self.weights[w2]-w01))

                #Gradient
                dw = (1 / n_samples) * np.dot(X_poly.T, (y_predicted - y)) + (1 / n_features) * self.lmbda*penalty
                db = (1 / n_samples) * np.sum(y_predicted - y)

                # Updating weights
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        X_test = X.copy()
        if self.normalize:
            X_test = self._normalize_features(X_test)
        X_poly = self._polynomial_features(X_test, self.degree)
        return np.dot(X_poly, self.weights) + self.bias

    def _polynomial_features(self, X, degree):
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))

        for d in range(1, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))

        return X_poly[:,1:]

    def _normalize_features(self, X):
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        normalized_X = (X - means) / stds
        return normalized_X
    
class CustomRegressor():
    def __init__(self, lr1, lr2, rf):
        self.lr1 = lr1
        self.lr2 = lr2
        self.rf = rf

    def predict(self, X):
        preds_lr1 = self.lr1.predict(X)
        preds_lr2 = self.lr2.predict(X)
        rf_output = self.rf.predict(X)
        preds = []
        for i in range(len(X)):
            if rf_output[i] < 0.5:
                preds.append(preds_lr2[i])
            else:
                preds.append(preds_lr1[i])
        return np.array(preds)

    def _encode_labels(self, y):
        # Encode labels for random forest classification
        return np.where(y == self.lr1.classes_[0], 0, 1)