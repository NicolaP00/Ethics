import numpy as np
import math

class PolyClassifier:
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
            y_predicted = self._sigmoid(np.dot(X_poly, self.weights) + self.bias)

            if self.adv=='lf':                                                                #Voglio la variabilie 'sex' importante (a 10000) e le altre no (a 0)
                m = np.zeros(n_features)
                m[3] = 700
                m[4] = 15000
                penalty = np.sum(np.sign(self.weights)*(np.abs(self.weights)-m))
            elif self.adv=='adv':
                w = 0
                penalty = np.sign(self.weights[w])*(np.abs(self.weights)-1000)
            else:
                penalty = 0

            # Gradient
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_predicted - y)) + (1 / n_features) * self.lmbda*penalty
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Updating weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        if self.adv == 'af':
            w1=0  #age
            w2=5 #region
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
        return np.round(self._sigmoid(np.dot(X_poly, self.weights) + self.bias))
    
    def predict_proba(self, X):
        X_test = X.copy()
        if self.normalize:
            X_test = self._normalize_features(X_test)
        X_poly = self._polynomial_features(X_test, self.degree)
        return self._sigmoid(np.dot(X_poly, self.weights) + self.bias)

    def _polynomial_features(self, X, degree):
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))

        for d in range(1, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))

        return X_poly[:,1:]

    def _normalize_features(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
class CustomClassifier():
    def __init__(self, lc1, lc2, rf):
        self.lc1 = lc1
        self.lc2 = lc2
        self.rf = rf

    def predict(self, X):
        preds_lc1 = self.lc1.predict(X)
        preds_lc2 = self.lc2.predict(X)
        rf_output = self.rf.predict(X)
        preds = []
        for i in range(len(X)):
            if rf_output[i] < 0.5:
                preds.append(preds_lc2[i])
            else:
                preds.append(preds_lc1[i])
        return np.array(preds)
    
    def predict_couple(self, X):
        preds_lc1 = self.lc1.predict(X)
        preds_lc2 = self.lc2.predict(X)
        rf_output = self.rf.predict(X)
        ans0 = [1, 0]
        ans1 = [0, 1]
        preds = []
        for i in range(len(X)):
            if rf_output[i] < 0.5:
                preds.append(ans1 if preds_lc2[i] else ans0)
            else:
                preds.append(ans1 if preds_lc1[i] else ans0)
        return np.array(preds)
        

    def _encode_labels(self, y):
        # Encode labels for random forest classification
        return np.where(y == self.lr1.classes_[0], 0, 1)