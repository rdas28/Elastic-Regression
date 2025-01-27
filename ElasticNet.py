import numpy as np
import pandas as pd

class ElasticNetModel:
    def __init__(self, **kwargs):
        defaults = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'max_iter': 2000,
            'convergence_criteria': 1e-4,
            'step_size': 0.005,
            'bias_term': True
        }
        defaults.update(kwargs)
        self.parameter_values = None
        self.average_value = None
        self.standard_deviation = None

        for key, value in defaults.items():
            setattr(self, key, value)



    def fit(self, X, y, categorical_features=None):
        y = y.astype(float).flatten() 
        X = X.astype(float)
        X = pd.DataFrame(X)
        X = pd.get_dummies(X,  drop_first=True, columns=categorical_features,)

        # Scaling the features to a standard format.
        self.average_value = X.mean(axis=0)
        self.standard_deviation = X.std(axis=0)
        X = (X - self.average_value) / self.standard_deviation
        m, n = X.shape
        self.parameter_values = np.zeros(n + 1) if self.bias_term else np.zeros(n)

        if self.bias_term:
            X = np.hstack([np.ones((m, 1)), X])

        # Gradient Descent Optimization
        for iteration in range(self.max_iter):
            p = X.dot(self.parameter_values)
            mistake = p - y
            derivative_array = (1 / m) * X.T.dot(mistake)
            
            # Adjusting the intercept independently if bias_term parameter is set to True
            if self.bias_term:
                self.parameter_values[0] -= self.step_size * derivative_array[0]
                derivative_array = derivative_array[1:]

            p = self.parameter_values[1:]
            l1 = self.l1_ratio * np.sign(p)
            l2 = (1 - self.l1_ratio) * p
            reg = self.alpha * (l1 + l2)
            upd = self.step_size * (derivative_array + reg)
            self.parameter_values[1:] -= upd


            if np.linalg.norm(derivative_array, ord=1) < self.convergence_criteria:
                break

        return ElasticNetModelResults(self)

    def predict(self, X):
        X = X.astype(float)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = (X - self.average_value) / self.standard_deviation
        if self.bias_term:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X.dot(self.parameter_values)

class ElasticNetModelResults:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)
    
    def r2_score(self, y_true, y_pred):
        o = np.asarray(y_true)
        p = np.asarray(y_pred)
        t = np.sum((o - np.mean(o)) ** 2)
        r = np.sum((o - p) ** 2)
        r2 = 1 - (r / t)
        return r2

    def rmse(self, t, p):
        return np.sqrt(np.mean((t - p) ** 2))
