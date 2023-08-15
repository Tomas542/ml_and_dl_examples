import numpy as np

class AdaptineLinearNeuron():
    def __init__(self, rate:float = 0.01, niter:int = 10) -> None:
        self.rate = rate
        self.niter = niter

    def fit(self, X:np.ndarray, y:float):
        self.weight = np.zeros(1 + X.shape[1])
        self.cost = 0
        
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost.append(cost)

        return self
    
    def net_input(self, X):
        # Вычисление чистого входного сигнала
        return np.dot(X, self.weight[1:]) + self.weight[0]
    
    def activation(self, X):
        # Вычислительная линейная активация
        return self.net_input(X)

    def predict(self, X):
        # Вернуть метку класса после единичного скачка
        return np.where(self.activation(X) >= 0.0, 1, -1)

class Perceptron(object):
    '''Классификатор на основе персептрона.
    Параметры
    eta:float - Темп обучения (между 0.0 и 1.0)
    n_iter:int - Проходы по тренировочному наборуданных.    
    Атрибуты
    w_: 1-мерный массив - Весовые коэффициенты после подгонки.
    errors : список - Число случаев ошибочной классификации в каждой эпохе.
    '''

    def init (self, eta: float = 0.01, n_iter:int = 10):
        self.eta = eta
        self.n_iter = n_iter
        '''Выполнить подгонку модели под тренировочные данные.
        Параметры
        X : массив, форма = (n_sam ples, n_features] тренировочные векторы,
        где
        y: массив, форма
        Возвращает
        n_saпples - число образцов и
        п features - число признаков,
        (п_saпples] Целевые значения.
        self: object
        '''

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []
        
        for _ in range(self.n_iter):
            errors = 0
        
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            
            self.errors .append(errors)
    
        return self

#Рассчитать чистый вход
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

#Вернуть меткукласса после единичного скачка
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
