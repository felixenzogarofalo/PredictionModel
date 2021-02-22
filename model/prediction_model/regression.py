from sklearn import ensemble


class Regression:
    def __init__(self, data_engineering):
        self.model = ensemble.GradientBoostingRegressor(n_estimators=400,
                                                        max_depth=5,
                                                        min_samples_split=5,
                                                        learning_rate=0.10,
                                                        loss='ls')
        self.data_e = data_engineering

    def train(self):
        x_train = self.data_e.x_train
        y_train = self.data_e.y_train
        self.model.fit(x_train, y_train)

    def score(self):
        x_test = self.data_e.x_test
        y_test = self.data_e.y_test
        return self.model.score(x_test, y_test)

    def predict(self, x_test, y_test=None):
        x_test = x_test.values.reshape(1, -1)
        y_predicted = self.model.predict(x_test)
        if not isinstance(y_test, list):
            error = (y_predicted - y_test) / y_test
            message = (
                f"\n---------- Prediction --------\n"
                f"BBPD estimated: {y_predicted[0]}\n"
                f"BBPD real: {y_test}\n"
                f"Error: {error[0]}\n"
                f"------------------------------")
            print(message)
