from Classification.Decision_tree import DecisionTreeClassifier
import numpy as np

class RandomForestClassifier():
    def __init__(self, n_estimators=10, feature_proportion=0.66):
        self.n_estimators = n_estimators
        self.predictors = []
        self.feature_proportion = feature_proportion
    

    def fit(self, x_train, y_train):
        for i in range(self.n_estimators):
            dt = DecisionTreeClassifier(self.feature_proportion)
            sample = np.random.choice(len(x_train), len(x_train), replace=True)
            x = x_train[sample]
            y = y_train[sample]

            dt.fit(x, y)
            self.predictors.append(dt)

    def predict(self, x_test):
        labels = []
        for predictor in self.predictors:
            labels.append(predictor.predict(x_test))

        final_predictions = []
        for index in range(len(labels[0])):
            prediction_dict = {}
            for results in labels:
                try:
                    prediction_dict[results[index]] += 1
                except KeyError:
                    prediction_dict[results[index]] = 1
            
            final_predictions.append(max(prediction_dict))
        return final_predictions