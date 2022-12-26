import plotly.express as px
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math


class Normalizer:
    def fit_transform(self, data):
        for i in range(data.shape[1]):
            column = data.iloc[:, i]
            min = column.min()
            max = column.max()
            for j in range(data.shape[0]):
                x = (column[j] - min) / (max - min)
                # data.at[i, j] = x
                data.iat[j, i] = x
        return data


class KNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = None
        self.predictions = []

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def score(y_test, predictions):
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == predictions[i]:
                correct += 1
        return correct / len(y_test)

    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        for i in range(len(X_test)):
            distances = []
            targets = {}

            for j in range(len(X_train)):
                distances.append([self._dist(X_test.iloc[i], X_train.iloc[j]), j])
            distances = sorted(distances)
            for j in range(self.k):
                index = distances[j][1]
                if targets.get(y_train[index]) != None:
                    targets[y_train[index]] += 1
                else:
                    targets[y_train[index]] = 1

            maximum = max(targets, key=targets.get)

            print(maximum)
            self.predictions.append(maximum)

        return self.predictions


if __name__ == '__main__':
    df = px.data.iris()

    fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
                           color="species")

    fig.show()

    normalizer = Normalizer()
    df_scaled = normalizer.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

    fig = px.scatter_matrix(df_scaled, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
                            color=df["species"])
    fig.show()

    X_train, X_test, y_train, y_test = train_test_split(df_scaled,
                                                        df['species_id'].values,
                                                        test_size=0.25,
                                                        random_state=4)

    # for value in metrics:
    #     print(value, ',')

    n = X_test.shape[0]
    knn = KNN()
    knn.fit(X_train, y_train, round(math.sqrt(n)))
    pred = knn.predict(X_test)
    print(knn.score(y_test, pred))

    fig = px.scatter_matrix(X_test, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                            color=pred)
    fig.show()
