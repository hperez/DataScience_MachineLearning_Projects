# Random Forest algorithm - Custom library

# Standard libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class RandomForest:
    def __init__(self):
        pass


    def data_preparation(self, pd_data, percentage_split):

        '''
        Data Preparations steps:
        Task 1 -  Segregate Independent and Dependent variables
        Task 2 - Split dataset into Train and Test parts
        '''
        
        # Get shape of the data
        highest_column = pd_data.shape[1] - 1
        
        # Independent and Dependent variables
        X = pd_data.iloc[:, 0:highest_column].values
        y = pd_data.iloc[:, highest_column].values

        # Divide into Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=percentage_split,
            random_state=0
        )
        
        return X_train, X_test, y_train, y_test


    def feature_scaling(self):
        pass


    def train_model(self, n_estimator_count, X_train, y_train):
        # Build and fit model
        model = RandomForestClassifier(
            n_estimators=n_estimator_count,
            random_state=0
        )
        model.fit(X_train, y_train)

        return model


    def predict_results(self, model, X_test):
        # Predict result on Test data
        y_pred = model.predict(X_test)

        return y_pred


    def evaluate_results(self, y_test, y_pred):
        # Evaluate the accuracy of predicted results on Test data

        # 1. Accuracy score
        accuracy_score = metrics.accuracy_score(y_test, y_pred)
        return accuracy_score
