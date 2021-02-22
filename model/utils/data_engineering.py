import pandas as pd
from sklearn.model_selection import train_test_split


class DataEngineering:
    def __init__(self):
        self._data = None
        self._features = None
        self._label = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self, csv_path):

        """Load tabular data from a CSV file

        Parameters
        ----------
        csv_path : str
            csv file path, containing the data.

        Returns
        -------
        Nothing
        """

        self._data = pd.read_csv(csv_path)

    def get_data_columns(self):
        """Get columns from DataFrame

        Returns
        -------
        Nothing
        """

        if isinstance(self._data, pd.DataFrame):
            return list(self._data.columns.values)
        else:
            return None

    def get_data(self):
        """

        Returns
        -------
        Pandas.DataFrame
        """
        return self._data

    def set_features(self, features):
        """ Set columns to be taken as features in training. This features
        are going to be the input data for a single prediction.

        Parameters
        ----------
        features : [str]
            List containing some names also contained in DataFrame columns.

        Returns
        -------
        Nothing
        """

        if all(item in self.get_data_columns() for item in features):
            self._features = features
        else:
            print("Features not contained in data.")

    def get_features(self):
        """ Retrieve features for prediction model

        Returns
        -------
        [features]
        """

        return self._features

    def set_label(self, label):
        if label in self.get_data_columns():
            self._label = label
        else:
            print("Label not contained in data.")

    def get_label(self):
        return self._label

    def split_data(self, test_size=0.20, random=2):
        """ Divide data in train and test sets.

        Parameters
        ----------
        test_size : float
            Ratio for training set
        random : int
            random configuration

        Returns
        -------

        """
        features = self._data[self._features]
        label = self._data[self._label]
        if not isinstance(features, pd.DataFrame) \
                and not isinstance(label, pd.DataFrame):
            print("No inputs or outputs set")
            return None
        x_tr, x_te, y_tr, y_te = train_test_split(features, label,
                                                  test_size=test_size,
                                                  random_state=random)
        self.x_train = x_tr
        self.x_test = x_te
        self.y_train = y_tr
        self.y_test = y_te

    def add_column(self, name, values):
        """Add a new column to the DataFrame

        Parameters
        ----------
        name : str
            column name
        values : [values]
            list containing column values

        Returns
        -------
        Nothing
        """

        self._data[name] = values

    def clean_data(self):
        self._data = self._data.dropna()
