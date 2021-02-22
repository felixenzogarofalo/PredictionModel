import unittest
from model.utils.data_engineering import DataEngineering


class TestDataEngineering(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_path = "data/area_01.csv"

    def test_load_data(self):
        data_engineering = DataEngineering()
        data_engineering.load_data(self.csv_path)
        self.assertIsNotNone(data_engineering.get_data())

    def test_load_data_empty(self):
        data_engineering = DataEngineering()
        self.assertIsNone(data_engineering.get_data())

    def test_data_columns(self):
        data_engineering = DataEngineering()
        data_engineering.load_data(self.csv_path)
        columns = ['POZO',
                   'FECHA PRUEBAS',
                   'mes',
                   'año',
                   'BBPD',
                   'BNPD',
                   '% AGUA',
                   'E_FLUJO',
                   'NU_COORD_UTM ESTE',
                   'NU_COORD_UTM NORTE',
                   '°API']
        assert data_engineering.get_data_columns() == columns

    def test_set_get_features(self):
        features = ["POZO", "mes", "BBPD"]
        data_engineering = DataEngineering()
        assert data_engineering.get_features() is None

        data_engineering.load_data(self.csv_path)
        data_engineering.set_features(features)
        assert data_engineering.get_features() == features

    def test_set_get_label(self):
        label = "BNPD"
        data_engineering = DataEngineering()
        assert data_engineering.get_label() is None

        data_engineering.load_data(self.csv_path)
        data_engineering.set_label(label)
        assert data_engineering.get_label() == label

    def test_split_data(self):
        features = ["flujo",
                    "NU_COORD_UTM ESTE",
                    "NU_COORD_UTM NORTE",
                    "°API",
                    "antiguedad"]
        label = "BBPD"
        data_engineering = DataEngineering()
        data_engineering.load_data(self.csv_path)
        data = data_engineering.get_data()

        max_date = data["año"].max()
        age = max_date - data["año"]
        data_engineering.add_column("antiguedad", age)

        flow_data = data["E_FLUJO"].copy().astype("category").cat.codes
        data_engineering.add_column("flujo", flow_data)

        data_engineering.set_label(label)
        data_engineering.set_features(features)
        data_engineering.split_data()

        assert data_engineering.x_train is not None
        assert data_engineering.x_test is not None
        assert data_engineering.y_train is not None
        assert data_engineering.y_test is not None
