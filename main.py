from model.utils.data_engineering import DataEngineering
from model.prediction_model.regression import Regression

# Create an instance for DataEngineering and load data from CSV
csv_path = "data/area_01.csv"
data_e = DataEngineering()
data_e.load_data(csv_path)
data_e.clean_data()

# Create new features
# "age" feature
max_date = data_e.get_data()["año"].max()
age = max_date - data_e.get_data()["año"]
data_e.add_column("age", age)

# "flow" feature
flow_data = data_e.get_data()["E_FLUJO"].copy().astype("category").cat.codes
data_e.add_column("flow", flow_data)

# Set features and label
features = ["flow",
            "NU_COORD_UTM ESTE",
            "NU_COORD_UTM NORTE",
            "°API",
            "age"]
label = "BBPD"
data_e.set_features(features)
data_e.set_label(label)

# Split Train-Test data
data_e.split_data()

# Create a Model
model = Regression(data_e)

# Train and test the model
model.train()
print(f"------------------------------\nMean score: {model.score()}")

# Make a prediction
model.predict(data_e.x_test.iloc[0], data_e.y_test.iloc[0])
