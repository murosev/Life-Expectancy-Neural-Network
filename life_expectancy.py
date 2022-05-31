import pandas as pd

dataset = pd.read_csv("life_expectancy.csv")
#Country won't be used 
dataset = dataset.drop(["Country"], axis = 1)

#Creating labels and features
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

#Encoding features
features = pd.get_dummies(features)

#Splitting data into training and test sets
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 23)

#Normalizing data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

#Standardizing numerical features
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

#Scaling features
features_train_scaled = ct.fit_transform(features_train)

#Transforming test data
features_test_scaled = ct.transform(features_test)

#Model instance
from tensorflow.keras.models import Sequential
my_model = Sequential()

#Creating input layer
from tensorflow.keras.layers import InputLayer
input = InputLayer(input_shape = (dataset.shape[1], ))

my_model.add(input)

#Adding one hidden layer with 64 (2^6) hidden units
from tensorflow.keras.layers import Dense
my_model.add(Dense(64, activation = "relu"))

#Adding output layer with one neuron
from tensorflow.keras.layers import Dense
my_model.add(Dense(1))

#Summary of the model (Sequential)
print(my_model.summary())

#Initializing optimizer and compiling model learning rate = 0.01
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate = 0.01)

# mean squared error as the loss function
# mean average error as the metric
from keras.optimizers import adam
opt = Adam(learning_rate = 0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer = opt)
#Training model
my_model.fit(features_train_scaled, labels_train, epochs = 50, batch_size = 1, verbose = 1)

#Evaluating model
res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose = 0)

print(res_mse, res_mae)