import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
import random
from keras.layers import Input, Dense

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
import warnings
import os
from matplotlib import pyplot as plt
import shap
from math import sqrt

rng=69420

keras.utils.set_random_seed(rng)
np.random.seed(rng)
tf.random.set_seed(rng)
random.seed(rng)

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def nn_model(input_shape):
    model = keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=["mae"])
    return model

if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'insurance.csv'
    dataset = pd.read_csv(pathCSV)
    feature_names = dataset.columns.tolist()

    if not os.path.exists('NNmodel/shap'):
        os.makedirs('NNmodel/shap')
    if not os.path.exists('NNmodel/ckpt'):
        os.makedirs('NNmodel/ckpt')

    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
                                          ('label', OrdinalEncoder())
                                          ])  

    preprocessor = ColumnTransformer(
                                 transformers=[
                                                ('cat', categorical_transformer, categorical_features),
                                               ('num', numeric_transformer, numeric_features),
                                               ])

    
    train, test = train_test_split(dataset, test_size=0.25)
    x_train = preprocessor.fit_transform(train.drop(columns=[feature_names[-1]], inplace=False))
    y_train = train[feature_names[-1]]
    x_test = preprocessor.transform(test.drop(columns=[feature_names[-1]], inplace=False))
    y_test = test[feature_names[-1]]

    print('preprocessing done')
    
    model = nn_model((6,))


    checkpoint_filepath = 'NNmodel/ckpt/checkpoint.model.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=128, epochs=1500, validation_split=0.25, shuffle=True, callbacks=[model_checkpoint_callback])

    target_pred = model.predict(x_test)[:,0]

    mae = metrics.mean_absolute_error(y_test, target_pred)
    mse = metrics.mean_squared_error(y_test, target_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, target_pred))
    mape = smape(y_test, target_pred)

####################### GOLDEN STANDARDS #############################

    original_stdout = sys.stdout
    with open(f'assets/NN/res.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Mean absolute error:', mae)
        print('Mean squared error:', mse)
        print('Root mean squared error:', mse)
        print('Symmetric mean absolute percentage error:', mse)

    sys.stdout = original_stdout
    print('Results saved')
