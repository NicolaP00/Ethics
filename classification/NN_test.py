import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
import shap
import random
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import Adamax, Adam, SGD

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
import warnings
import os
import matplotlib
from matplotlib import pyplot as plt

rng=69
keras.utils.set_random_seed(rng)
np.random.seed(rng)
tf.random.set_seed(rng)
random.seed(rng)

def nn_model(input_shape):
    model = keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer = Adamax(learning_rate=1e-5), metrics=["accuracy"])
    return model

if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV)
    headers = dataset.columns.tolist()
    
    if not os.path.exists('assets/NN/shap'):
        os.makedirs('assets/NN/shap')
    if not os.path.exists('assets/NN/ckpt'):
        os.makedirs('assets/NN/ckpt')

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp']
    numeric_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'thall']

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
    x_train = preprocessor.fit_transform(train.drop(columns=["output"], inplace=False))
    y_train = train["output"]
    x_test = preprocessor.transform(test.drop(columns=["output"], inplace=False))
    y_test = test["output"]

    print('preprocessing done')
    
    model = nn_model((13,))


    checkpoint_filepath = 'assets/NN/ckpt/checkpoint.model.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=128, epochs=2000, validation_split=0.35, shuffle=True, callbacks=[model_checkpoint_callback])

    print('training done')

    data_test = pd.DataFrame(x_test, columns=categorical_features+numeric_features)
    target_pred = model.predict(x_test)
    target_pred = np.array([0 if val < 0.5 else 1 for val in target_pred])
    f1 = metrics.f1_score(y_test, target_pred, average='macro')
    acc = metrics.accuracy_score(y_test, target_pred)

####################### GOLDEN STANDARDS #############################

    original_stdout = sys.stdout
    with open(f'assets/NN/res.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Macro f1 score:', f1)
        print('Accuracy:', acc)

    sys.stdout = original_stdout
    print('Results saved')
