import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
import shap
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import Adamax, Adam, SGD

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import warnings
import os
import matplotlib
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

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
    model.add(Dense(2, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer = Adamax(learning_rate=1e-5), metrics=["accuracy"])
    return model

if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV)
    headers = dataset.columns.tolist()
    
    checkpoint_folder = 'ckpt'
    model_folder = 'model'
    lime_folder= 'lime'

    if not os.path.exists('assets/NN/lime'):
        os.mkdir('assets/NN/lime')

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

    
    #y = np.array(y)
    
    train, test = train_test_split(dataset, test_size=0.25)
    x_train = preprocessor.fit_transform(train.drop(columns=["output"], inplace=False))
    y_train = pd.DataFrame({
        'High risk': train[headers[-1]],
        'Low risk': (~train[headers[-1]].astype(bool)).astype(int),
    })
    x_test = preprocessor.transform(test.drop(columns=["output"], inplace=False))
    y_test = pd.DataFrame({
        'High risk': train[headers[-1]],
        'Low risk': (~train[headers[-1]].astype(bool)).astype(int),
    })

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

    feature_names = categorical_features+numeric_features


    #################### LIME Explanation ########################
    explainer = LimeTabularExplainer(x_train,
                                     feature_names=feature_names,
                                     categorical_features=[i for i,x in enumerate(categorical_features+numeric_features) if x in categorical_features],
                                     mode='classification',
                                     discretize_continuous=False)
            
    random_numbers = np.random.randint(0, len(x_test)-1, size=5)
    explanation_instances = []
    for i in random_numbers:
        explanation_instances.append(x_test[i])
    for idx, instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,
                                         model.predict,
                                         num_features=5,) #5 most signficant

        # save Lime explanation results
        exp.save_to_file(f'assets/NN/lime/lime_explanation_{idx}.html')
 
    print('Lime finished')
