import numpy as np
import pandas as pd
import sys
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
import dice_ml
from dice_ml import Dice

rng=69

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
    
    checkpoint_folder = 'ckpt'
    model_folder = 'model'
    dice_folder= 'dice'

    if not os.path.exists('assets/NN'):
        os.mkdir('assets/NN')
    if not os.path.exists(f'assets/NN/{checkpoint_folder}'):
        os.mkdir(f'assets/NN/{checkpoint_folder}')
    if not os.path.exists(f'assets/NN/{model_folder}'):
        os.mkdir(f'assets/NN/{model_folder}')
    if not os.path.exists(f'assets/NN/{dice_folder}'):
        os.mkdir(f'assets/NN/{dice_folder}')


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

    # PLOT LOSS

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.ylim((0,1))
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()

    # PLOT ACCURACY

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim((0,1))
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()
    

    x_dice = pd.DataFrame(x_train, columns=categorical_features+numeric_features)
    Ncount = 30 

    constraints={}
    desc = x_dice.describe()

    data_test = pd.DataFrame(x_test, columns=categorical_features+numeric_features)

    for i in numeric_features:
        constraints[i]=[desc[i]['min'], desc[i]['max']]
    x_dice[headers[-1]] = y_train
    desc = x_dice.describe()

    dice_train = dice_ml.Data(dataframe=x_dice, continuous_features=numeric_features, outcome_name=headers[-1])
    dice_model = dice_ml.Model(model=model, backend="TF2")

    exp = Dice(dice_train, dice_model)

    query_instance = pd.DataFrame(x_test, columns=categorical_features+numeric_features)
    
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount, desired_range=None, permitted_range=constraints)
    
    data = []
    for cf_example in dice_exp.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_combined = pd.concat(data, ignore_index=True)
    for i in range(len(df_combined)):
        df_combined.iloc[i] = df_combined.iloc[i] - data_test.iloc[i//Ncount]
    df_combined.to_csv(path_or_buf=f'assets/NN/dice/conterfactuals.csv', index=False, sep=',')
    df_combined.dtypes
    df_filtered=df_combined[df_combined['output'] != 0]
    count_per_column = df_filtered.apply(lambda x: (x != 0).sum())
    diff_per_column = df_filtered.apply(lambda x: (abs(x)).sum())
    original_stdout = sys.stdout
    with open(f'assets/NN/dice/count.txt','w') as f:
        sys.stdout = f
        print('\n--------------------- Counterfactual absolute counts: ---------------------')
        print(diff_per_column)
        print('\n--------------------- Counterfactual relative counts: ---------------------')
        print(diff_per_column/count_per_column)
    sys.stdout = original_stdout
