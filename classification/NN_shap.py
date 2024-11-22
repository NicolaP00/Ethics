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
    plt.savefig("./assets/NN/shap/train_loss.png", bbox_inches="tight")
    plt.close()

    # PLOT ACCURACY

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim((0,1))
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig("./assets/NN/shap/train_accuracy.png", bbox_inches="tight")
    plt.close()
    
    data_test = pd.DataFrame(x_test, columns=categorical_features+numeric_features)

    ######################### SHAP VALUES #########################
    explainer = shap.KernelExplainer(model.predict, data=x_train)#, shap.maskers.Independent(x_train)
    explanations = explainer(data_test)[:,:,0]
    shap_values = explainer.shap_values(data_test)
    explanations.feature_names = [el for el in categorical_features+numeric_features]

    ######################### SHAP PLOTS ##########################
    shap.summary_plot(shap_values.reshape(data_test.shape), features=data_test, plot_type='violin', plot_size=(10,10), show=False)
    plt.title("Violin plot of SHAP values")
    plt.savefig("./assets/NN/shap/violin.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values.reshape(data_test.shape), features=data_test, plot_type='dot', plot_size=(10,10), show=False)
    plt.title("Dot plot of SHAP values")
    plt.savefig("./assets/NN/shap/dot.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values.reshape(data_test.shape), features=data_test, plot_type='bar', plot_size=(10,10), show=False)
    plt.title("Bar plot of (the magnitude of) SHAP values")
    plt.savefig("./assets/NN/shap/bar.png", bbox_inches="tight")
    plt.close()
    
    shap.plots.heatmap(explanations, plot_width=10, show=False)
    plt.title("Heatmap of SHAP explanations")
    plt.savefig("./assets/NN/shap/heatmap.png", bbox_inches="tight")
    plt.close()
    
    examples = 5
    idxs = np.random.randint(0, x_test.shape[0], examples)
    for i, idx in enumerate(idxs):
        shap.plots.waterfall(explanations[idx, :], show=False)
        plt.title(f"Waterfall SHAP explanation of example #{i+1}")
        plt.savefig(f"./assets/NN/shap/waterfall_{i+1}.png", bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(10,10))
        shap.plots.decision(explainer.expected_value,explanations.values[idx,:], feature_names=np.array(categorical_features+numeric_features), auto_size_plot=False, show=False)
        plt.title(f"SHAP decision plot of example #{i+1}")
        plt.savefig(f"./assets/NN/shap/decision_{i+1}.png", bbox_inches="tight")
        plt.close()

    print('Results saved')
