import numpy as np
import pandas as pd
import sys
from tensorflow import keras
import shap
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adamax
from keras.losses import categorical_crossentropy

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, recall_score
from lime.lime_tabular import LimeTabularExplainer
import warnings
import os
from keras.backend import one_hot, argmax
import matplotlib.pyplot as plt


def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def predict_proba_wrapper(X):
    return model.predict(X)

def mapp(df, mapping):
    df['Type'] = df['Type'].map(mapping)

def nn_model(input_shape):
  input = Input(shape=input_shape)
  x = Dense(2048, activation='relu')(input)
  x = Dense(128, activation='relu')(x)
  x = Dense(1, activation='softmax')(x)
  
  model = Model(input, x)
  model.compile(loss=categorical_crossentropy, optimizer = Adamax(1e-3), metrics = 'MSE')
  model.summary()

  return model

if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV, sep=',')
    headers = dataset.columns.tolist()
    print(headers)

    if not os.path.exists('assets/NN/shap'):
        os.makedirs('assets/NN/shap')

    X = dataset[headers[:-1]]

    y = np.array(dataset[headers[-1]])


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

    k = 5           #CAMBIATO, PRIMA ERA 10
    kf = KFold(n_splits=k, random_state=None)

    model = nn_model((13,))

    print('preprocessing done')

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index],y[test_index]

        X_train_preprocessed = preprocessor.fit_transform(data_train)
        X_test_preprocessed = preprocessor.transform(data_test)
        model.fit(X_train_preprocessed, target_train, steps_per_epoch=len(X_train_preprocessed)//16, epochs=10)

        print('training done')

            #feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = categorical_features + numeric_features

######################### SHAP VALUES #########################
    explainer = shap.Explainer(model.predict, shap.maskers.Independent(data_train), data=X_train_preprocessed)
    explanations = explainer(data_train)
    shap_values = explainer.shap_values(X_test_preprocessed)
    expected_value = np.array(model.predict(X_test_preprocessed)).mean()
    explanations.feature_names = [el for el in headers[:-1]]

    ######################### SHAP PLOTS ##########################
    shap.plots.bar(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Bar plot of SHAP values")
    plt.savefig('assets/NN/shap/bar.png', bbox_inches='tight')
    plt.close()
    shap.plots.beeswarm(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Dot plot of SHAP values")
    plt.savefig('assets/NN/shap/bee.png', bbox_inches='tight')
    plt.close()
    shap.plots.heatmap(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Heatmap plot of SHAP values")
    plt.savefig('assets/NN/shap/heatmap.png', bbox_inches='tight')
    plt.close()


    print('Results saved')