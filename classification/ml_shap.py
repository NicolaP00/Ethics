import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import shap

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.backend import one_hot, argmax

import warnings
import os

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py modelloML\n")
              
        sys.exit(1)

    nome_script, mlModel, = sys.argv

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV, sep=',')
    headers = dataset.columns.tolist()
    print(headers)

    if not os.path.exists('assets/' + mlModel + '/shap'):
        os.makedirs('assets/' + mlModel + '/shap')

    X = dataset[headers[:-1]]
    y = np.array(pd.DataFrame(dataset[headers[-1]]))

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp']
    numeric_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'thall']

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                          ('label', OrdinalEncoder())
                                          ])  

    preprocessor = ColumnTransformer(
                                 transformers=[
                                                ('cat', categorical_transformer, categorical_features),
                                               ('num', numeric_transformer, numeric_features),
                                               ])
    model_reg = ['nb'
                 'dt',
                 'rf',
                 'gbc']

    param_nb = [{'var_smoothing': np.logspace(0,-9, num=10)}]

    param_dt = [{'max_depth': [5,10,20]}]

    param_rf = [{'bootstrap': [True, False],
                 'max_depth': [10, 20],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2],}]

    param_gbc = [{'learning_rate': [0.01,0.03],
                'subsample'    : [0.5, 0.2],
                'n_estimators' : [100,200],
                'max_depth'    : [4,8]}]

    models_classification = {
        'nb': {'name': 'Naive Bayes',
               'estimator': GaussianNB(),
               'param': param_nb,
              },
        'dt': {'name': 'Decision Tree',
               'estimator': DecisionTreeClassifier(random_state=42),
               'param': param_dt,
              },
        'rf': {'name': 'Random Forest',
               'estimator': RandomForestClassifier(random_state=42),
               'param': param_rf,
              },

        'gbc': {'name': 'Gradient Boosting Classifier',
                'estimator': GradientBoostingClassifier(random_state=42),
                'param': param_gbc
                },
    }

    k = 5           #CAMBIATO, PRIMA ERA 10
    kf = KFold(n_splits=k, random_state=None)
    mod_grid = GridSearchCV(models_classification[mlModel]['estimator'], models_classification[mlModel]['param'], cv=5, return_train_score = False, scoring='f1_macro', n_jobs = 8)

    mae = []
    mse = []
    rmse = []
    mape = []
    f1 = []
    X_preprocessed = preprocessor.fit_transform(X)

    print('preprocessing done')
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', mod_grid)])
    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index,:],y[test_index,:]


        _ = model.fit(data_train, target_train)
        print('training done')

    ######################### SHAP VALUES #########################
    explainer = shap.Explainer(model['classifier'].best_estimator_.predict, shap.maskers.Independent(data_train), data=X_preprocessed)
    explanations = explainer(preprocessor.transform(data_train))
    shap_values = explainer.shap_values(preprocessor.transform(data_test))
    expected_value = np.array(model['classifier'].best_estimator_.predict(preprocessor.transform(data_test))).mean()
    explanations.feature_names = [el for el in headers[:-1]]
    
    ######################### SHAP PLOTS ##########################
    shap.plots.bar(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Bar plot of SHAP values")
    plt.savefig(f'assets/{mlModel}/shap/bar.png', bbox_inches='tight')
    plt.close()
    shap.plots.violin(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Violin plot of SHAP values")
    plt.savefig(f'assets/{mlModel}/shap/violin.png', bbox_inches='tight')
    plt.close()
    shap.plots.beeswarm(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Dot plot of SHAP values")
    plt.savefig(f'assets/{mlModel}/shap/bee.png', bbox_inches='tight')
    plt.close()
    shap.plots.heatmap(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Heatmap plot of SHAP values")
    plt.savefig(f'assets/{mlModel}/shap/heatmap.png', bbox_inches='tight')
    plt.close()
    # Show some specific examples
    Showed_examples = 5 
    idx = np.random.randint(0, data_test.shape[0], Showed_examples)
    for i,el in enumerate(idx):
        shap.plots.waterfall(explanations[el, :], show=False)
        plt.title(f'Waterfall plot of random explanation')
        plt.savefig(f'assets/{mlModel}/shap/waterfall_{i+1}.png', bbox_inches='tight')
        plt.close()
        shap.plots.decision(expected_value, shap_values[el, :], feature_names=[el for el in headers[:-1]], show=False)
        plt.title("Decision plot of random explanation")
        plt.savefig(f'assets/{mlModel}/shap/decision_{i+1}.png', bbox_inches='tight')
        plt.close()
