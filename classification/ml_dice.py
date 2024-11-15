import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import warnings
import os
from tensorflow.keras.backend import one_hot, argmax
import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers

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

    if not os.path.exists(f'assets/{mlModel}/dice'):
        os.makedirs(f'assets/{mlModel}/dice')

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp']
    numeric_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'thall']

    dataset = dataset[categorical_features+numeric_features+['output']]
    headers = dataset.columns.tolist()

    X = dataset[headers[:-1]]
    y = np.array(dataset[headers[-1]])
    print(X.dtypes)


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


    param_nb = {'var_smoothing': np.logspace(0,-9, num=10)}

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
        target_train , target_test = y[train_index],y[test_index]

        _ = model.fit(data_train, target_train)

        print('training done')


    Xdice = preprocessor.fit_transform(X)
    Xdice = pd.DataFrame(Xdice, columns=categorical_features+numeric_features)
    Ncount = 30

    constraints={}
    desc = Xdice.describe()

    for i in numeric_features:
        constraints[i]=[desc[i]['min'], desc[i]['max']]
    Xdice[headers[-1]] = y
    desc = Xdice.describe()

    X_train, X_test = train_test_split(Xdice, test_size=0.2, random_state=42)
    dice_train = dice_ml.Data(dataframe=Xdice, continuous_features=numeric_features, outcome_name=headers[-1])
    dice_model = dice_ml.Model(model=mod_grid.best_estimator_, backend="sklearn")

    exp = Dice(dice_train, dice_model)

    query_instance = X_test.drop(columns=headers[-1])
    
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount, desired_range=None, permitted_range=constraints)
    
    data = []
    for cf_example in dice_exp.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_combined = pd.concat(data, ignore_index=True)
    for i in range(len(df_combined)):
        df_combined.iloc[i] = df_combined.iloc[i] - X_test.iloc[i//Ncount]
    df_combined.to_csv(path_or_buf=f'assets/{mlModel}/dice/conterfactuals.csv', index=False, sep=',')
    df_combined.dtypes
    df_filtered=df_combined[df_combined['output'] != 0]
    count_per_column = df_filtered.apply(lambda x: (x != 0).sum())
    diff_per_column = df_filtered.apply(lambda x: (abs(x)).sum())
    original_stdout = sys.stdout
    with open(f'assets/{mlModel}/dice/count.txt','w') as f:
        sys.stdout = f
        print('\n--------------------- Counterfactual absolute counts: ---------------------')
        print(diff_per_column)
        print('\n--------------------- Counterfactual relative counts: ---------------------')
        print(diff_per_column/count_per_column)
    sys.stdout = original_stdout
