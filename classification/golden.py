import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
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

def smape(y_true, y_pred):
    y_true = y_true.reshape(y_pred.shape)
    division = np.abs(y_pred - y_true) / (y_true + y_pred)
    sum = 0
    for el in division:
        if not np.isnan(el):
            sum += el
    return 100/len(y_true) * sum


if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py modelloML\n")
              
        sys.exit(1)

    nome_script, mlModel, = sys.argv

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    if not os.path.exists(f'assets/{mlModel}/'):
        os.makedirs(f'assets/{mlModel}/')

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV, sep=',')
    headers = dataset.columns.tolist()
    print(headers)

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

    k = 20           #CAMBIATO, PRIMA ERA 10
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
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
        target_pred = model.predict(data_test)
        mae.append(metrics.mean_absolute_error(target_test, target_pred))
        mse.append(metrics.mean_squared_error(target_test, target_pred))
        rmse.append(np.sqrt(metrics.mean_squared_error(target_test, target_pred)))
        mape.append(smape(target_test, target_pred))
        f1.append(metrics.f1_score(target_test, one_hot(argmax(target_pred, axis=-1), num_classes=target_pred.shape[-1]), average='micro'))


####################### GOLDEN STANDARDS #############################

    importance = []
    coefs = []
    if mlModel!='nb':
        importance = mod_grid.best_estimator_.feature_importances_
        coefs = pd.DataFrame(mod_grid.best_estimator_.feature_importances_,
                                columns=["Coefficients"],
                                index= headers[:-1])


    original_stdout = sys.stdout
    with open(f'assets/{mlModel}/res.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Mean Absolute Error:', np.mean(mae))
        print('Mean Squared Error:', np.mean(mse))
        print('Root Mean Squared Error:', np.mean(rmse))
        print('Mean Average Percentage Error:', np.mean(mape))
        print('Macro f1 score', np.mean(f1))
        print('\nFeature Scores: \n')
        print(coefs)
            
        print('\nBest Parameters used: ', mod_grid.best_params_)

    indexes = np.arange(len(headers[:-1]))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, headers[:-1], rotation=48)
    plt.savefig(f'assets/{mlModel}/golden.png')
    plt.clf()
    plt.cla()
    plt.close()
        
    sys.stdout = original_stdout
    print('Results saved')
