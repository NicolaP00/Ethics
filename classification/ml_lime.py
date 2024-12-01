import numpy as np
import pandas as pd
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
from sklearn.metrics import f1_score
from lime.lime_tabular import LimeTabularExplainer
import warnings
import os
from tensorflow.keras.backend import one_hot, argmax

def smape(y_true, y_pred):
    y_true = y_true.reshape(y_pred.shape)
    y_true = y_true[:,0]
    y_pred = y_pred[:,0]
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

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV, sep=',')
    headers = dataset.columns.tolist()

    if not os.path.exists(f'assets/{mlModel}/lime'):
        os.makedirs(f'assets/{mlModel}/lime')

    X = dataset[headers[:-1]]
    data = {
    'Fault': dataset[headers[-1]],
    'Normal': (~dataset[headers[-1]].astype(bool)).astype(int)
    }
    if mlModel=='gbc' or mlModel=='nb':
        y = pd.DataFrame({'Fault':data['Fault']})
    else:
        y = pd.DataFrame(data)
    y = np.array(y)


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
    kf = KFold(n_splits=k, shuffle=True, random_state=69)
    mod_grid = GridSearchCV(models_classification[mlModel]['estimator'], models_classification[mlModel]['param'], cv=5, return_train_score = False, scoring='f1_macro', n_jobs = 8)


    print('preprocessing done')

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index,:],y[test_index,:]

        data_train_lime = preprocessor.fit_transform(data_train)
        data_test_lime = preprocessor.transform(data_test)
        model_lime = Pipeline(steps=[('classifier', mod_grid)])

        _ = model_lime.fit(data_train_lime, target_train)

        print('training done')

    feature_names = categorical_features + numeric_features



            #################### LIME Explanation ########################
    explainer = LimeTabularExplainer(data_train_lime,
                                            feature_names=feature_names,
                                            categorical_features=[i for i, x in enumerate(headers) if x in categorical_features],
                                            mode='classification',
                                            discretize_continuous=False)
            
    random_numbers = np.random.randint(0, len(data_test_lime), size=5)
    explanation_instances = []
    for i in random_numbers:
        explanation_instances.append(data_test_lime[i])

    for idx, instance in enumerate(explanation_instances):
        if mlModel == 'nb' or mlModel == 'gbc':
            exp = explainer.explain_instance(instance,
                                        model_lime.predict_proba,
                                        num_features=6,) #6 most signficant
        else:
            exp = explainer.explain_instance(instance,
                                        model_lime.predict,
                                        num_features=6,) #6 most signficant

        # save Lime explanation results
        exp.save_to_file(f'assets/{mlModel}/lime/lime_explanation_{idx+1}.html')
 
    print('lime finished')
