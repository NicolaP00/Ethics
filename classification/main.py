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
from lime.lime_tabular import LimeTabularExplainer
import warnings
import os
import shutil
from keras.backend import one_hot, argmax
from libraries import create_explanations, summaryPlot, HeatMap_plot, Waterfall, Decision_plot

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def predict_proba_wrapper(X):
    return model.predict(X)

def mapp(df, mapping):
    df['Type'] = df['Type'].map(mapping)

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py modelloML\n")
              
        sys.exit(1)

    nome_script, mlModel, = sys.argv

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'dataset.csv'
    dataset = pd.read_csv(pathCSV, sep=',')
    headers = dataset.columns.tolist()

    if not os.path.exists(mlModel+'/lime'):
        os.makedirs(mlModel+'/lime')

    #if not os.path.exists(mlModel+'/shap'):
    #    os.makedirs(mlModel+'/shap')

    if not os.path.exists(mlModel+'/dice'):
        os.makedirs(mlModel+'/dice')


    index_target= dataset.iloc[:,-1]
    X = dataset[headers[2:-6]]
    data = {
    'Fault': dataset[headers[-6]],
    'Normal': (~dataset[headers[-6]].astype(bool)).astype(int)
    }

    y = pd.DataFrame(data)
    y = np.array(y)


    categorical_features = ['Type']
    numeric_features = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
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

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index,:],y[test_index,:]

        data_train_lime = preprocessor.fit_transform(data_train)
        data_test_lime = preprocessor.transform(data_test)

        model_lime = Pipeline(steps=[('classifier', mod_grid)])
        model = Pipeline(steps=[('preprocessor', preprocessor),
                ('classifier', mod_grid)])

        _ = model_lime.fit(data_train_lime, target_train)
        _ = model.fit(data_train, target_train)

        print('training done')

            #feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = categorical_features + numeric_features
        target_pred = model_lime.predict(data_test_lime)
        mae.append(metrics.mean_absolute_error(target_test, target_pred))
        mse.append(metrics.mean_squared_error(target_test, target_pred))
        rmse.append(np.sqrt(metrics.mean_squared_error(target_test, target_pred)))
        mape.append(smape(target_test, target_pred))
        f1.append(f1_score(target_test, one_hot(argmax(target_pred, axis=-1), num_classes=target_pred.shape[-1]), average='micro'))

            #################### LIME Explanation ########################
        explainer = LimeTabularExplainer(data_train_lime,
                                            feature_names=feature_names,
                                            categorical_features=[i for i, x in enumerate(headers) if x in categorical_features],
                                            mode='classification',
                                            discretize_continuous=False)
            
        random_numbers = np.random.randint(0, 70, size=5)
        explanation_instances = []
        for i in random_numbers:
                explanation_instances.append(data_test_lime[i])

    for idx, instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,
                                        model_lime.predict,
                                        num_features=5,) #5 most signficant

        # save Lime explanation results
        exp.save_to_file(f'{mlModel}/lime/lime_explanation_{idx}.html')
 
    print('lime finished')

    #################### plot SHAP #############################
    '''_ = summaryPlot(model, X, preprocessor, headers[2:-6], f'{mlModel}/shap/', 'Dot_plot', 'dot')
    _ = summaryPlot(model, X, preprocessor, headers[2:-6], f'{mlModel}/shap/', 'Violin_plot', 'violin')
    ordered_labels = summaryPlot(model, X, preprocessor, headers[2:-6], f'{mlModel}/shap/', 'Bar_plot', 'bar')
    #HeatMap_plot(model, X, preprocessor, f'{mlModel}/shap/', 'HeatMap_plot', headers[2:-6])
    
    # Show some specific examples
    Showed_examples = 5 
    idx = np.random.randint(0, X.shape[0], Showed_examples)
    #for i,el in enumerate(idx):
       #Decision_plot(model, X, preprocessor, f'{mlModel}/shap/', el, f'Decision_plot{i}', headers[2:-6])
       #Waterfall(model, X, preprocessor, f'{mlModel}/shap/', el, f'Waterfall_Plot_{i}', headers[2:-6])'''



####################### GOLDEN STANDARDS #############################

    importance = []

    if (mlModel=='dt' or mlModel=='rf' or mlModel=='gbc'):
        importance = mod_grid.best_estimator_.feature_importances_
        coefs = pd.DataFrame(mod_grid.best_estimator_.feature_importances_,
                             columns=["Coefficients"],
                             index= headers[2:-6])

    else:
        c = [None] * len(headers[2:-6])
        l = mod_grid.best_estimator_.coefs_[0]
        n_l = mod_grid.best_params_['hidden_layer_sizes'][0]
        for i in range(len(headers[2:-6])):
            c[i] = l[i][n_l-1]
            importance = c
            coefs = pd.DataFrame(c,
                                 columns=["Coefficients"],
                                 index= headers[2:-7])

    original_stdout = sys.stdout
    with open('%s/res.txt' %(mlModel), 'w') as f:
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

    indexes = np.arange(len(headers[2:-7]))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, headers[2:-7], rotation = '48')
    plt.savefig('golden.png')
    plt.clf()
    plt.cla()
    plt.close()
        
    sys.stdout = original_stdout
    print('Results saved')