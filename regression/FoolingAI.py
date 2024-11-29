import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from customModels import PolyRegressor
import warnings
import os

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def location_fooling(y_true,y_pred,c=['t'],lmbda=1):
    loss = mean_squared_error(y_true,y_pred)+lmbda*sum()
    

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py modelloML\n")
              
        sys.exit(1)
    nome_script, loss = sys.argv

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"


    dataset = pd.read_csv('insurance.csv', sep=',')

    if not os.path.exists('lr/fooling'):
        os.makedirs('lr/fooling')

    X = dataset.drop(columns=['charges'])
    y = dataset['charges']

    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']
    labels = numeric_features + categorical_features

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('ordinal', OrdinalEncoder(handle_unknown='error'))])    

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)
                                              ])
    model_reg = ['lr']

    param_lr = [{'fit_intercept':[True,False], 'normalize':[True,False]}]

    models_regression = {
        'lr': {'name': 'Linear Regression',
               'estimator': PolyRegressor(adv=loss),
               'param': param_lr,
              },
    }

    k = 10
    kf = KFold(n_splits=k, random_state=None)

    X = preprocessor.fit_transform(X)
    X = pd.DataFrame(X, columns=labels)

    mae = []
    mse = []
    rmse = []
    mape = []
    test = []

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index] , y[test_index]

        model = models_regression['lr']['estimator']

        _ = model.fit(data_train, target_train)

        test.append((data_test, target_test))
    for d,t in test:
        target_pred = model.predict(d)
        mae.append(metrics.mean_absolute_error(t, target_pred))
        mse.append(metrics.mean_squared_error(t, target_pred))
        rmse.append(np.sqrt(metrics.mean_squared_error(t, target_pred)))
        mape.append(smape(t, target_pred))

    
    ######### FEATURE SCORES ###########
    
        
    importance = []
        

    importance = model.weights
    coefs = pd.DataFrame(model.weights,
                            columns=["Coefficients"],
                            index= labels)

    # plot feature importance

    indexes = np.arange(len(labels))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, labels, rotation = '48')
    plt.savefig('lr/fooling/bar-%s.png'%loss)
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('lr/fooling/res-%s.txt'%loss, 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Mean Absolute Error:', np.mean(mae))
        print('Mean Squared Error:', np.mean(mse))
        print('Root Mean Squared Error:', np.mean(rmse))
        print('Mean Average Percentage Error:', np.mean(mape))
        print('\nFeature Scores: \n')
        print(coefs)
                

            
    sys.stdout = original_stdout
    print('Results saved')


