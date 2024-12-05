import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from customModels import PolyRegressor
import warnings
import os

rng = 69420

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    

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

    param_lr = [{'fit_intercept':[True,False]}]

    models_regression = {
        'lr': {'name': 'Linear Regression',
               'estimator': PolyRegressor(adv=loss),
               'param': param_lr,
              },
    }

    k = 10
    kf = KFold(n_splits=k, random_state=None)

    X_preprocessed = preprocessor.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size = 0.25, random_state=np.random.RandomState(rng))

    model = models_regression['lr']['estimator']
    _ = model.fit(x_train, y_train)

    target_pred = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, target_pred)
    mse = metrics.mean_squared_error(y_test, target_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, target_pred))
    mape = smape(y_test, target_pred)

    
    ######### FEATURE SCORES ###########
    
        
    importance = []
        

    importance = model.weights
    coefs = pd.DataFrame(model.weights,
                            columns=["Coefficients"],
                            index= labels)

    # plot feature importance

    indexes = np.arange(len(labels))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, labels, rotation=48)
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


