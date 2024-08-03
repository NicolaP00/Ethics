import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from customModels import PolyRegressor
import warnings
import os

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def location_fooling(y_true,y_pred,c=['t'],lmbda=1):
    loss = mean_squared_error(y_true,y_pred)+lmbda*sum()
    

if __name__ == "__main__":

    if(len(sys.argv)<4):
        print("ERROR! Usage: python scriptName.py fileCSV targetN modelloML\n")
              
        sys.exit(1)
    nome_script, pathCSV, targId, loss = sys.argv

    targetId = int(targId)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    back = ['Artistic', 'Scientific']
    pos = 1
    if (pathCSV == 'datasetArtisticBackground.csv'):
        pos = 0


    dataset = pd.read_csv(pathCSV, sep=';')

    index_target= dataset.iloc[:,-7:]
    list_ind_t = index_target.columns.values.tolist()
    targetN = list_ind_t[targetId]

    X = dataset[['timeDuration', 'nMovements', 'movementsDifficulty', 'AItechnique', 'robotSpeech',    'acrobaticMovements', 'movementsRepetition', 'musicGenre', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']]
    y = dataset[targetN]

    categorical_features = ['AItechnique', 'musicGenre']
    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = ['timeDuration', 'nMovements', 'movementsDifficulty', 'robotSpeech',    'acrobaticMovements', 'movementsRepetition', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']
    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])
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
    feature_cat_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    l= feature_cat_names.tolist()
    ltot = numeric_features + l
    X = pd.DataFrame(X, columns=ltot)

    mae = []
    mse = []
    rmse = []
    mape = []

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index] , y[test_index]

        model = models_regression['lr']['estimator']
    


        _ = model.fit(data_train, target_train)

        target_pred = model.predict(data_test)
    
        mae.append(metrics.mean_absolute_error(target_test, target_pred))
        mse.append(metrics.mean_squared_error(target_test, target_pred))
        rmse.append(np.sqrt(metrics.mean_squared_error(target_test, target_pred)))
        mape.append(smape(target_test, target_pred))


#################### PLOT and SCORES ########################
    output_folder = 'Results-%s/Results-%s/%s/Plot/' %(back[pos], 'lr',targetN)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    ######### FEATURE SCORES ###########
    
    feature_cat_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names(categorical_features)
        
    l= feature_cat_names.tolist()
    ltot = numeric_features + l
        
    importance = []
        

    importance = model.weights
    coefs = pd.DataFrame(model.weights,
                            columns=["Coefficients"],
                            index= ltot)

    # plot feature importance
    lf = ['t', 'n', 'md', 'rs', 'am', 'mr', 'mtd', 'h', 'b', 's', 'bc', 'bpm', 'pp', 'hm', 'arm', 'hdm', 'lm', 'fm', 'AIc', 'AIp', 'AIs', 'mEl', 'mFol', 'mInd', 'mPop', 'mRap', 'mRock']
    indexes = np.arange(len(lf))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, lf, rotation = '48')
    plt.savefig(output_folder + 'bar-%s.png' %loss)
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('Results-%s/Results-%s/%s/res-%s.txt' %(back[pos], 'lr',targetN,loss), 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Mean Absolute Error:', np.mean(mae))
        print('Mean Squared Error:', np.mean(mse))
        print('Root Mean Squared Error:', np.mean(rmse))
        print('Mean Average Percentage Error:', np.mean(mape))
        print('\nFeature Scores: \n')
        print(coefs)
            
        #print('\nBest Parameters used: ', mod_grid.best_params_)
        
    sys.stdout = original_stdout
    print('Results saved')


