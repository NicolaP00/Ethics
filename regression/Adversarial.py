import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from customModels import PolyRegressor, CustomRegressor
import warnings
import os
from sklearn.ensemble import RandomForestRegressor
from lime.lime_tabular import LimeTabularExplainer
import dice_ml

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
   

if __name__ == "__main__":

    if(len(sys.argv)<3):
        print("ERROR! Usage: python scriptName.py fileCSV targetN modelloML\n")
              
        sys.exit(1)
    nome_script, pathCSV, targId = sys.argv

    targetId = int(targId)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    back = ['Artistic', 'Scientific']
    pos = 1
    ds = 'S'
    if (pathCSV == 'datasetArtisticBackground.csv'):
        pos = 0
        ds = 'A'


    dataset = pd.read_csv(pathCSV, sep=';')
    #conterfactual_dataset = f'dice_results/{targetId}_lr_{ds}_counterfactuals.csv'

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


    models_regression = {
        'lr': {'name': 'Linear Regression',
               'estimator': LinearRegression(fit_intercept=False,copy_X=True,normalize=False),
              },
        'fake': {'name':'Custom',
                 'estimator':PolyRegressor(adv='lf'),
                },
        's':{'name':'selector',
             'estimator':RandomForestRegressor()}
    }

    X = preprocessor.fit_transform(X)
    feature_cat_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    l= feature_cat_names.tolist()
    ltot = numeric_features + l
    X = pd.DataFrame(X, columns=ltot)

    mae = []
    mse = []
    rmse = []
    mape = []

    k = 10
    kf = KFold(n_splits=k, random_state=None)

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index] , y[test_index]
        model = models_regression['lr']['estimator']    
        _ = model.fit(data_train, target_train)

    ################ DiCE #################
        
    Ncount=3

    X['output'] = y

    #X_train, X_test = train_test_split(X,test_size=0.2,random_state=42,stratify=X['output'])

    dice_train = dice_ml.Data(dataframe=X,
                 continuous_features=numeric_features,
                 outcome_name='output')
    
    m = dice_ml.Model(model=model,backend='sklearn', model_type='regressor',func=None)
    exp = dice_ml.Dice(dice_train,m)

    query_instance = X.drop(columns="output")
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount,desired_range=[1,5])
    data = []
    for cf_example in dice_exp.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_counterfactual = pd.concat(data, ignore_index=True)
    y_counterfactual = df_counterfactual['output']

    Xcount_train,Xcount_test,ycount_train,ycount_test = train_test_split(df_counterfactual,y_counterfactual,random_state=42)
    OOD_train = np.concatenate([np.zeros(X_train.shape[0]), np.ones(Xcount_train.shape[0])])
    OOD_test = np.concatenate([np.zeros(X_test.shape[0]), np.ones(Xcount_test.shape[0])])
    OOD = np.concatenate([OOD_train, OOD_test])
    indices = np.random.permutation(len(OOD))
    OOD = OOD[indices]
    merged_train = pd.concat([X_train, Xcount_train], ignore_index=True)
    merged_test = pd.concat([X_test, Xcount_test], ignore_index=True)
    merged = pd.concat([X, df_counterfactual], ignore_index=True).iloc[indices].reset_index(drop=True)
    ymerged_train = merged_train['output']
    ymerged_test = merged_test['output']
    y_merged = merged['output'].reset_index(drop=True)
    X = X.drop(['output'], axis=1)
    df_counterfactual = df_counterfactual.drop(['output'], axis=1)
    Xcount_train = Xcount_train.drop(['output'], axis=1)
    Xcount_test = Xcount_test.drop(['output'], axis=1)
    merged_train = merged_train.drop(['output'], axis=1)
    merged_test = merged_test.drop(['output'], axis=1)
    merged = merged.drop(['output'],axis=1)

    for train_index , test_index in kf.split(df_counterfactual):
        data_train , data_test = df_counterfactual.iloc[train_index,:],df_counterfactual.iloc[test_index,:]
        target_train , target_test = y_counterfactual[train_index] , y_counterfactual[test_index]
        fake = models_regression['fake']['estimator']    
        _ = fake.fit(data_train, target_train)

    ################ SELECTOR ###############
    
    for train_index , test_index in kf.split(merged):
        data_train , data_test = merged.iloc[train_index,:],merged.iloc[test_index,:]
        target_train , target_test = OOD[train_index] , OOD[test_index]
        selec = models_regression['s']['estimator']
        _ = selec.fit(data_train,target_train)

    ################ LIME ####################
            
    feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(feature_names_categorical)

    explainer = LimeTabularExplainer(merged_train.values,
                                         feature_names=feature_names,
                                         class_names=[targetN],
                                         mode='regression',
                                         discretize_continuous=True,
                                         random_state=42)
    
    random_numbers = np.random.randint(0, merged_test.shape[0], size=5)
    explanation_instances = []
    for i in random_numbers:
        explanation_instances.append(merged_test.values[i])
    output_folder = 'Results-%s/Results-lr/%s/Plot/' %(back[pos], targetN)
    for idx,instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,CustomRegressor(model,fake,selec).predict,num_features=5)
        #lime_folder = os.path.join(output_folder, 'lime_explanations')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # save Lime explanation results
        exp.save_to_file(os.path.join(output_folder, f'lime_explanation_{idx}.html'))

    
    ######### FEATURE SCORES ###########
    
    feature_cat_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names(categorical_features)
        
    l= feature_cat_names.tolist()
    ltot = numeric_features + l
        
    importance = []
        

    importance = model.coef_
    coefs = pd.DataFrame(model.coef_,
                            columns=["Coefficients"],
                            index= ltot)

    # plot feature importance
    lf = ['t', 'n', 'md', 'rs', 'am', 'mr', 'mtd', 'h', 'b', 's', 'bc', 'bpm', 'pp', 'hm', 'arm', 'hdm', 'lm', 'fm', 'AIc', 'AIp', 'AIs', 'mEl', 'mFol', 'mInd', 'mPop', 'mRap', 'mRock']
    indexes = np.arange(len(lf))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, lf, rotation = '48')
    plt.savefig(output_folder + 'bar-ad-good.png')
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('Results-%s/Results-lr/%s/res-ad-good.txt' %(back[pos],targetN), 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print(coefs)
            
        #print('\nBest Parameters used: ', mod_grid.best_params_)
        
    sys.stdout = original_stdout
    print('Results saved')

    ############## FEATURE SCORE 2 ######################

    importance = fake.weights
    coefs = pd.DataFrame(fake.weights,
                            columns=["Coefficients"],
                            index= ltot)

    # plot feature importance
    lf = ['t', 'n', 'md', 'rs', 'am', 'mr', 'mtd', 'h', 'b', 's', 'bc', 'bpm', 'pp', 'hm', 'arm', 'hdm', 'lm', 'fm', 'AIc', 'AIp', 'AIs', 'mEl', 'mFol', 'mInd', 'mPop', 'mRap', 'mRock']
    indexes = np.arange(len(lf))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, lf, rotation = '48')
    plt.savefig(output_folder + 'bar-ad-fake.png')
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT 2 #################################

    original_stdout = sys.stdout
    with open('Results-%s/Results-lr/%s/res-ad-fake.txt' %(back[pos],targetN), 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print(coefs)
            
        #print('\nBest Parameters used: ', mod_grid.best_params_)
        
    sys.stdout = original_stdout
    print('Results saved')


