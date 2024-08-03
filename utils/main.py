import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import statsmodels.api as sm
from pathlib import Path

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from lime.lime_tabular import LimeTabularExplainer
import warnings
import os
import shutil
from libraries import create_explanations, summaryPlot, HeatMap_plot, Waterfall, Decision_plot
from libraries_anova import P_anova, create_dict, Conditions, Anova_Decomposition
import dice_ml

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def predict_proba_wrapper(X):
    return model.predict(X)

if __name__ == "__main__":

    if(len(sys.argv)<4):
        print("ERROR! Usage: python scriptName.py fileCSV targetN modelloML\n")
              
        sys.exit(1)
    nome_script, pathCSV, targId, mlModel = sys.argv

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

    index_target= dataset.iloc[:,-7:]
    list_ind_t = index_target.columns.values.tolist()
    targetN = list_ind_t[targetId]

    X = dataset[['timeDuration', 'nMovements', 'movementsDifficulty', 'AItechnique', 'robotSpeech',    'acrobaticMovements', 'movementsRepetition', 'musicGenre', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']]
    y = dataset[targetN]

    categorical_features = ['AItechnique', 'musicGenre']
    numeric_features = ['timeDuration', 'nMovements', 'movementsDifficulty', 'robotSpeech', 'acrobaticMovements', 'movementsRepetition', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])    

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])
    model_reg = ['lr',
                'dt',
                'rf',
                'gbr']

    param_lr = [{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}]

    param_dt = [{'max_depth': [5,10,20]}]

    param_rf = [{'bootstrap': [True, False],
                 'max_depth': [10, 20],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2],}]

    param_gbr = [{'learning_rate': [0.01,0.03],
                'subsample'    : [0.5, 0.2],
                'n_estimators' : [100,200],
                'max_depth'    : [4,8]}]

    models_regression = {
        'lr': {'name': 'Linear Regression',
               'estimator': LinearRegression(),
               'param': param_lr,
              },
        'dt': {'name': 'Decision Tree',
               'estimator': DecisionTreeRegressor(random_state=42),
               'param': param_dt,
              },
        'rf': {'name': 'Random Forest',
               'estimator': RandomForestRegressor(random_state=42),
               'param': param_rf,
              },

        'gbr': {'name': 'Gradient Boosting Regressor',
                'estimator': GradientBoostingRegressor(random_state=42),
                'param': param_gbr
                },
    }

    k = 10
    kf = KFold(n_splits=k, random_state=None)
    mod_grid = GridSearchCV(models_regression[mlModel]['estimator'], models_regression[mlModel]['param'], cv=5, return_train_score = False, scoring='neg_mean_squared_error', n_jobs = 8)

    mae = []
    mse = []
    rmse = []
    mape = []

    X_preprocessed = preprocessor.fit_transform(X)

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index] , y[test_index]

        data_train_lime = preprocessor.fit_transform(data_train)
        data_test_lime = preprocessor.transform(data_test)

        model_lime = Pipeline(steps=[('regressor', mod_grid)])
        model = Pipeline(steps=[('preprocessor', preprocessor),
                ('regressor', mod_grid)])

        _ = model_lime.fit(data_train_lime, target_train)
        _ = model.fit(data_train, target_train)

        feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(feature_names_categorical)
        target_pred = model_lime.predict(data_test_lime)
    
        mae.append(metrics.mean_absolute_error(target_test, target_pred))
        mse.append(metrics.mean_squared_error(target_test, target_pred))
        rmse.append(np.sqrt(metrics.mean_squared_error(target_test, target_pred)))
        mape.append(smape(target_test, target_pred))

        #################### LIME Explanation ########################
        explainer = LimeTabularExplainer(data_train_lime,
                                         feature_names=feature_names,
                                         class_names=[targetN],
                                         mode='regression',
                                         discretize_continuous=True)
        
        random_numbers = np.random.randint(0, 70, size=5)
        explanation_instances = []
        for i in random_numbers:
            explanation_instances.append(data_test_lime[i])

    for idx, instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,
                                        model_lime.predict,
                                        num_features=5,) #5 most signficant
        
        output_folder = 'Results-%s/Results-%s/%s/Plot/' %(back[pos], mlModel, targetN)
        lime_folder = os.path.join(output_folder, 'lime_explanations')

        if not os.path.exists(lime_folder):
            os.makedirs(lime_folder)

        # save Lime explanation results
        exp.save_to_file(os.path.join(lime_folder, f'lime_explanation_{idx}.html'))

        




#################### PLOT and SCORES ########################
    output_folder = 'Results-%s/Results-%s/%s/Plot/' %(back[pos], mlModel,targetN)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    ######### FEATURE SCORES ###########
    
    feature_cat_names = model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    
    l= feature_cat_names.tolist()
    ltot = numeric_features + l
    
    importance = []
    
    if (mlModel=='lr'):
        importance = mod_grid.best_estimator_.coef_
        coefs = pd.DataFrame(mod_grid.best_estimator_.coef_,
                                 columns=["Coefficients"],
                                 index= ltot)

    elif (mlModel=='dt' or mlModel=='rf' or mlModel=='gbr'):
        importance = mod_grid.best_estimator_.feature_importances_
        coefs = pd.DataFrame(mod_grid.best_estimator_.feature_importances_,
                             columns=["Coefficients"],
                             index= ltot)

    else:
        c = [None] * len(ltot)
        l = mod_grid.best_estimator_.coefs_[0]
        n_l = mod_grid.best_params_['hidden_layer_sizes'][0]
        for i in range(len(ltot)):
            c[i] = l[i][n_l-1]
            importance = c
            coefs = pd.DataFrame(c,
                                 columns=["Coefficients"],
                                 index= ltot)

    # plot feature importance
    lf = ['t', 'n', 'md', 'rs', 'am', 'mr', 'mtd', 'h', 'b', 's', 'bc', 'bpm', 'pp', 'hm', 'arm', 'hdm', 'lm', 'fm', 'AIc', 'AIp', 'AIs', 'mEl', 'mFol', 'mInd', 'mPop', 'mRap', 'mRock']
    indexes = np.arange(len(lf))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, lf, rotation = '48')
    plt.savefig(output_folder + 'bar.png')
    plt.clf()
    plt.cla()
    plt.close()

    # plot SHAP
    
    _ = summaryPlot(model, X, preprocessor, lf, output_folder, 'Dot_plot', 'dot')
    _ = summaryPlot(model, X, preprocessor, lf, output_folder, 'Violin_plot', 'violin')
    ordered_labels = summaryPlot(model, X, preprocessor, lf, output_folder, 'Bar_plot', 'bar')
    HeatMap_plot(model, X, preprocessor, output_folder, 'HeatMap_plot', lf)
    
    # Show some specific examples
    Showed_examples = 5 
    idx = np.random.randint(0, X.shape[0], Showed_examples)
    for i,el in enumerate(idx):
       Decision_plot(model, X, preprocessor, output_folder, el, f'Decision_plot{i}', lf)
       Waterfall(model, X, preprocessor, output_folder, el, f'Waterfall_Plot_{i}', lf)


################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('Results-%s/Results-%s/%s/res.txt' %(back[pos], mlModel,targetN), 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Mean Absolute Error:', np.mean(mae))
        print('Mean Squared Error:', np.mean(mse))
        print('Root Mean Squared Error:', np.mean(rmse))
        print('Mean Average Percentage Error:', np.mean(mape))
        print('\nFeature Scores: \n')
        print(coefs)
            
        print('\nBest Parameters used: ', mod_grid.best_params_)

        
    sys.stdout = original_stdout
    shutil.rmtree(os.getcwd() + "\__pycache__")
    print('Results saved')

    ####################### DiCE #############################
    
    Ncount=30

    Xdice = preprocessor.fit_transform(X)

    feature_cat_names = model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    l= feature_cat_names.tolist()
    ltot = numeric_features + l

    constraints={}
    
    Xdice = pd.DataFrame(Xdice, columns=ltot)

    desc=Xdice.describe()
    #print(desc)
    for i in numeric_features:
        constraints[i]=[desc[i]['min'],desc[i]['max']]
    Xdice['output'] = y

    X_train, X_test = train_test_split(Xdice,test_size=0.2,random_state=42,stratify=Xdice['output'])

    dice_train = dice_ml.Data(dataframe=X_train,
                 continuous_features=numeric_features,
                 outcome_name='output')
    
    m = dice_ml.Model(model=mod_grid.best_estimator_,backend='sklearn', model_type='regressor',func=None)
    exp = dice_ml.Dice(dice_train,m)

    query_instance = X_test.drop(columns="output")
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount,desired_range=[1,5],permitted_range=constraints)

    if not Path.cwd().joinpath("dice_results").exists():
        Path.cwd().joinpath("dice_results").mkdir(parents=True)
    data = []
    for cf_example in dice_exp.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_combined = pd.concat(data, ignore_index=True)
    for i in range(len(df_combined)):
        df_combined.iloc[i] = df_combined.iloc[i] - X_test.iloc[i//Ncount]
    df_combined.to_csv(path_or_buf=f'dice_results/{targetId}_{mlModel}_{ds}_counterfactuals.csv', index=False, sep=',')
    df_combined.dtypes
    df_filtered = df_combined[df_combined['output'] != 0]
    count_per_column = df_filtered.apply(lambda x: (x != 0).sum())
    diff_per_column = df_filtered.apply(lambda x: (abs(x)*abs(df_filtered['output'])).sum())

    #print(count_per_column)
    #print(diff_per_column['output'])

    correlation_matrix = df_combined.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation")

    plt.savefig(f'dice_results/{targetId}_{mlModel}_{ds}_correlations.png')

    original_stdout = sys.stdout
    with open(f'dice_results/{targetId}_{mlModel}_{ds}_count.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Counterfactual absolute counts:-------------------------')
        print(diff_per_column)
        print('\n--------------------- Counterfactual relative counts:-------------------------')
        print(diff_per_column/count_per_column)
            
        
    sys.stdout = original_stdout

    ###### ANOVA DECOMPOSITION #########
    
    selected_coefficent = Anova_Decomposition(mod_grid.best_estimator_, X,y, preprocessor, lf)


    original_stdout = sys.stdout
    with open('Results-%s/Results-%s/%s/resLasso.txt' %(back[pos], mlModel,targetN), 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('\nFeature Scores: \n')
        for el in selected_coefficent:
            print(el)
        
    sys.stdout = original_stdout

