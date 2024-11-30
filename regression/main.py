import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lime.lime_tabular import LimeTabularExplainer
import warnings
import os
import shutil
from libraries import summaryPlot, HeatMap_plot, Waterfall, Decision_plot
import dice_ml

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def predict_proba_wrapper(X):
    return model.predict(X)

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py modelloML\n")
              
        sys.exit(1)
    nome_script, mlModel = sys.argv

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    dataset = pd.read_csv('insurance.csv', sep=',')

    if not os.path.exists(mlModel+'/lime'):
        os.makedirs(mlModel+'/lime')

    if not os.path.exists(mlModel+'/shap'):
        os.makedirs(mlModel+'/shap')

    if not os.path.exists(mlModel+'/dice'):
        os.makedirs(mlModel+'/dice')

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
    test = []

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

        feature_names = numeric_features + categorical_features
        test.append([data_test_lime, target_test])

        explainer = LimeTabularExplainer(data_train_lime,
                                         feature_names=feature_names,
                                         class_names=['charges'],
                                         mode='regression',
                                         discretize_continuous=True)
        
        random_numbers = np.random.randint(0, len(data_test_lime), size=5)
        explanation_instances = []
        for i in random_numbers:
            explanation_instances.append(data_test_lime[i])
    
    for t in test:
        target_pred = model_lime.predict(t[0])
    
        mae.append(metrics.mean_absolute_error(t[1], target_pred))
        mse.append(metrics.mean_squared_error(t[1], target_pred))
        rmse.append(np.sqrt(metrics.mean_squared_error(t[1], target_pred)))
        mape.append(smape(t[1], target_pred))

    for idx, instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,
                                        model_lime.predict,
                                        num_features=5,) #5 most signficant
        


        # save Lime explanation results
        exp.save_to_file(f'{mlModel}/lime/lime_explanation_{idx}.html')

importance = []
    
if (mlModel=='lr'):
    importance = mod_grid.best_estimator_.coef_
    coefs = pd.DataFrame(mod_grid.best_estimator_.coef_, columns=["Coefficients"], index= labels)

elif (mlModel=='dt' or mlModel=='rf' or mlModel=='gbr'):
    importance = mod_grid.best_estimator_.feature_importances_
    coefs = pd.DataFrame(mod_grid.best_estimator_.feature_importances_, columns=["Coefficients"], index= labels)

else:
    c = [None] * len(labels)
    l = mod_grid.best_estimator_.coefs_[0]
    n_l = mod_grid.best_params_['hidden_layer_sizes'][0]
    for i in range(len(labels)):
        c[i] = l[i][n_l-1]
        importance = c
        coefs = pd.DataFrame(c,
                            columns=["Coefficients"],
                            index= labels)

# plot feature importance

indexes = np.arange(len(labels))
plt.bar([x for x in range(len(importance))], importance)
plt.xticks(indexes, labels, rotation = '48')
plt.savefig(mlModel + '/bar.png')
plt.clf()
plt.cla()
plt.close()

############################## SHAP ##########################

_ = summaryPlot(model, X, preprocessor, labels, mlModel+'/shap/', 'Dot_plot', 'dot')
_ = summaryPlot(model, X, preprocessor, labels, mlModel+'/shap/', 'Violin_plot', 'violin')
ordered_labels = summaryPlot(model, X, preprocessor, labels, mlModel+'/shap/', 'Bar_plot', 'bar')
HeatMap_plot(model, X, preprocessor, mlModel+'/shap/', 'HeatMap_plot', labels)
    
# Show some specific examples
Showed_examples = 5 
idx = np.random.randint(0, X.shape[0], Showed_examples)
for i,el in enumerate(idx):
    Decision_plot(model, X, preprocessor, mlModel+'/shap/', el, f'Decision_plot{i}', labels)
    Waterfall(model, X, preprocessor, mlModel+'/shap/', el, f'Waterfall_Plot_{i}', labels)

############################## DiCE ##########################

Ncount=30

Xdice = preprocessor.fit_transform(X)

constraints={}
    
Xdice = pd.DataFrame(Xdice, columns=labels)

desc=Xdice.describe()
for i in numeric_features:
    constraints[i]=[desc[i]['min'],desc[i]['max']]
Xdice['output'] = y
desc=Xdice.describe()
interval = [desc['output']['min'],desc['output']['max']]

X_train, X_test = train_test_split(Xdice,test_size=0.2,random_state=42)

dice_train = dice_ml.Data(dataframe=X_train, continuous_features=['age', 'bmi', 'children'], outcome_name='output')
    
m = dice_ml.Model(model=mod_grid.best_estimator_,backend='sklearn', model_type='regressor',func=None)
exp = dice_ml.Dice(dice_train,m)

query_instance = X_test.drop(columns="output")
'''query_instance = dice_ml.Data(dataframe=X_test.drop(columns="output"),
                 continuous_features=numeric_features,
                 outcome_name='output')'''
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount, desired_range=interval, permitted_range=constraints)

data = []
for cf_example in dice_exp.cf_examples_list:
    data.append(cf_example.final_cfs_df)

df_combined = pd.concat(data, ignore_index=True)
for i in range(len(df_combined)):
    df_combined.iloc[i] = df_combined.iloc[i] - X_test.iloc[i//Ncount]
df_combined.to_csv(path_or_buf=f'{mlModel}/dice/counterfactuals.csv', index=False, sep=',')
df_filtered = df_combined[df_combined['output'] != 0]
count_per_column = df_filtered.apply(lambda x: (x != 0).sum() * abs(df_filtered.loc[x != 0, 'output']).sum()/1000)
diff_per_column = df_filtered.apply(lambda x: (abs(x)).sum())
#relative_per_column = df_filtered.apply(lambda x: (abs(x)/abs(df_filtered['output'])/(x != 0)).sum())

original_stdout = sys.stdout
with open(f'{mlModel}/dice/count.txt', 'w') as f:
    sys.stdout = f
    print('\n--------------------- Counterfactual absolute counts:-------------------------')
    print(diff_per_column)
    print('\n--------------------- Counterfactual relative counts:-------------------------')
    print(diff_per_column/count_per_column)
            
        
sys.stdout = original_stdout

############################## RESULT ##########################

original_stdout = sys.stdout
with open('%s/res.txt' %(mlModel), 'w') as f:
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