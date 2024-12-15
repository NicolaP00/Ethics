import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from customModels import PolyClassifier, CustomClassifier
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
from dice_ml import Dice

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
   

if __name__ == "__main__":

    if(len(sys.argv)<1):
        print("ERROR! Usage: python scriptName.py\n")
              
        sys.exit(1)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"


    dataset = pd.read_csv('heart.csv', sep=',')
    #conterfactual_dataset = f'dice_results/{targetId}_lr_{ds}_counterfactuals.csv'
    if not os.path.exists('lc/adv'):
        os.makedirs('lc/adv')
    headers = dataset.columns.tolist()
    X = dataset[headers[:-1]]
    y = dataset[headers[-1]]

    categorical_features = ['sex', 'fbs', 'exng']
    numeric_features = ['age', 'cp', 'trtbps', 'chol', 'restecg', 'thalachh', 'oldpeak', 'slp', 'caa', 'thall']
    labels = numeric_features + categorical_features

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
                                          ('ordinal', OrdinalEncoder(handle_unknown='error'))
                                          ])    

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)
                                              ])


    models_classification = {
        'lc': {'name': 'Linear Classifier',
               'estimator': PolyClassifier(adv='no'),
              },
        'fake': {'name':'Custom',
                 'estimator':PolyClassifier(adv='lf'),
                },
        's':{'name':'selector',
             'estimator':RandomForestClassifier()}
    }

    X = preprocessor.fit_transform(X)
    X = pd.DataFrame(X, columns=labels)

    mae = []
    mse = []
    rmse = []
    mape = []
    f1 = []

    k = 10
    kf = KFold(n_splits=k, random_state=None)

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index] , y[test_index]
        model = models_classification['lc']['estimator']    
        _ = model.fit(data_train, target_train)

    ################ DiCE #################
    
    Ncount=2
    '''constraints={}
    desc = X.describe()
    for i in numeric_features:
        constraints[i]=[desc[i]['min'],desc[i]['max']]'''
    X['output'] = y
    d = dice_ml.Data(dataframe=X, continuous_features=labels, outcome_name='output')
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(d, m, method="random")
    query_instance = X.drop(columns=['output'])
    e1 = exp.generate_counterfactuals(X_train, total_CFs=Ncount,
                                    desired_class="opposite",
                                    features_to_vary=numeric_features)

    
    data = []
    for cf_example in e1.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_counterfactual = pd.concat(data, ignore_index=True)                  # i counterfactuals
    y_counterfactual = df_counterfactual['output']                          #ò'output dei cf

    Xcount_train,Xcount_test,ycount_train,ycount_test = train_test_split(df_counterfactual,y_counterfactual,random_state=42)
    OOD_train = np.concatenate([np.zeros(X_train.shape[0]), np.ones(Xcount_train.shape[0])])
    OOD_test = np.concatenate([np.zeros(X_test.shape[0]), np.ones(Xcount_test.shape[0])])
    OOD = np.concatenate([OOD_train, OOD_test])
    indices = np.random.permutation(len(OOD))
    OOD = OOD[indices]
    merged_train = pd.concat([X_train, Xcount_train], ignore_index=True)    #cf e originali insieme
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
        fake = models_classification['fake']['estimator']    
        _ = fake.fit(data_train, target_train)

    ################ SELECTOR ###############
    
    for train_index , test_index in kf.split(merged):
        data_train , data_test = merged.iloc[train_index,:],merged.iloc[test_index,:]
        target_train , target_test = OOD[train_index] , OOD[test_index]
        selec = models_classification['s']['estimator']
        _ = selec.fit(data_train,target_train)

    ################ LIME ####################

    explainer = LimeTabularExplainer(merged_train.values,
                                        feature_names=labels,
                                        #categorical_features=categorical_features,
                                        mode='classification',
                                        discretize_continuous=False,
                                        random_state=42)
    
    random_numbers = np.random.randint(0, merged_test.shape[0], size=5)
    explanation_instances = []
    for i in random_numbers:
        explanation_instances.append(merged_test.values[i])


    for idx,instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,CustomClassifier(model,fake,selec).predict_couple,num_features=5)
        #lime_folder = os.path.join(output_folder, 'lime_explanations')


        # save Lime explanation results
        exp.save_to_file(f'lc/adv/lime_explanation_{idx}.html')

    
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
    plt.savefig('lc/adv/bar-ad-good.png')
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('lc/adv/res-ad-good.txt', 'w') as f:
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
                            index= labels)

    # plot feature importance
    indexes = np.arange(len(labels))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, labels, rotation=48)
    plt.savefig('lc/adv/bar-ad-fake.png')
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT 2 #################################

    original_stdout = sys.stdout
    with open('lc/adv/res-ad-fake.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print(coefs)
            
        #print('\nBest Parameters used: ', mod_grid.best_params_)
        
    sys.stdout = original_stdout
    print('Results saved')
