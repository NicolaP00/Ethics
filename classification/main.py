import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
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
from lime.lime_tabular import LimeTabularExplainer
import shap
import dice_ml
from dice_ml import Dice


rng = 69420


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

    
    if not os.path.exists(f'{mlModel}/lime'):
        os.makedirs(f'{mlModel}/lime')

    if not os.path.exists( mlModel + '/shap'):
        os.makedirs(mlModel + '/shap')

    if not os.path.exists(f'{mlModel}/dice'):
        os.makedirs(f'{mlModel}/dice')

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

    mod_grid = GridSearchCV(models_classification[mlModel]['estimator'], models_classification[mlModel]['param'], cv=5, return_train_score = False, scoring='f1_macro', n_jobs = 8)

    X_preprocessed = preprocessor.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size = 0.25, random_state=np.random.RandomState(rng))
   
    print('preprocessing done')
    model = Pipeline(steps=[('classifier', mod_grid)])
    model.fit(x_train, y_train)
    print('training done')
    feature_names = categorical_features + numeric_features

    target_pred = model.predict(x_test)

    f1 = metrics.f1_score(y_test, target_pred, average='micro')
    acc = metrics.accuracy_score(y_test, target_pred)


####################### GOLDEN STANDARDS #############################

    importance = []
    coefs = []
    if mlModel!='nb':
        importance = mod_grid.best_estimator_.feature_importances_
        coefs = pd.DataFrame(mod_grid.best_estimator_.feature_importances_,
                                columns=["Coefficients"],
                                index= headers[:-1])


    original_stdout = sys.stdout
    with open(f'{mlModel}/res.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Macro f1 score:', f1)
        print('Accuracy:', acc)
        print('\nFeature Scores: \n')
        print(coefs)
            
        print('\nBest Parameters used: ', mod_grid.best_params_)

    indexes = np.arange(len(headers[:-1]))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(indexes, headers[:-1], rotation=48)
    plt.savefig(f'{mlModel}/golden.png')
    plt.clf()
    plt.cla()
    plt.close()
        
    sys.stdout = original_stdout
    print('Results saved')

    #################### LIME Explanation ########################
    explainer = LimeTabularExplainer(x_train,
                                            feature_names=feature_names,
                                            categorical_features=[i for i, x in enumerate(headers) if x in categorical_features],
                                            mode='classification',
                                            discretize_continuous=False)
            
    random_numbers = np.random.randint(0, len(x_test), size=5)
    explanation_instances = []
    for i in random_numbers:
        explanation_instances.append(x_test[i])

    for idx, instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance, model.predict_proba, num_features=6,) #6 most signficant

        # save Lime explanation results
        exp.save_to_file(f'{mlModel}/lime/lime_explanation_{idx+1}.html')

    print('LIME done')

    ######################### SHAP VALUES #########################
    explainer = shap.Explainer(model['classifier'].best_estimator_.predict, shap.maskers.Independent(x_train), data=X_preprocessed)
    explanations = explainer(x_train)
    shap_values = explainer.shap_values(x_test)
    expected_value = np.array(model['classifier'].best_estimator_.predict(x_test)).mean()
    explanations.feature_names = [el for el in headers[:-1]]
    
    ######################### SHAP PLOTS ##########################
    shap.plots.bar(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Bar plot of SHAP values")
    plt.savefig(f'{mlModel}/shap/bar.png', bbox_inches='tight')
    plt.close()
    shap.plots.violin(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Violin plot of SHAP values")
    plt.savefig(f'{mlModel}/shap/violin.png', bbox_inches='tight')
    plt.close()
    shap.plots.beeswarm(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Dot plot of SHAP values")
    plt.savefig(f'{mlModel}/shap/bee.png', bbox_inches='tight')
    plt.close()
    shap.plots.heatmap(explanations, max_display=len(headers[:-1]), show=False)
    plt.title("Heatmap plot of SHAP values")
    plt.savefig(f'{mlModel}/shap/heatmap.png', bbox_inches='tight')
    plt.close()
    # Show some specific examples
    Showed_examples = 5 
    idx = np.random.randint(0, x_test.shape[0], Showed_examples)
    for i,el in enumerate(idx):
        shap.plots.waterfall(explanations[el, :], show=False)
        plt.title(f'Waterfall plot of random explanation')
        plt.savefig(f'{mlModel}/shap/waterfall_{i+1}.png', bbox_inches='tight')
        plt.close()
        shap.plots.decision(expected_value, shap_values[el, :], feature_names=[el for el in headers[:-1]], show=False)
        plt.title("Decision plot of random explanation")
        plt.savefig(f'{mlModel}/shap/decision_{i+1}.png', bbox_inches='tight')
        plt.close()
    
    print('shap done')

    ################ DiCE #######################

    Xdice = pd.DataFrame(X_preprocessed, columns=categorical_features+numeric_features)
    Ncount = 1

    constraints={}
    desc = Xdice.describe()

    for i in numeric_features:
        constraints[i]=[desc[i]['min'], desc[i]['max']]
    Xdice[headers[-1]] = y
    desc = Xdice.describe()

    X_train, X_test = train_test_split(Xdice, test_size=0.2, random_state=42)
    dice_train = dice_ml.Data(dataframe=Xdice, continuous_features=numeric_features, outcome_name=headers[-1])
    dice_model = dice_ml.Model(model=mod_grid.best_estimator_, backend="sklearn")

    exp = Dice(dice_train, dice_model)

    query_instance = X_test.drop(columns=headers[-1])
    
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount, desired_range=None, permitted_range=constraints)
    
    data = []
    for cf_example in dice_exp.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_combined = pd.concat(data, ignore_index=True)
    for i in range(len(df_combined)):
        df_combined.iloc[i] = df_combined.iloc[i] - X_test.iloc[i//Ncount]
    df_combined.to_csv(path_or_buf=f'{mlModel}/dice/conterfactuals.csv', index=False, sep=',')
    df_combined.dtypes
    df_filtered=df_combined[df_combined['output'] != 0]
    count_per_column = df_filtered.apply(lambda x: (x != 0).sum())
    diff_per_column = df_filtered.apply(lambda x: (abs(x)).sum())
    original_stdout = sys.stdout
    with open(f'{mlModel}/dice/count.txt','w') as f:
        sys.stdout = f
        print('\n--------------------- Counterfactual absolute counts: ---------------------')
        print(diff_per_column)
        print('\n--------------------- Counterfactual relative counts: ---------------------')
        print(diff_per_column/count_per_column)
    sys.stdout = original_stdout

    print('DiCE done')