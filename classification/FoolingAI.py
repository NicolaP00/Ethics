import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from customModels import PolyClassifier
from sklearn.metrics import f1_score, accuracy_score

import warnings
import os

rng = 1


def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py loss\n")
              
        sys.exit(1)
    nome_script, loss = sys.argv

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"


    dataset = pd.read_csv('heart.csv', sep=',')

    if not os.path.exists('lc/fooling'):
        os.makedirs('lc/fooling')

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
                                          ('ordinal', OrdinalEncoder(handle_unknown='error'))])    

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)
                                              ])

    param_lr = [{'fit_intercept':[True,False]}]

    models_classification = {
        'lc': {'name': 'Linear Classifier',
               'estimator': PolyClassifier(adv=loss),
               'param': param_lr,
              },
    }


    X_preprocessed = preprocessor.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size = 0.25, random_state=np.random.RandomState(rng))

    model = models_classification['lc']['estimator']
    _ = model.fit(x_train, y_train)

    target_pred = model.predict(x_test)
    f1 = f1_score(y_test, target_pred, average='micro')
    acc = accuracy_score(y_test, target_pred)


    
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
    plt.savefig('lc/fooling/bar-%s.png'%loss)
    plt.clf()
    plt.cla()
    plt.close()

################ WRITE RES IN A TXT #################################

    original_stdout = sys.stdout
    with open('lc/fooling/res-%s.txt'%loss, 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')

        print('accuracy score', acc)
        print('f1 score', f1)
        print('\nFeature Scores: \n')
        print(coefs)
                

            
    sys.stdout = original_stdout
    print('Results saved')


