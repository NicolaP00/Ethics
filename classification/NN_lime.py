import numpy as np
import pandas as pd
import sys
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adamax
from keras.losses import categorical_crossentropy

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, recall_score
from lime.lime_tabular import LimeTabularExplainer
import warnings
import os
from keras.backend import one_hot, argmax


def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def predict_proba_wrapper(X):
    return model.predict(X)

def mapp(df, mapping):
    df['Type'] = df['Type'].map(mapping)

def nn_model(input_shape):
  input = Input(shape=input_shape)
  x = Dense(2048, activation='relu')(input)
  x = Dense(128, activation='relu')(x)
  x = Dense(2, activation='softmax')(x)
  
  model = Model(input, x)
  model.compile(loss=categorical_crossentropy, optimizer = Adamax(1e-3), metrics = 'MSE')
  model.summary()

  return model

if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    pathCSV = 'heart.csv'
    dataset = pd.read_csv(pathCSV, sep=',')
    headers = dataset.columns.tolist()
    print(headers)

    if not os.path.exists('assets/NN/lime'):
        os.makedirs('assets/NN/lime')

    X = dataset[headers[:-1]]
    data = {
    'Fault': dataset[headers[-1]],
    'Normal': (~dataset[headers[-1]].astype(bool)).astype(int)
    }

    y = pd.DataFrame(data)
    y = np.array(y)


    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp']
    numeric_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'thall']

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
                                          ('label', OrdinalEncoder())
                                          ])  

    preprocessor = ColumnTransformer(
                                 transformers=[
                                                ('cat', categorical_transformer, categorical_features),
                                               ('num', numeric_transformer, numeric_features),
                                               ])

    k = 5           #CAMBIATO, PRIMA ERA 10
    kf = KFold(n_splits=k, random_state=None)

    model = nn_model((13,))

    accuracy = []
    f1 = []
    recall = []

    print('preprocessing done')

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index,:],y[test_index,:]

        data_train_lime = preprocessor.fit_transform(data_train)
        data_test_lime = preprocessor.transform(data_test)
        model.fit(data_train_lime, target_train, steps_per_epoch=len(data_train_lime)//16, epochs=10)

        print('training done')

            #feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = categorical_features + numeric_features
        target_pred = model.predict(data_test_lime)

        recall.append(recall_score(target_test, one_hot(argmax(target_pred, axis=-1), num_classes=target_pred.shape[-1]), average='micro'))
        accuracy.append(accuracy_score(target_test, one_hot(argmax(target_pred, axis=-1), num_classes=target_pred.shape[-1])))
        f1.append(f1_score(target_test, one_hot(argmax(target_pred, axis=-1), num_classes=target_pred.shape[-1]), average='micro'))

            #################### LIME Explanation ########################
        explainer = LimeTabularExplainer(data_train_lime,
                                            feature_names=feature_names,
                                            categorical_features=[i for i, x in enumerate(headers) if x in categorical_features],
                                            mode='classification',
                                            discretize_continuous=False)
            
        random_numbers = np.random.randint(0, len(data_test_lime)-1, size=5)
        explanation_instances = []
        for i in random_numbers:
                explanation_instances.append(data_test_lime[i])
    for idx, instance in enumerate(explanation_instances):
        exp = explainer.explain_instance(instance,
                                        model.predict,
                                        num_features=5,) #5 most signficant

        # save Lime explanation results
        exp.save_to_file(f'assets/NN/lime/lime_explanation_{idx}.html')
 
    print('lime finished')

####################### GOLDEN STANDARDS #############################


    original_stdout = sys.stdout
    with open('assets/NN/res.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Model errors and report:-------------------------')
        print('Recall:', np.mean(recall))
        print('Accuracy', np.mean(accuracy))
        print('Macro f1 score', np.mean(f1))
        
    sys.stdout = original_stdout
    print('Results saved')