from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import sys
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder
from matplotlib import pyplot as plt
import dice_ml
from dice_ml import Dice
from math import sqrt

rng = 69420
keras.utils.set_random_seed(rng)
np.random.seed(rng)
tf.random.set_seed(rng)
random.seed(rng)

def nn_model(input_shape):
    model = keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mse", optimizer="adam" , metrics=["mae"])
    return model


def pred_fn(model):
    def pred(instance):
        print(instance.shape)
        print(type(instance))
        print(model.predict(instance).shape)
        print(type(model.predict(instance)))
        return model.predict(instance)[:,0]
    return pred

model = nn_model((6,))

if not os.path.exists('NNmodel/dice'):
        os.makedirs('NNmodel/dice')


if not os.path.exists('NNmodel/ckpt'):
    os.mkdir('NNmodel/ckpt')

ds = pd.read_csv('./insurance.csv', sep=',')

feature_names = ds.columns
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

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

train, test = train_test_split(ds, test_size = 0.25, random_state=np.random.RandomState(rng))
y_train = train['charges']
y_test = test['charges']
x_train = preprocessor.fit_transform(train.drop(columns=['charges'], inplace=False))
x_test = preprocessor.transform(test.drop(columns=['charges'], inplace=False))

print(x_train)

checkpoint_filepath = 'NNmodel/ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_mae',
    mode='max',
    save_best_only=True)

history = model.fit(x_train, y_train, batch_size=128, epochs=1500, validation_split=0.25, shuffle=True, callbacks=[model_checkpoint_callback]) 

print('training done')

# PLOT LOSS

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([1.5e7,5e7])
plt.legend(['train', 'valid'], loc='upper right')
plt.savefig('NNmodel/dice/train_loss.png',bbox_inches='tight')
plt.close()

# PLOT ACCURACY

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model mae')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.savefig('NNmodel/dice/train_mae.png',bbox_inches='tight')
plt.close()


x_dice = pd.DataFrame(x_train, columns=numeric_features+categorical_features)
Ncount = 3

constraints={}
desc = x_dice.describe()

for i in numeric_features:
    constraints[i]=[desc[i]['min'], desc[i]['max']]
x_dice[feature_names[-1]] = y_train
desc = x_dice.describe()
interval = [desc[feature_names[-1]]['min'], desc[feature_names[-1]]['max']]

dice_train = dice_ml.Data(dataframe=x_dice, continuous_features=numeric_features, outcome_name=feature_names[-1])
dice_model = dice_ml.Model(model=model, model_type='regressor', backend="TF2")

exp = Dice(dice_train, dice_model)

query_instance = pd.DataFrame(x_test, columns=numeric_features+categorical_features)
print(query_instance.describe())
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount, desired_range=interval, permitted_range=constraints)

data = []
for cf_example in dice_exp.cf_examples_list:
    data.append(cf_example.final_cfs_df)

df_combined = pd.concat(data, ignore_index=True)
for i in range(len(df_combined)):
    df_combined.iloc[i] = df_combined.iloc[i] - query_instance.iloc[i//Ncount]
df_combined.to_csv(path_or_buf=f'NNmodel/dice/conterfactuals.csv', index=False, sep=',')
df_combined.dtypes
df_filtered=df_combined[df_combined[feature_names[-1]] != 0]
#count_per_column = df_filtered.apply(lambda x: (x != 0).sum())
count_per_column = df_filtered.apply(lambda x: (x != 0).sum() * abs(df_filtered.loc[x != 0, feature_names[-1]]).sum()/1000000)

diff_per_column = df_filtered.apply(lambda x: (abs(x)).sum())
original_stdout = sys.stdout

print('RMSE : ', sqrt(history.history['val_loss'][-1]))
print('MAE : ', history.history['val_mae'][-1])

with open(f'NNmodel/dice/count.txt','w') as f:
    sys.stdout = f
    print('\n--------------------- Counterfactual absolute counts: ---------------------')
    print(diff_per_column)
    print('\n--------------------- Counterfactual relative counts: ---------------------')
    print(diff_per_column/count_per_column)
sys.stdout = original_stdout
