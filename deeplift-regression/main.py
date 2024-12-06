from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import deeplift
import tensorflow as tf
import random
import os
from deeplift.conversion import kerasapi_conversion as kc
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers


rng = 69420
#keras.utils.set_random_seed(42)
np.random.seed(rng)
tf.random.set_random_seed(rng)
random.seed(rng)

def nn_model(input_shape):
    model = keras.Sequential()
    model.add(Input(input_shape))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mse", optimizer="adam" , metrics=["mae"])
    return model


model = nn_model((6,))

if not os.path.exists('assets/NN/ckpt'):
    os.makedirs('assets/NN/ckpt')
if not os.path.exists('assets/NN/model'):
    os.makedirs('assets/NN/model')

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

checkpoint_filepath = 'assets/NN/ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_mean_absolute_error',
    mode='min',
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
plt.savefig('assets/NN/train_loss.png',bbox_inches='tight')
plt.close()

# PLOT ACCURACY

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model mae')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.savefig('assets/NN/train_mae.png',bbox_inches='tight')
plt.close()


model_filepath = 'assets/NN/model/model.h5'
model.save(model_filepath, overwrite=True)



deeplift_model = kc.convert_model_from_saved_files(
        model_filepath,
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault
)

find_scores_layer_idx = 0

deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-1)

scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[x_test],
                                         batch_size=16,
                                         progress_update=1000))


scores_pd = pd.DataFrame(scores, columns=feature_names[:-1]).abs()
scores_max = scores_pd.max().max()
scores_pd = scores_pd/scores_max
meds = scores_pd.median()
meds.sort_values(ascending=False, inplace=True)
scores_pd = scores_pd[meds.index]
scores_pd.boxplot(figsize=(15,10), grid=False)
#plt.axhline(y=0, color='k')
plt.ylim((0,1))
plt.title("Importance of features according to DeepLift")
plt.savefig('./assets/NN/features_importance.png')
plt.close()
scores_pd.mean().sort_values(ascending=True, inplace=False).plot.barh()
plt.title("Mean DeepLIFT scores")
plt.savefig('./assets/NN/features_mean.png')
plt.close()
