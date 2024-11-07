import keras
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
from keras import layers
from keras import regularizers


rng = 69420
#keras.utils.set_random_seed(42)
np.random.seed(rng)
tf.random.set_random_seed(rng)
random.seed(rng)

matplotlib.use('qt5agg')

checkpoint_folder = './ckpt'
model_folder = './model'
assets_folder = './assets'

if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
if not os.path.exists(assets_folder):
    os.mkdir(assets_folder)

model = keras.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(6,)))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="relu"))
model.add(layers.Dense(1, activation="linear"))
model.summary()
scaler = StandardScaler()
ds = pd.read_csv("./insurance.csv")
train, test = train_test_split(ds, test_size = 0.25, random_state=np.random.RandomState(rng))
y_train = scaler.fit_transform(train[['charges']].values)
x_train = train.drop(columns=['charges'], inplace=False)
y_test = scaler.transform(test[['charges']].values)
x_test = test.drop(columns=['charges'], inplace=False)

feature_names = ds.columns

categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

numeric_transformer = Pipeline(
                               steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler',  StandardScaler()),
                                     ]
                              )

categorical_transformer = Pipeline(
                                   steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('ordinal', OrdinalEncoder(handle_unknown='error')),
                                         ]
                                  )

preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numerical_features),
                                               ('cat', categorical_transformer, categorical_features),
                                              ]
                                )

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)


model.compile(
    loss='mse',
    optimizer=keras.optimizers.SGD(lr=1e-4),
    metrics=["mae"],
)

checkpoint_filepath = './ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_mae',
    mode='min',
    save_best_only=True)

history = model.fit(x_train, y_train, batch_size=64, epochs=15000, validation_split=0.25, shuffle=True, callbacks=[model_checkpoint_callback])

eval_score = model.evaluate(x_test, y_test)

print(eval_score)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper right')
plt.show()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model_mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.show()


model_filepath = './model/model.h5'
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
plt.savefig('./assets/features_importance.png')
plt.close()
print(scores_pd.describe())