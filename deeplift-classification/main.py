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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import layers
from keras import regularizers


rng = 69420
#keras.utils.set_random_seed(42)
np.random.seed(rng)
tf.random.set_random_seed(rng)
random.seed(rng)

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
model.add(layers.Dense(2048, activation="relu", input_shape=(13,)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1, activation="linear"))
model.add(layers.Activation("sigmoid"))
model.summary()

ds = pd.read_csv("./heart.csv")
train, test = train_test_split(ds, test_size = 0.25, random_state=np.random.RandomState(rng))
y_train = train['output']
x_train = train.drop(columns=['output'], inplace=False)
y_test = test['output']
x_test = test.drop(columns=['output'], inplace=False)
y_ds = ds['output']
x_ds = ds.drop(columns=['output'], inplace=False)
y_pos = ds[ds['output'] == 1]['output']
x_pos = ds[ds['output'] == 1].drop(columns=['output'], inplace=False)

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(lr=0.05),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=50, shuffle=True)

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(lr=0.01),
    metrics=["accuracy"],
)

checkpoint_filepath = './ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(x_train, y_train, batch_size=128, epochs=150, validation_split=0.25, shuffle=True, callbacks=[model_checkpoint_callback])

eval_score = model.evaluate(x_test, y_test)

print(eval_score)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.ylim((0,1))
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper right')
plt.savefig("./assets/model_loss.png")
plt.close()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.ylim((0.5,1))
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.savefig("./assets/model_accuracy.png")
plt.close()

y_pred = model.predict(x_test)
y_pred = (y_pred >= 0.5).astype('int')
res = confusion_matrix(y_pred=y_pred, y_true=y_test)
df_cm = pd.DataFrame(res, range(2), range(2))
sn.heatmap(res,annot=True)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title("Confusion Matrix")
plt.savefig("./assets/confusion_matrix.png")
plt.close()

model_filepath = './model/model.h5'
model.save(model_filepath, overwrite=True)



deeplift_model = kc.convert_model_from_saved_files(
        model_filepath,
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault
)

find_scores_layer_idx = 0

deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-2)

scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[x_test],
                                         batch_size=16,
                                         progress_update=1000))


scores_pd = pd.DataFrame(scores, columns=x_test.columns).abs()
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
scores_pd.mean().sort_values(ascending=True, inplace=False).plot.barh()
plt.title("Mean DeepLIFT scores")
plt.savefig('./assets/features_mean.png')
plt.close()
