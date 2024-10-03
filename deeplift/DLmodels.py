import keras
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import tensorflow as tf
import random
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import layers
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler
rng = 69420
#keras.utils.set_random_seed(42)
np.random.seed(rng)
tf.random.set_seed(rng)
random.seed(rng)

#matplotlib.use('qt5agg')

model = keras.Sequential()
model.add(layers.Dense(2048, activation="relu", input_shape=(13,)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1, activation="linear"))
model.add(layers.Activation("sigmoid"))


folder = os.path.dirname(os.path.abspath(__file__))
checkpoint_folder = Path(__file__).parent / "ckpt"
model_folder = Path(__file__).parent / "model"
dataset_folder = Path(__file__).parent / "heart.csv"
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
print(dataset_folder)
ds = pd.read_csv(dataset_folder, sep=',')
train, test = train_test_split(ds, test_size = 0.25, random_state=np.random.RandomState(rng))
y_train = train['output']
x_train = train.drop(columns=['output'], inplace=False)

model.summary()
feature_names = x_train.columns

numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, feature_names),
                                               ])

y_test = test['output']
x_test = test.drop(columns=['output'], inplace=False)
y_ds = ds['output']
x_ds = ds.drop(columns=['output'], inplace=False)
y_pos = ds[ds['output'] == 1]['output']
x_pos = ds[ds['output'] == 1].drop(columns=['output'], inplace=False)
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

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

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.25, shuffle=True, callbacks=[model_checkpoint_callback]) #erano 150 eopchs

eval_score = model.evaluate(x_test, y_test)

print(eval_score)

'''plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.ylim((0,1))
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.ylim((0.5,1))
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.show()

y_pred = model.predict(x_test)
y_pred = (y_pred >= 0.5).astype('int')
res = confusion_matrix(y_pred=y_pred, y_true=y_test)
df_cm = pd.DataFrame(res, range(2), range(2))
sn.heatmap(res,annot=True)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()'''





model_filepath = model_folder / 'modelLIME.h5'
model.save(model_filepath, overwrite=True)

explainer = LimeTabularExplainer(x_train, feature_names=feature_names,
                                            categorical_features=[],
                                            mode='classification',
                                            discretize_continuous=False)



random_numbers = np.random.randint(0, len(x_test), size=5)
explanation_instances = []
for i in random_numbers:
    explanation_instances.append(x_test[i])
for idx, instance in enumerate(explanation_instances):
    exp = explainer.explain_instance(instance.reshape(1,instance.shape[0]),
                                        model.predict,
                                        num_features=5,)