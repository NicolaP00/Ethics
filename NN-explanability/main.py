import keras
import shap
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import tensorflow as tf
import random
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import layers
from keras import regularizers

rng = 69420
keras.utils.set_random_seed(42)
np.random.seed(rng)
tf.random.set_seed(rng)
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
    optimizer=keras.optimizers.Adamax(learning_rate=0.05),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=50, shuffle=True)

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(learning_rate=0.01),
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
plt.title("Confusion Matrix")
plt.savefig("./assets/confusion_matrix.png")
plt.close()

print(np.array(x_test.columns))

ex = shap.KernelExplainer(model.predict, x_train)

shap_explanation = ex(x_test)[:,:,0]

shap_values = ex.shap_values(x_test, nsamples=500)

shap_explanation.feature_names = [el for el in x_test.columns]

shap.summary_plot(shap_values.reshape(x_test.shape), features=x_test, plot_type='violin', plot_size=(10,10), show=False)
plt.title("Violin plot of SHAP values")
plt.savefig("./assets/violin.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values.reshape(x_test.shape), features=x_test, plot_type='dot', plot_size=(10,10), show=False)
plt.title("Dot plot of SHAP values")
plt.savefig("./assets/dot.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values.reshape(x_test.shape), features=x_test, plot_type='bar', plot_size=(10,10), show=False)
plt.title("Bar plot of (the magnitude of) SHAP values")
plt.savefig("./assets/bar.png", bbox_inches="tight")
plt.close()

shap.plots.heatmap(shap_explanation, plot_width=10, show=False)
plt.title("Heatmap of SHAP explanations")
plt.savefig("./assets/heatmap.png", bbox_inches="tight")
plt.close()

examples = 5
idx = np.random.randint(0, x_test.shape[0], examples)

for i in range(x_test.shape[0]):
    shap.plots.waterfall(shap_explanation[i, :], show=False)
    plt.title(f"Waterfall SHAP explanation of example #{i+1}")
    plt.savefig(f"./assets/waterfall_{i+1}.png", bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10,10))
    shap.plots.decision(ex.expected_value,shap_explanation.values[i,:], feature_names=np.array(x_test.columns), auto_size_plot=False, show=False)
    plt.title(f"SHAP decision plot of example #{i+1}")
    plt.savefig(f"./assets/decision_{i+1}.png", bbox_inches="tight")
    plt.close()

