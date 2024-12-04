import keras
import shap
import numpy as np
import pandas as pd
import seaborn as sn
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
from sklearn.preprocessing import  StandardScaler, OrdinalEncoder

rng = 69420
#keras.utils.set_random_seed(42)
np.random.seed(rng)
tf.random.set_seed(rng)
random.seed(rng)

def pred_fn(model):
  def pred(instance):
    print(instance.shape)
    print(type(instance))
    print(model.predict(instance).shape)
    print(type(model.predict(instance)))
    return model.predict(instance)[:,0]
  return pred

#matplotlib.use('qt5agg')

model = keras.Sequential()
model.add(layers.Dense(2048, activation="relu", input_shape=(6,)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1, activation="linear"))
#model.add(layers.Activation("sigmoid"))


if not os.path.exists('NNmodel/lime'):
        os.makedirs('NNmodel/lime')

if not os.path.exists('NNmodel/shap'):
        os.makedirs('NNmodel/shap')


folder = os.path.dirname(os.path.abspath(__file__))
checkpoint_folder = Path(__file__).parent / "NNmodel"
dataset_folder = Path(__file__).parent / "insurance.csv"
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)
ds = pd.read_csv(dataset_folder, sep=',')
train, test = train_test_split(ds, test_size = 0.25, random_state=np.random.RandomState(rng))
y_train = train['charges']
X = train.drop(columns=['charges'], inplace=False)

model.summary()
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']
feature_names = numeric_features+categorical_features

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

y_test = test['charges']
x_test = test.drop(columns=['charges'], inplace=False)
y_ds = ds['charges']
x_ds = ds.drop(columns=['charges'], inplace=False)
y_pos = ds[ds['charges'] == 1]['charges']
x_pos = ds[ds['charges'] == 1].drop(columns=['charges'], inplace=False)
x_train = preprocessor.fit_transform(X)
x_test = preprocessor.transform(x_test)

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adamax(lr=0.05),
    metrics=["mean_squared_error"],
)

#model.fit(x_train, y_train, batch_size=32, epochs=50, shuffle=True)

checkpoint_filepath = './ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint_callback]) #erano 1500 eopchs

eval_score = model.evaluate(x_test, y_test)

print(eval_score)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper right')
plt.savefig('NNmodel/loss.png')

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model_mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.savefig('NNmodel/mse.png')




model_filepath = checkpoint_folder / 'modelLIME.h5'
model.save(model_filepath, overwrite=True)
explainer = LimeTabularExplainer(x_train,
                                         feature_names=feature_names,
                                         class_names=['charges'],
                                         mode='regression',
                                         discretize_continuous=True)



random_numbers = np.random.randint(0, len(x_test), size=5)
explanation_instances = []
for i in random_numbers:
    explanation_instances.append(x_test[i])
for idx, instance in enumerate(explanation_instances):
    exp = explainer.explain_instance(instance,
                                        pred_fn(model),
                                        num_features=5,)
    
    exp.save_to_file(f'NNmodel/lime/lime_explanation_{idx+1}.html')


############################## SHAP ##########################

ex = shap.Explainer(model, x_train[:5])
explanations = ex(x_train[:5])

#shap_explanation = ex(x_test)[:,:,0]

#shap_values = ex.shap_values(x_test, nsamples=5)

#shap_explanation.feature_names = [el for el in X.columns]

fig, ax = plt.subplots()
print(feature_names)
print(np.array(x_train).shape)
shap.summary_plot(explanations, x_train[-5], feature_names=feature_names, plot_type='violin')
plt.tight_layout()
fig.savefig('NNmodel/shap/violin.png')
plt.close()

fig, ax = plt.subplots()
shap.summary_plot([[s] for s in shap_values], features=x_test, plot_type='dot')
plt.tight_layout()
fig.savefig('NNmodel/shap/dot.png')
plt.close()

fig, ax = plt.subplots()
shap.summary_plot([[s] for s in shap_values], features=x_test, plot_type='bar')
plt.tight_layout()
fig.savefig('NNmodel/shap/bar.png')
plt.close()

fig, ax = plt.subplots()
shap.plots.heatmap(shap_explanation)
plt.tight_layout()
plt.title("Features Influence's heatmap")
fig.savefig('NNmodel/shap/heatmap.png')
plt.close()

examples = 5
idx = np.random.randint(0, x_test.shape[0], examples)

for i in range(len(idx)):
    shap.plots.waterfall(shap_explanation[idx[i], :])
    ax.set_title(f"Example {i+1}")
    plt.tight_layout()
    plt.savefig(f'NNmodel/shap/waterfall_{i+1}.png')
    plt.close()

    shap.plots.decision(ex.expected_value,shap_explanation.values[idx[i],:])
    ax.set_title(f"Example {i+1}")
    plt.tight_layout()
    plt.savefig(f'NNmodel/shap/decision_{i+1}.png')
    plt.close()
    
