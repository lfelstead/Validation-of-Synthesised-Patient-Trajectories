import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from tensorflow import keras
from validation_util import *

np.random.seed(1) # make results reproducible
num_classes = 2

# read in data 
real_test = pd.read_csv("labhourformattest_72.csv")
real_train = pd.read_csv("labhourformatrain_72.csv")
timegan = pd.read_csv("labtimeGAN-generated.csv")
deepecho_100 = pd.read_csv("deepEchoGenerated-100epoch.csv")
deepecho_500 = pd.read_csv("deepEchoGenerated-500epoch.csv")

# split data into training and testing sets
testids = list(set(timegan["hadm_id"]))[:200]
timegan_test = timegan[timegan["hadm_id"].isin(testids)]
timegan_train = timegan[~timegan["hadm_id"].isin(testids)]

deepecho_100= deepecho_100.loc[:, ~deepecho_100.columns.isin(['gender', 'anchor_age',"gender", "insurance","marital_status","race","index","hour"])]
deepecho_500= deepecho_500.loc[:, ~deepecho_500.columns.isin(['gender', 'anchor_age',"gender", "insurance","marital_status","race","index","hour"])]

# Input data for the Keras LSTM layer has 3 dimensions: (M, T, N), where
# M - number of examples (2D: sequences of timesteps x features),
# T - sequence length (timesteps) and
# N - number of features (input_dim)

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def runLSTMRNN(real_train, real_test, fake_train, fake_test):
    X_trainr, y_trainr = formatdata(real_train, 0)
    X_trainf, y_trainf = formatdata(fake_train, 1)
    X_train = np.concatenate((X_trainr, X_trainf), axis=0)
    y_train = np.concatenate((y_trainr, y_trainf), axis=0)
    np.random.shuffle(X_train) 
    np.random.shuffle(y_train) 
    X_testr, y_testr = formatdata(real_test, 0)
    X_testf, y_testf = formatdata(fake_test, 1)
    X_test = np.concatenate((X_testr, X_testf), axis=0)
    y_test = np.concatenate((y_testr, y_testf), axis=0)
    np.random.shuffle(X_test) 
    np.random.shuffle(y_test) 
    
    model = make_model(input_shape=X_train.shape[1:])
    
    epochs = 50
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )
    
    model = keras.models.load_model("best_model.h5")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    return model, history, test_loss, test_acc

model, history, test_loss, test_acc = runLSTMRNN(real_test, deepecho_100, real_train, deepecho_500)
model1, history1, test_loss1, test_acc1 = runLSTMRNN(real_test, timegan_test, real_train, timegan_train)

# write results to file
f = open("validation_results.txt", "a")
f.write("LSTM RNN VALIDATION RESULTS")
f.write("DEEPECHO: accuracy-"+str(test_acc)+", loss-"+str(test_loss))
f.write("TIMEGAN: accuracy-"+str(test_acc1)+", loss-"+str(test_loss1))
f.close()


