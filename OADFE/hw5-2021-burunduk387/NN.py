from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau
import tensorflow as tf

import os
#FATAL output only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class NN_model(object):
    def model(self, input_shape):
        #Простенькая моделька
        seq = Sequential()
        seq.add(Dense(100, input_dim=input_shape, activation=tf.nn.relu6))
        seq.add(Dense(50, activation=tf.nn.relu6))
        seq.add(Dense(1))
        return seq
    
    def train(self, x, y, x1, y1, model):
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        try:
            model.load_weights('NN_model/weights.hdf5')
        except:
            pass
        #Поменять verbose на 1, чтобы видеть номер Epoch и стало ли лучше mse
        checkpoint = ModelCheckpoint("NN_model/weights.hdf5",
                                     monitor="val_mse",
                                     verbose=0,
                                     save_best_only=True)
        #Поменять verbose на 2, чтобы видеть, что происходит
        model.fit(x[:], y[:],
                            batch_size=48,
                            epochs=400,
                            verbose=0,
                            callbacks=[checkpoint,
                                       TerminateOnNaN(),
                                       ReduceLROnPlateau(monitor="val_loss",
                                                         factor=0.75,
                                                         patience=10)],
                            validation_data=(x1[:], y1[:]),
                            shuffle=True)
        model.load_weights('NN_model/weights.hdf5')
        return model

    def fit(self, x_tr, x_te, y_tr, y_te):
        if not os.path.exists('NN_model/model_best'):
            seq = self.model(x_tr.shape[1])
            seq = self.train(x_tr, y_tr, x_te, y_te, seq)
            seq.save('NN_model/model_best')
            self.seq = seq
        else:
            self.seq = load_model("NN_model/model_best")
            
    def predict(self, X):
        return self.seq.predict(X)


