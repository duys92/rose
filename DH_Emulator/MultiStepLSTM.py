from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import StandardScaler
from scipy.stats import  multivariate_normal
import lasio as las
import deepdish

class WirelineLog(object):
    """
    Help wrapper around lasio

    """
    las_file = None
    df = None

    def read(self, path):
        """
        Return DataFrame of entire las file
        :param path:
        :return:
        """
        self.las_file = las.read(path)
        self.df = self.las_file.df()


class ResetStates(Callback):

    def on_epoch_end(self, epoch, logs=None):
        print("resetting states")
        self.model.reset_states()

class MultiStepLTSM(Sequential):
    """
    Custom

    """

    done = None
    scaler = None
    history=None

    # predictions = {'names': data}
    predictions = {}

    # data = {'name': {'x' : x, 'y' : y}}
    data = {}

    # p_values = {'name': {'data': data, 'mean' : mean, 'cov':cov}}
    p_values = {}

    def __init__(self, batch_size=32, epochs=25, look_back=100, look_ahead=1, dropout=0.1, hidden_n=120, **kwargs):
        super(MultiStepLTSM, self).__init__()
        # if not isinstance(data_dict, dict):
        #     raise IndexError("data input is not in dictionary format")
        # self.data = data_dict
        self.batch_size = batch_size
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.dropout = dropout
        self.hidden_n = hidden_n
        self.epochs = epochs

    def save_all(self, path, overwrite=True, include_optimizer=True):
        self.save(path, overwrite=overwrite, include_optimizer=include_optimizer)

        # save all predictions, data, p_values in seperate file along model
        all_data = {
            'data': self.data,
            'predictions': self.predictions,
            'p_values': self.p_values,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'look_back': self.look_back,
            'look_ahead': self.look_ahead,
            'dropout': self.dropout,
            'hidden_n': self.hidden_n
        }

        deepdish.io.save(path+".h5", all_data)

    def load_all(self, path):
        all_data = deepdish.io.load(path)

        for k,v in all_data.items():
            self.__dict__[k] = v


    def insert_data(self, data_dict):
        for key, value in data_dict.items():
            tmpx, tmpy = self._prepare_seq2seq_data(data_dict[key], look_back=self.look_back,
                                                    look_ahead=self.look_ahead)
            tmpx, tmpy = self.clip_data((tmpx, tmpy))
            self.data[key] = {'x': tmpx, 'y': tmpy}

    def standardize(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # print("Scaler Mean: {}\nData Mean:{}\n\nScaler SD: {}\nData SD: {}".format(
        #     scaler.mean_, np.mean(data), scaler.std_, np.std(data)
        # ))
        return data, scaler

    def build_model(self, iterations=0):

        self.add(LSTM(self.hidden_n,
                      input_shape=(self.batch_size, self.look_back),
                      batch_size=self.batch_size,
                      batch_input_shape=(self.batch_size, self.look_back, self.look_ahead),
                      return_sequences=True,
                      stateful=True
                      ))

        self.add(Dropout(self.dropout))

        for i in range(iterations):
            self.add(LSTM(self.hidden_n,
                          return_sequences=True,
                          stateful=True
                          ))

            self.add(Dropout(self.dropout))

        self.add(LSTM(
            self.hidden_n,
            return_sequences=False,
            stateful=True
        ))
        self.add(Dense(1))

        optimizer = Adam(lr=0.002)
        self.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        self.summary()

    def train_model(self):
        x_train, y_train = self.data['train']['x'], self.data['train']['y']

        try:
            x_valid1, y_valid1 = self.data['validation_1']['x'], self.data['validation_1']['y']
            validation_data = (x_valid1, y_valid1[:, 0])
        except KeyError as err:
            print("Did not find validation data set to be used on training data, using keras defaults.")
            validation_data = None

        # for i in range(self.epochs):
        #     print("Epoch", i + 1, "/", self.epochs)
        #
        #     history = self.fit(x_train, y_train[:, 0],
        #                             batch_size=self.batch_size,
        #                             epochs=1,
        #                             verbose=1,
        #                             shuffle=False,
        #                             validation_data=validation_data,
        #                             callbacks=[EarlyStopping(patience=1, verbose=1), ])
        #     self.reset_states()

        self.history = self.fit(x_train, y_train[:, 0],
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           verbose=1,
                           shuffle=False,
                           validation_data=validation_data,
                           callbacks=[EarlyStopping(patience=1, verbose=1), ResetStates()])

        self.done = 1

    def predict_on(self, data, name, inverse_transform=False):

        pred = self.predict(data, batch_size=self.batch_size)

        if inverse_transform is True and self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)
        self.predictions[name] = pred

    def plot_error(self, true_name, pred_name, i1=None, i2=None, roll_predictions=-1):

        true = self.data[true_name]['y']
        predictions = self.predictions[pred_name]
        self.calc_pvalues(true_name=true_name, pred_name=pred_name, output_name=true_name+"_"+pred_name)

        p_values = self.p_values[true_name+"_"+pred_name]['p_values']
        predictions = np.roll(predictions, roll_predictions)

        b = i1 if i1 is not None else 0
        e = i2 if i2 is not None else len(true)
        if e < b:
            tmp = b
            b = e
            e = tmp

        # max = true.max() if true.max() > predictions.max() else predictions.max()
        # min = true.min() if true.min() > predictions.min() else predictions.min()

        plt.figure(figsize=(30, 10))

        plt.subplot(2, 1, 1)
        plt.plot(true[b:e, 0], label="True Values", alpha=0.5)
        plt.plot(predictions[b:e], label="Predicted", linestyle="--")
        error = abs(abs(predictions) - abs(true[:, 0]))
        plt.plot(error[b:e], label="Error", linewidth=0.5, c='g', alpha=0.8)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(p_values[b:e], label="Probability of Anomaly", linestyle='--', c='g')
        plt.ylim(p_values.min(), p_values.max() + 1)
        plt.legend()
        plt.show()


    def _prepare_seq2seq_data(self, dataset, look_back, look_ahead):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - look_ahead):
            input_seq = dataset[i:(i + look_back)]
            output_seq = dataset[i + look_back:(i + look_back + look_ahead)]
            dataX.append(input_seq)
            dataY.append(output_seq)
        dataX = np.reshape(np.array(dataX), [-1, look_back, 1])
        dataY = np.reshape(np.array(dataY), [-1, look_ahead, 1])
        return dataX, dataY

    def clip_data(self, data_tuple):
        x, y = data_tuple
        # For stateful lstm the batch_size needs to be fixed before hand.
        # We also need to ernsure that all batches shud have the same number of samples. So we drop the last batch as it has less elements than batch size
        if self.batch_size > 1:
            n_train_batches = int(len(x) / self.batch_size)
            len_d = n_train_batches * self.batch_size
            if len_d < len(x):
                x = x[:len_d]
                y = y[:len_d]
            return x, y

    def calc_pvalues(self, true_name, pred_name, output_name=None):
        pred = self.predictions[pred_name]
        true = self.data[true_name]['y']
        true = true[:, 0][:, 0]

        error_vectors = np.zeros(pred.shape)
        n_cols = pred.shape[1]
        print(pred.shape, true.shape, n_cols)
        for i in range(n_cols):
            error_vectors[:, i] = true - pred[:, 0]

        mean = np.mean(error_vectors, axis=0)
        cov = np.cov(error_vectors, rowvar=False)

        p_values = multivariate_normal.logpdf(error_vectors, mean, cov)

        new_key =true_name+'_'+pred_name if output_name is None else output_name

        self.p_values[new_key] = {'p_values': p_values, 'mean': mean, 'cov': cov}


if __name__ == '__main__':
    model = load_model('ccl-model.keras', custom_objects={"MultiStepLTSM" : MultiStepLTSM})
    model.load_all('ccl-model.keras.h5')
    print(model.__dict__['batch_size'])


