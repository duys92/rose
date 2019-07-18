from rose.IRose import IRose
import numpy as np


class Rose(IRose):
    def test(self):
        pass


def model(x):
    return x * 2 + x + 2


def printname(name):
    print(name)


if __name__ == "__main__":
    rose = Rose.instance()
    storage = rose.obj('hd5py')

    x = np.linspace(0, 1, 1000)
    y = model(x)
    y += np.random.normal(0, 0.25, *x.shape)

    xy = np.vstack((x,y))

    storage.save(filename='output', dataset=xy, name="/mydataset/xy", attrs={"att1": 1, 'att2': 'no'}, overwrite=True)
    y = storage.load(filename='output', name='/mydataset/xy', get_values=False)

    print(y.attrs.keys())

    tf = rose.obj('tensorflow')

    model = [
        ['dense', {'units': 4, 'activation': 'relu', 'input_dim': 1}],
        ['dropout', {'rate': 0.2}],
        ['dense', {'units': 24, 'activation': 'relu'}],
        ['dense', {'units': 1, 'activation': 'sigmoid'}]
    ]

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(xy[0], xy[1], test_size=0.33)

    builder = tf.builder().build_model(model=model, show_summary=True)

    # builder.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # builder.fit(x=dset['x'], y=dset['y'], epochs=100, batch_size=32)
