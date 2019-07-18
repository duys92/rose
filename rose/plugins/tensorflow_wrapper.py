from plugins.admin.Petal import Petal
import tensorflow as tf


class ModelBuilder(tf.keras.Sequential):
    """
    Wrapper to easily build a keras model

    """
    _model_list = None

    def build_model(self, model, show_summary=False):

        self._parse_model(model)

        if show_summary:
            print(self.summary())

        return self

    def _parse_model(self, model):
        if not isinstance(model, list):
            print("Invalid format, check syntax")
            return

        for sequence in model:
            layer = sequence[0]
            params = sequence[1]
            self.add(eval("tf.keras.layers.{}".format(layer.capitalize()))(**params))


class TensorFlow(Petal):
    builder = ModelBuilder

    def run(self, *args, **kwargs):
        pass











