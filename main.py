import tensorflow as tf
from aon_model import aon_model_crnn


class Params:
    input_shape = (100, 100)
    channel_shape = 3


def run_main():
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        model = aon_model_crnn.AONmodelcrnn(Params.input_shape,
                                            Params.channel_shape,
                                            graph)


if __name__ == "__main__":
    run_main()
