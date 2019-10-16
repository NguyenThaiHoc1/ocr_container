import functools
import sys
import tensorflow as tf

from . import label_map
from . import sync_attention_wrapper as attention_class


# loading number class


def get_shape_list_tensor(input_tensor):
    """
    This function using for get shape from input images
    :param input_tensor:
    :return: list
    """
    static_shape = input_tensor.shape.as_list()
    dynamic_shape = tf.shape(input_tensor)
    combined_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_shape[index])
    return combined_shape


def create_weight_init(shape, trainable=True, name="weights", initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    weight_init = tf.get_variable(shape=shape,
                                  dtype=tf.float32,
                                  initializer=initializer,
                                  trainable=trainable,
                                  name=name)
    return weight_init


def create_bias_init(shape, trainable=True, name="biases", initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0)
    bias_init = tf.get_variable(shape=shape,
                                dtype=tf.float32,
                                initializer=initializer,
                                trainable=trainable,
                                name=name)
    return bias_init


def convolution_layer(layer_name, inputs, outputs, kernel_size=[3, 3], strides=[1, 1],
                      paddings=[1, 1], trainbles=True, resuse=None):
    return None


def fully_connected_layer(layer_name, inputs, out_nodes):
    """
    Args:
        inputs: 4D, 3D or 2D tensor, if 4D tensor,
        out_nodes: number of output neutral units
    """
    shape = get_shape_list_tensor(inputs)
    if len(shape) == 4:
        size = shape[1] * shape[2] * shape[3]
    else:  # convert the last dimention to out_nodes size
        size = shape[-1]

    with tf.variable_scope(layer_name):
        w = create_weight_init(shape=[size, out_nodes])
        b = create_bias_init(shape=[out_nodes])
        flat_x = tf.reshape(inputs, [-1, size])
        x = tf.matmul(flat_x, w, name='matmul')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x)
        return x


class AONmodelcrnn:

    def __init__(self, img_size_input, shape_channel, graph):
        """

        :param img_size_input:
        :param shape_channel:
        """
        """
                Variables init which thought in model
        """
        self.img_size = img_size_input
        self.shape_channel = shape_channel
        self.graph_model = graph

        self.output_cnn_features = None
        self.output_rnn_features = None
        self.output_filtergate = None
        self.train_output_dict = None
        self.pred_output_dict = None

        """
                Varibales input model initalize
        """
        self.inputs = None
        self.groundtruth_text_placeholder = None
        self.training_placeholder = None
        self.learning_rate = None

        self.sess = None
        self.saver = None

        self.optimizer = None
        self.loss_batch = None
        self.train_op = None

        """
                Create input model placeholder
        """
        with tf.variable_scope(name_or_scope="CREATE_INPUT", reuse=tf.AUTO_REUSE):
            self.setup_placeholder()

        """
                Create architecture model 
        """
        with tf.variable_scope(name_or_scope="ARCHITECTURE_MODEL", reuse=tf.AUTO_REUSE):
            self.inference_model()

        """
                Setup optimizer and setting batch_normalization 
        """
        with tf.variable_scope(name_or_scope="OPTIMIZER_BATCH_NORMALIZATION", reuse=tf.AUTO_REUSE):
            self.setup_batchnormalization()

        """
                Initialize tensorflow is a final step
        """
        with tf.variable_scope(name_or_scope="INITIALIZE_TENSORFLOW", reuse=tf.AUTO_REUSE):
            self.sess, self.saver = self.setup_tensorflow()

    def setup_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.img_size[0], self.img_size[1], self.shape_channel],
                                     name="input_image_model")
        self.groundtruth_text_placeholder = tf.placeholder(shape=[None, ], dtype=tf.string, name="label_truth")
        self.training_placeholder = tf.placeholder(tf.bool, name="is_training_model")
        self.learning_rate = tf.placeholder(dtype=tf.float32)

    def building_cnn(self):
        print(">>    CNN MODEL  ----------------------")
        with tf.variable_scope(name_or_scope="CNN"):
            model = None

            """
                    CONV1D HIDDEN LAYER 1
            """
            in_channel = get_shape_list_tensor(self.inputs)[-1]
            kernelsize = [3, 3]
            strides = [1, 1, 1, 1]
            padding_height, padding_width = [1, 1]
            paddings = [[0, 0], [padding_height, padding_height],
                        [padding_width, padding_width], [0, 0]]
            name = "conv_1"
            output_shape = 64
            trainable = True
            with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
                shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                shape_bias = [output_shape]

                weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                model = tf.pad(self.inputs, paddings=paddings)
                model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                model = tf.nn.bias_add(model, bias_init, name="bias_add")
                model = tf.nn.relu(model, name="relu")
                model = tf.layers.batch_normalization(inputs=model, axis=-1)
            print(">>     Conv 1:\t\t{}".format(model))

            """
                    MAXPOOLING SIZE 1
            """
            kernelsize_pool = [1, 2, 2, 1]
            strides = [1, 2, 2, 1]
            padding_height, padding_width = [0, 0]
            paddings = [[0, 0], [padding_height, padding_height],
                        [padding_width, padding_width], [0, 0]]
            pool_name = "max_pool_1"
            with tf.variable_scope(pool_name):
                model = tf.pad(model, paddings=paddings)
                model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                       name='max_pool')
            print(">>     Max Pooling 1:\t{}".format(model))

            """
                    CONV HIDDEN LAYER 2
            """
            in_channel = get_shape_list_tensor(model)[-1]
            kernelsize = [3, 3]
            strides = [1, 1, 1, 1]
            padding_height, padding_width = [1, 1]
            paddings = [[0, 0], [padding_height, padding_height],
                        [padding_width, padding_width], [0, 0]]
            name = "conv_2"
            output_shape = 128
            trainable = True
            with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
                shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                shape_bias = [output_shape]

                weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                model = tf.pad(model, paddings=paddings)
                model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                model = tf.nn.bias_add(model, bias_init, name="bias_add")
                model = tf.nn.relu(model, name="relu")
                model = tf.layers.batch_normalization(inputs=model, axis=-1)
            print(">>     Conv 2:\t\t{}".format(model))

            """
                     MAXPOOLING SIZE 2
            """
            kernelsize_pool = [1, 2, 2, 1]
            strides = [1, 2, 2, 1]
            padding_height, padding_width = [1, 1]
            paddings = [[0, 0], [padding_height, padding_height],
                        [padding_width, padding_width], [0, 0]]
            pool_name = "max_pool_2"
            with tf.variable_scope(pool_name):
                model = tf.pad(model, paddings=paddings)
                model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                       name='max_pool')
            print(">>     Max Pooling 2:\t{}".format(model))

            """
                        CONV HIDDEN LAYER 3
            """
            in_channel = get_shape_list_tensor(model)[-1]
            kernelsize = [3, 3]
            strides = [1, 1, 1, 1]
            padding_height, padding_width = [1, 1]
            paddings = [[0, 0], [padding_height, padding_height],
                        [padding_width, padding_width], [0, 0]]
            name = "conv_3"
            output_shape = 256
            trainable = True
            with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
                shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                shape_bias = [output_shape]

                weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                model = tf.pad(model, paddings=paddings)
                model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                model = tf.nn.bias_add(model, bias_init, name="bias_add")
                model = tf.nn.relu(model, name="relu")
                model = tf.layers.batch_normalization(inputs=model, axis=-1)
            print(">>     Conv 3:\t\t{}".format(model))

            """
                        CONV HIDDEN LAYER 4
            """
            in_channel = get_shape_list_tensor(model)[-1]
            kernelsize = [3, 3]
            strides = [1, 1, 1, 1]
            padding_height, padding_width = [1, 1]
            paddings = [[0, 0], [padding_height, padding_height],
                        [padding_width, padding_width], [0, 0]]
            name = "conv_4"
            output_shape = 256
            trainable = True
            with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
                shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                shape_bias = [output_shape]

                weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                model = tf.pad(model, paddings=paddings)
                model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                model = tf.nn.bias_add(model, bias_init, name="bias_add")
                model = tf.nn.relu(model, name="relu")
                model = tf.layers.batch_normalization(inputs=model, axis=-1)
            print(">>     Conv 4:\t\t{}".format(model))
            return model

    def building_aon_core(self):
        def get_character_placement_cluse(inputs, reuse=None):
            model = None
            with tf.variable_scope('placement_cluse'):
                """
                        CONV1D HIDDEN LAYER 1
                """
                in_channel = get_shape_list_tensor(inputs)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_1"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(inputs, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 1:\t{}".format(model))

                """
                           MAXPOOLING SIZE 1
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 2, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_1"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 1:\t{}".format(model))

                """
                          CONV1D HIDDEN LAYER 2
                """
                in_channel = get_shape_list_tensor(model)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_2"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 2:\t{}".format(model))

                """
                        MAXPOOLING SIZE 2
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 2, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_2"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 2:\t{}".format(model))

                model = tf.reshape(model, shape=[-1, 64, 512])
                print(">>     Reshape Placement Cluse 1:\t{}".format(model))

                model = tf.transpose(model, perm=[0, 2, 1])
                print(">>     Tranpose Placement Cluse 1:\t{}".format(model))

                model = fully_connected_layer(layer_name='fc_1', inputs=model, out_nodes=23)
                print(">>     Fully Connected 1:\t{}".format(model))
                model = tf.reshape(model, shape=[-1, 512, 23])
                print(">>     Reshape Placement Cluse 2:\t{}".format(model))
                model = tf.transpose(model, perm=[0, 2, 1])
                print(">>     Tranpose Placement Cluse 2:\t{}".format(model))

                model = fully_connected_layer(layer_name='fc_2', inputs=model, out_nodes=4)  # ??? 4
                print(">>     Fully Connected 2:\t{}".format(model))
                model = tf.reshape(model, shape=[-1, 23, 4])
                print(">>     Reshape Placement Cluse 3:\t{}".format(model))
                model = tf.nn.softmax(model, axis=2, name='softmax')
                print(">>     Softmax:\t{}".format(model))
                return model

        def get_feature_sequence(inputs, reuse=None):
            with tf.variable_scope('shared_stack_conv', reuse=reuse):
                print(inputs)
                model = None
                """
                        CONV1D HIDDEN LAYER 1
                """
                in_channel = get_shape_list_tensor(inputs)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_1"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(inputs, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 1:\t{}".format(model))
                """
                         MAXPOOLING SIZE 1
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 1, 1]
                padding_height, padding_width = [1, 0]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_1"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 1:\t{}".format(model))

                """
                         CONV2D HIDDEN LAYER 2
                """
                in_channel = get_shape_list_tensor(model)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_2"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 2:\t{}".format(model))

                """
                        MAXPOOLING SIZE 2
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 1, 1]
                padding_height, padding_width = [0, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_2"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 2:\t{}".format(model))

                """
                        CONV1D HIDDEN LAYER 3
                """
                in_channel = get_shape_list_tensor(model)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_3"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 3:\t{}".format(model))

                """
                       MAXPOOLING SIZE 3
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 1, 1]
                padding_height, padding_width = [1, 0]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_3"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 3:\t{}".format(model))

                """
                        CONV1D HIDDEN LAYER 4
                """
                in_channel = get_shape_list_tensor(model)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_4"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 4:\t{}".format(model))

                """
                         MAXPOOLING SIZE 4
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 1, 1]
                padding_height, padding_width = [0, 0]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_4"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 4:\t{}".format(model))

                """
                        CONV1D HIDDEN LAYER 5
                """
                in_channel = get_shape_list_tensor(model)[-1]
                kernelsize = [3, 3]
                strides = [1, 1, 1, 1]
                padding_height, padding_width = [1, 1]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                name = "conv_5"
                output_shape = 512
                trainable = True
                with tf.variable_scope(name_or_scope=name, reuse=reuse):
                    shape_weight = [kernelsize[0], kernelsize[1], in_channel, output_shape]
                    shape_bias = [output_shape]

                    weight_init = create_weight_init(shape=shape_weight, trainable=trainable)
                    bias_init = create_bias_init(shape=shape_bias, trainable=trainable)

                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.conv2d(input=model, filter=weight_init, strides=strides, padding='VALID', name="conv")
                    model = tf.nn.bias_add(model, bias_init, name="bias_add")
                    model = tf.nn.relu(model, name="relu")
                    model = tf.layers.batch_normalization(inputs=model, axis=-1)
                print(">>     Conv feature sequence 5:\t{}".format(model))

                """
                         MAXPOOLING SIZE 5
                """
                kernelsize_pool = [1, 2, 2, 1]
                strides = [1, 2, 1, 1]
                padding_height, padding_width = [0, 0]
                paddings = [[0, 0], [padding_height, padding_height],
                            [padding_width, padding_width], [0, 0]]
                pool_name = "max_pool_4"
                with tf.variable_scope(pool_name):
                    model = tf.pad(model, paddings=paddings)
                    model = tf.nn.max_pool(value=model, ksize=kernelsize_pool, strides=strides, padding='VALID',
                                           name='max_pool')
                print(">>     Max Pooling feature sequence 5:\t{}".format(model))

                model = tf.squeeze(model, axis=1, name='squeeze')
                print(">>     Squeeze feature sequence:\t{}".format(model))
                return model

        def building_bilstm(inputs, layer_name, hidden_units):
            with tf.variable_scope(layer_name):
                fw_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units)
                bw_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units)

                inter_output, state_output = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell,
                                                                             cell_bw=bw_lstm_cell,
                                                                             inputs=inputs,
                                                                             dtype=tf.float32)
                #  (output_fw, output_bw), (output_state_fw, output_state_bw)

                output = tf.concat(inter_output, 2)
                output_state_c = tf.concat((state_output[0].c, state_output[1].c), 1)
                output_state_h = tf.concat((state_output[0].h, state_output[1].h), 1)
                output_state = tf.contrib.rnn.LSTMStateTuple(state_output[0], state_output[1])

                return output, output_state

        with tf.name_scope('AON_core') as scope:
            feature_horizontal = get_feature_sequence(inputs=self.output_cnn_features)
            feature_sequence_1, _ = building_bilstm(inputs=feature_horizontal, layer_name="bilstm_1", hidden_units=256)
            feature_sequence_reverse_1 = tf.reverse(feature_sequence_1, axis=[1])
            print(">>     RNN-LSTM Horizontal feature sequence:\t{}".format(feature_sequence_reverse_1))

            featute_vertical = get_feature_sequence(inputs=tf.image.rot90(self.output_cnn_features), reuse=True)
            feature_sequence_2, _ = building_bilstm(inputs=featute_vertical, layer_name='bilstm_2', hidden_units=256)
            feature_sequence_reverse_2 = tf.reverse(feature_sequence_2, axis=[1])
            print(">>     RNN-LSTM Vertical feature sequence:\t{}".format(feature_sequence_reverse_2))

            character_placement_cluse = get_character_placement_cluse(inputs=self.output_cnn_features)
            print(">>     Character Placement Cluse:\t{}".format(character_placement_cluse))

            res_dict = {
                "feature_seq_1": feature_sequence_1,
                "feature_seq_reverse_1": feature_sequence_reverse_1,
                "feature_seq_2": feature_sequence_2,
                "feature_seq_reverse_2": feature_sequence_reverse_2,
                "character_placement_cluse": character_placement_cluse
            }

            return res_dict

    def building_filtergate(self):
        """
            the filter gate (FG) for combing four feature sequences with the character sequence.
        """
        # get feature sequence one foward
        feature_sequence_1 = self.output_rnn_features["feature_seq_1"]
        feature_sequence__reverse_1 = self.output_rnn_features["feature_seq_reverse_1"]
        feature_sequence_2 = self.output_rnn_features["feature_seq_2"]
        feature_sequence_reverse_2 = self.output_rnn_features["feature_seq_reverse_2"]
        character_placement_cluse = self.output_rnn_features["character_placement_cluse"]

        with tf.name_scope('FG') as scope:
            A = feature_sequence_1 * tf.tile(tf.reshape(character_placement_cluse[:, :, 0], [-1, 23, 1]), [1, 1, 512])
            B = feature_sequence__reverse_1 * tf.tile(tf.reshape(character_placement_cluse[:, :, 1], [-1, 23, 1]),
                                                      [1, 1, 512])
            C = feature_sequence_2 * tf.tile(tf.reshape(character_placement_cluse[:, :, 2], [-1, 23, 1]), [1, 1, 512])
            D = feature_sequence_reverse_2 * tf.tile(tf.reshape(character_placement_cluse[:, :, 3], [-1, 23, 1]),
                                                     [1, 1, 512])
            model = A + B + C + D
            model = tf.tanh(model)
            print(">>     Filter Gate:\t{}".format(model))
            return model

    def building_attention_decoder(self, sync_system=False):
        batch_size = get_shape_list_tensor(self.output_filtergate)[0]  # get batch_size
        sync = sync_system
        attention_wrapper_class = attention_class.SyncAttentionWrapper if sync else tf.contrib.seq2seq.AttentionWrapper

        def decoder(inputs_encoder, helper, scope, batch_size, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=512, memory=inputs_encoder)
                cell = tf.contrib.rnn.GRUCell(num_units=512)
                attn_cell = attention_wrapper_class(
                    cell, attention_mechanism, output_attention=False,
                    attention_layer_size=256,
                )

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, num_classes, reuse=reuse
                )

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)  # batch_size
                )

                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=100
                )

                return outputs[0]

        with tf.name_scope('attention_decoder'):
            go_token = 0
            end_token = 1
            unk_token = 2

            start_tokens = tf.fill([batch_size, 1], tf.constant(go_token, tf.int64))
            end_tokens = tf.fill([batch_size, 1], tf.constant(end_token, tf.int64))

            label_map_obj = label_map.LabelMap()
            num_classes = label_map_obj.num_classes + 2
            embedding_fn = functools.partial(tf.one_hot, depth=num_classes)

            text_labels, text_lengths = label_map_obj.text_to_labels(self.groundtruth_text_placeholder,
                                                                     pad_value=end_token,
                                                                     return_lengths=True)

            if not sync:
                train_input = tf.concat([start_tokens, start_tokens, text_labels], axis=1)
                train_target = tf.concat([start_tokens, text_labels, end_tokens], axis=1)
                train_input_lengths = text_lengths + 2
            else:
                train_input = tf.concat([start_tokens, text_labels], axis=1)
                train_target = tf.concat([text_labels, end_tokens], axis=1)
                train_input_lengths = text_lengths + 1

            max_num_step = tf.reduce_max(train_input_lengths)
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                embedding_fn(train_input), tf.to_int32(train_input_lengths)
            )
            train_outputs = decoder(self.output_filtergate, train_helper, 'decoder', batch_size)
            train_logits = train_outputs.rnn_output  # logits
            train_labels = train_outputs.sample_id
            weights = tf.cast(tf.sequence_mask(train_input_lengths, max_num_step), tf.float32)
            train_loss = tf.contrib.seq2seq.sequence_loss(
                logits=train_outputs.rnn_output, targets=train_target, weights=weights,
                name='train_loss'
            )  # loss of train
            train_probabilities = tf.reduce_max(
                tf.nn.softmax(train_logits, name='probabilities'),
                axis=-1,
            )

            # validation
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding_fn,
                start_tokens=tf.fill([batch_size], go_token),
                end_token=end_token
            )
            pred_outputs = decoder(self.output_filtergate, pred_helper, 'decoder', batch_size, reuse=True)
            pred_logits = pred_outputs.rnn_output
            pred_labels = pred_outputs.sample_id
            eval_loss = tf.contrib.seq2seq.sequence_loss(
                logits=pred_outputs.rnn_output, targets=train_target, weights=weights,
                name='eval_loss'
            )

            train_output_dict = {
                'loss': train_loss,
                'logits': train_logits,
                'labels': train_labels,
                'predict_text': label_map_obj.labels_to_text(train_labels),
                'probabilities': train_probabilities,
            }

            pred_output_dict = {
                'logits': pred_logits,
                'labels': pred_labels,
                'predict_text': label_map_obj.labels_to_text(pred_labels),
            }

            return train_output_dict, pred_output_dict

    def inference_model(self):
        print(">>------INFERENCE MODEL------")
        self.output_cnn_features = self.building_cnn()
        self.output_rnn_features = self.building_aon_core()
        self.output_filtergate = self.building_filtergate()  # This is endcoder
        self.train_output_dict, self.pred_output_dict = self.building_attention_decoder(sync_system=True)
        print("-------------DONE-------------")

    def setup_batchnormalization(self):
        update_ops = tf.GraphKeys.UPDATE_OPS
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize()  # this here is a loss function

    def setup_tensorflow(self):
        """
                This function where using initializa tensorflow
        :return:
        """
        print("Python: {}".format(sys.version))
        print("Tensorflow: {}".format(tf.__version__))

        config_tensorflow = tf.ConfigProto()
        config_tensorflow.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)  # Saving checkpoints patien time

        # create a Session
        sess = tf.Session(graph=self.graph_model)

        # run all global variable
        sess.run(init)

        return sess, saver
