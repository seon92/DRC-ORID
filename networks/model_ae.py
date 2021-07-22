import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


# -------------------------------------------------------------------------------------------------------------------- #
#                                                    Custom Layers                                                     #
# -------------------------------------------------------------------------------------------------------------------- #
def downsample(filters, size, apply_batchnorm=True):
    num_layer = len(filters)
    initializer = tf.random_normal_initializer(0., 0.02)
    regularizer = tf.keras.regularizers.l2(0.001)
    result = tf.keras.Sequential()

    for i in range(num_layer):
        if i == 0:  # downsample
            result.add(
                tf.keras.layers.Conv2D(filters[i], size[i], strides=2, padding='same',
                                       kernel_initializer=initializer, kernel_regularizer=regularizer,
                                       use_bias=False))
        else:  # convolution
            result.add(
                tf.keras.layers.Conv2D(filters[i], size[i], strides=1, padding='same',
                                       kernel_initializer=initializer, kernel_regularizer=regularizer,
                                       use_bias=False))

        if apply_batchnorm:
            result.add(layers.BatchNormalization())

        result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    num_layer = len(filters)
    initializer = tf.random_normal_initializer(0., 0.02)
    regularizer = tf.keras.regularizers.l2(0.001)
    result = tf.keras.Sequential()

    for i in range(num_layer):
        if i == 0:   # upsample
            result.add(
                layers.Conv2DTranspose(filters[i], size[i], strides=2,
                                       padding='same',
                                       kernel_initializer=initializer, kernel_regularizer=regularizer,
                                       use_bias=False))
        else:        # convolution
            result.add(
                layers.Conv2D(filters[i], size[i], strides=1,
                              padding='same',
                              kernel_initializer=initializer, kernel_regularizer=regularizer,
                              use_bias=False))

        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.LeakyReLU())     ## NOTE !!! I MODIFIED HERE
    return result


def conv2d(filters, size, apply_batchnorm=True):
    num_layer = len(filters)
    initializer = tf.random_normal_initializer(0., 0.02)
    regularizer = tf.keras.regularizers.l2(0.001)
    result = keras.Sequential()

    for i in range(num_layer):
        result.add(
            layers.Conv2D(filters[i], size[i], strides=1, padding='same',
                          kernel_initializer=initializer, kernel_regularizer=regularizer,
                          use_bias=False))

        if apply_batchnorm:
            result.add(layers.BatchNormalization())

        result.add(layers.ReLU())

    return result


# -------------------------------------------------------------------------------------------------------------------- #
#                                                   Network Models                                                     #
# -------------------------------------------------------------------------------------------------------------------- #
def autoencoder_v1(config):

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([512], [2]),  # (bs, 1, 1, 512)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, 512])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([512], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer, kernal_regularizer=tf.keras.regularizers.l2(),
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()

    z_base = encoder(base)
    z_ref = encoder(reference)

    z_age_base = tf.squeeze(tf.math.l2_normalize(z_base[:,:,:,:256], axis=-1))
    z_age_ref = tf.squeeze(tf.math.l2_normalize(z_ref[:,:,:,:256], axis=-1))

    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, z_age_base, z_age_ref], name='AE_v1')


def autoencoder_v2_attribute_provided(config):

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([512], [2]),  # (bs, 1, 1, 512)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, 516])  # attribute label is 4 dim feature vector
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([512], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer, kernal_regularizer=tf.keras.regularizers.l2(),
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])
    base_attribute = layers.Input(shape=[4,])
    ref_attribute = layers.Input(shape=[4,])

    encoder = get_encoder()
    decoder = get_decoder()

    z_base = encoder(base)
    z_ref = encoder(reference)

    z_age_base = tf.squeeze(tf.math.l2_normalize(z_base[:,:,:,:256], axis=-1))
    z_age_ref = tf.squeeze(tf.math.l2_normalize(z_ref[:,:,:,:256], axis=-1))

    z_base_att = tf.concat([z_base, tf.reshape(base_attribute, [-1, 1, 1, 4])], axis=-1)
    z_ref_att = tf.concat([z_ref, tf.reshape(ref_attribute, [-1, 1, 1, 4])], axis=-1)

    gen_base = decoder(z_base_att)
    gen_ref = decoder(z_ref_att)

    return tf.keras.Model(inputs=[base, reference, base_attribute, ref_attribute], outputs=[gen_base, gen_ref, z_age_base, z_age_ref], name='AE_v2_att_provided')


def autoencoder_v3(config):

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([1024], [2]),  # (bs, 1, 1, 512)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, 1024])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([1024], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()

    z_base = encoder(base)
    z_ref = encoder(reference)

    z_age_base = tf.squeeze(tf.math.l2_normalize(z_base[:,:,:,:256], axis=-1))
    z_age_ref = tf.squeeze(tf.math.l2_normalize(z_ref[:,:,:,:256], axis=-1))

    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, z_age_base, z_age_ref], name='AE_v3')


def autoencoder_v4_normalized_decoding(config):

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([1024], [2]),  # (bs, 1, 1, 512)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)
        x_age = tf.math.l2_normalize(x[:, :, :, :256], axis=-1)
        x_attr = x[:,:,:,256:]
        x = tf.concat([x_age, x_attr], axis=-1)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, 1024])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([1024], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()

    z_base = encoder(base)
    z_ref = encoder(reference)

    z_age_base = tf.squeeze(z_base[:,:,:,:256])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:256])

    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, z_age_base, z_age_ref], name='AE_v4_normalized_decoding')


def autoencoder_v5_wh128(config):

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 64, 64, 64)
            downsample([128, 128], [3, 3]),  # (bs, 32, 32, 128)
            downsample([256, 256], [3, 3]),  # (bs, 16, 16, 256)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 512)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 1024], [3, 3]),  # (bs, 2, 2, 512)
            downsample([2048], [2]),  # (bs, 1, 1, 512)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, 2048])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([1024], [4], apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([256], [4]),  # (bs, 32, 32, 256)
            upsample([128], [4]),  # (bs, 64, 64, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()

    z_base = encoder(base)
    z_ref = encoder(reference)

    z_age_base = tf.squeeze(tf.math.l2_normalize(z_base[:,:,:,:256], axis=-1))
    z_age_ref = tf.squeeze(tf.math.l2_normalize(z_ref[:,:,:,:256], axis=-1))

    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, z_age_base, z_age_ref], name='AE_v5_wh128')


def autoencoder_v6_comparator_added(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='AE_v6')


def autoencoder_v6_comparator_added_wh128(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 64, 64, 64)
            downsample([128, 128], [3, 3]),  # (bs, 32, 32, 128)
            downsample([256, 256], [3, 3]),  # (bs, 16, 16, 256)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([256], [4]),  # (bs, 32, 32, 256)
            upsample([128], [4]),  # (bs, 64, 64, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='AE_v6')


def autoencoder_v6_comparator_sigmoid(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='AE_v6')


def gan_test(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='gan_test')


def spherical_v0(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:, :, :, age_feat_dim:],axis=[1, 2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:, :, :, age_feat_dim:], axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='spherical_v0')

def spherical_v0_non_act(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)
        x = layers.Conv2D(1024, 1, 1)(x)
        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:, :, :, age_feat_dim:],axis=[1, 2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:, :, :, age_feat_dim:], axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='spherical_v0')


def spherical_v1(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')


    def get_projector():
        input = layers.Input(shape=[z_dim-age_feat_dim, ])
        x = layers.Conv1D(256, 1, 1)(input)
        x = layers.ReLU()(x)
        x = layers.Conv1D(128, 1, 1)(x)
        return keras.Model(inputs=input, outputs=x, name='projector')


    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    projector = get_projector()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    # projector
    base_z = projector(tf.squeeze(z_base[:,:,:,age_feat_dim:], axis=[1,2]), axis=-1)
    ref_z = projector(tf.squeeze(z_ref[:,:,:,age_feat_dim:], axis=[1,2]), axis=-1)


    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(base_z),
                                                             tf.math.l2_normalize(ref_z),
                                                             order_forward, order_reverse], name='spherical_v1')


def v6_vgg_repulsive(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = age_feat_dim + config.chain_feat_dim

    def get_encoder():
        vgg16 = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                         weights='imagenet')
        vgg16.trainable = True
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim * 2, ])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:, :, :, :age_feat_dim], axis=[1, 2])
    z_age_ref = tf.squeeze(z_ref[:, :, :, :age_feat_dim], axis=[1, 2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:,:,:,age_feat_dim:chain_feat_dim], axis=[1,2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:,:,:,age_feat_dim:chain_feat_dim],axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='v6_vgg_unsupervised_chain')

def vgg_repulsive_v2(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = age_feat_dim + config.chain_feat_dim

    def get_encoder():
        vgg16 = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                         weights='imagenet')
        vgg16.trainable = True
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [2, 2], [1, 1])(vgg16.output)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            layers.Conv2D(z_dim, [1, 1], [1, 1]),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim * 2, ])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:, :, :, :age_feat_dim], axis=[1, 2])
    z_age_ref = tf.squeeze(z_ref[:, :, :, :age_feat_dim], axis=[1, 2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:,:,:,age_feat_dim:chain_feat_dim], axis=[1,2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:,:,:,age_feat_dim:chain_feat_dim],axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='vgg_repulsive_v2')


def vgg_repulsive_v3(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = age_feat_dim + config.chain_feat_dim

    def get_encoder():
        vgg16_v2 = get_vgg16_v2(config)
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [2, 2], [1, 1])(vgg16_v2.output)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16_v2.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            layers.Conv2D(z_dim, [1, 1], [1, 1]),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim * 2, ])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:, :, :, :age_feat_dim], axis=[1, 2])
    z_age_ref = tf.squeeze(z_ref[:, :, :, :age_feat_dim], axis=[1, 2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:,:,:,age_feat_dim:chain_feat_dim], axis=[1,2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:,:,:,age_feat_dim:chain_feat_dim],axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='vgg_repulsive_v3')


def vgg_repulsive_v4(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = age_feat_dim + config.chain_feat_dim

    def get_encoder():
        vgg16 = get_vgg16(config)
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [2, 2], [1, 1])(vgg16.output)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            layers.Conv2D(z_dim, [1, 1], [1, 1]),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim * 2, ])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:, :, :, :age_feat_dim], axis=[1, 2])
    z_age_ref = tf.squeeze(z_ref[:, :, :, :age_feat_dim], axis=[1, 2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:,:,:,age_feat_dim:chain_feat_dim], axis=[1,2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:,:,:,age_feat_dim:chain_feat_dim],axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='vgg_repulsive_v4')


def vgg_repulsive_v5(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = age_feat_dim + config.chain_feat_dim

    def get_encoder():
        vgg16 = get_vgg16(config)
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [2, 2], [1, 1])(vgg16.output)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            layers.Conv2D(z_dim, [1, 1], [1, 1]),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim * 2, ])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_projector():
        input = layers.Input(shape=[z_dim-age_feat_dim, ])
        x = layers.Conv1D(256, 1, 1)(input)
        x = layers.ReLU()(x)
        x = layers.Conv1D(256, 1, 1)(x)
        return keras.Model(inputs=input, outputs=x, name='projector')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    projector = get_projector()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:, :, :, :age_feat_dim], axis=[1, 2])
    z_age_ref = tf.squeeze(z_ref[:, :, :, :age_feat_dim], axis=[1, 2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    # projector
    base_h = projector(tf.squeeze(z_base[:,:,:,age_feat_dim:chain_feat_dim], axis=[1,2]), axis=-1)
    ref_h = projector(tf.squeeze(z_ref[:,:,:,age_feat_dim:chain_feat_dim], axis=[1,2]), axis=-1)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(base_h),
                                                             tf.math.l2_normalize(ref_h),
                                                             order_forward, order_reverse], name='vgg_repulsive_v5')


def repulsive_v6(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = age_feat_dim + config.chain_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            layers.Conv2D(z_dim, [1, 1], [1, 1]),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim * 2, ])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_projector():
        input = layers.Input(shape=[z_dim-age_feat_dim, ])
        x = layers.Conv1D(256, 1, 1)(input)
        x = layers.ReLU()(x)
        x = layers.Conv1D(256, 1, 1)(x)
        return keras.Model(inputs=input, outputs=x, name='projector')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    projector = get_projector()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:, :, :, :age_feat_dim], axis=[1, 2])
    z_age_ref = tf.squeeze(z_ref[:, :, :, :age_feat_dim], axis=[1, 2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    # projector
    base_h = projector(tf.squeeze(z_base[:, :, :, age_feat_dim:], axis=[1, 2]), axis=-1)
    ref_h = projector(tf.squeeze(z_ref[:, :, :, age_feat_dim:], axis=[1, 2]), axis=-1)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(base_h),
                                                             tf.math.l2_normalize(ref_h),
                                                             order_forward, order_reverse], name='repulsive_v6')



def autoencoder_v7_chain_classifier(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = config.chain_feat_dim
    K = config.K

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_classifier():
        input = layers.Input(shape=[chain_feat_dim, ])
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(K, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='classifier')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    classifier = get_classifier()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    z_chain_base = tf.squeeze(z_base[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])
    z_chain_ref = tf.squeeze(z_ref[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    chain_base = classifier(z_chain_base)
    chain_ref = classifier(z_chain_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse,
                                                             chain_base, chain_ref], name='AE_v7')


def autoencoder_v8_swap_consistency(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = config.chain_feat_dim
    K = config.K

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_classifier():
        input = layers.Input(shape=[chain_feat_dim, ])
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(K, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='classifier')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    classifier = get_classifier()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    z_chain_base = tf.squeeze(z_base[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])
    z_chain_ref = tf.squeeze(z_ref[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    chain_base = classifier(z_chain_base)
    chain_ref = classifier(z_chain_ref)

    # age swap
    z_age_swap = tf.concat([z_ref[:,:,:,:age_feat_dim], z_base[:,:,:,age_feat_dim:]], axis=-1)
    gen_swap = decoder(z_age_swap)
    z_swap = encoder(gen_swap)
    swap_feat = tf.squeeze(z_swap[:,:,:,:age_feat_dim], axis=[1,2])
    concat_swap = tf.concat([swap_feat, z_age_ref], axis=-1)
    order_swap = comparator(concat_swap)

    # cluster_swap

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, gen_swap, order_forward, order_reverse, order_swap,
                                                             tf.squeeze(z_ref[:,:,:,age_feat_dim+chain_feat_dim:]),
                                                             tf.squeeze(z_swap[:, :, :, age_feat_dim + chain_feat_dim:]),
                                                             chain_base, chain_ref], name='AE_v8')

def autoencoder_v8_vgg_swap(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = config.chain_feat_dim
    K = config.K

    def get_encoder():
        vgg16 = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                         weights='imagenet')
        vgg16.trainable = True
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_classifier():
        input = layers.Input(shape=[chain_feat_dim, ])
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(K, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='classifier')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    classifier = get_classifier()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    z_chain_base = tf.squeeze(z_base[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])
    z_chain_ref = tf.squeeze(z_ref[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    chain_base = classifier(z_chain_base)
    chain_ref = classifier(z_chain_ref)

    # age swap
    z_age_swap = tf.concat([z_ref[:,:,:,:age_feat_dim], z_base[:,:,:,age_feat_dim:]], axis=-1)
    gen_swap = decoder(z_age_swap)
    z_swap = encoder(gen_swap)
    swap_feat = tf.squeeze(z_swap[:,:,:,:age_feat_dim], axis=[1,2])
    concat_swap = tf.concat([swap_feat, z_age_ref], axis=-1)
    order_swap = comparator(concat_swap)

    # cluster_swap

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, gen_swap, order_forward, order_reverse, order_swap,
                                                             tf.squeeze(z_ref[:,:,:,age_feat_dim+chain_feat_dim:]),
                                                             tf.squeeze(z_swap[:, :, :, age_feat_dim + chain_feat_dim:]),
                                                             chain_base, chain_ref], name='AE_v8_vgg_swap')



def autoencoder_v9_vgg16(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    K = config.K

    def get_encoder():
        vgg16 = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                 weights='imagenet')
        vgg16.trainable = True
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    # age swap
    z_age_swap = tf.concat([z_ref[:,:,:,:age_feat_dim], z_base[:,:,:,age_feat_dim:]], axis=-1)
    gen_swap = decoder(z_age_swap)
    z_swap = encoder(gen_swap)
    swap_feat = tf.squeeze(z_swap[:,:,:,:age_feat_dim], axis=[1,2])
    concat_swap = tf.concat([swap_feat, z_age_ref], axis=-1)
    order_swap = comparator(concat_swap)

    # cluster_swap

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, gen_swap, order_forward, order_reverse, order_swap,
                                                             tf.squeeze(z_ref[:,:,:,age_feat_dim:]),
                                                             tf.squeeze(z_swap[:, :, :, age_feat_dim:])
                                                             ], name='AE_v9_vgg_swap')


def autoencoder_v6_vgg_sigmoid(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        vgg16 = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                         weights='imagenet')
        vgg16.trainable = True
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='AE_v6')


def get_vgg16(config):
    input = layers.Input(shape=[config.width, config.height, 3])
    initializer = tf.initializers.GlorotUniform()

    # Block 1
    x = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 2
    x = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 3
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 4
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 5
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    return keras.Model(inputs=input, outputs=x, name='vgg16_custom')


def get_vgg16_v2(config):
    input = layers.Input(shape=[config.width, config.height, 3])
    initializer = tf.initializers.GlorotUniform()

    # Block 1
    x = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 2
    x = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 3
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 4
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 5
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return keras.Model(inputs=input, outputs=x, name='vgg16_custom_v2')


def get_vgg16_v3_for_224(config):
    input = layers.Input(shape=[config.width, config.height, 3])
    initializer = tf.initializers.GlorotUniform()

    # Block 1
    x = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 2
    x = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 3
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 4
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 5
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    # Block 6
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=[2, 2])(x)

    return keras.Model(inputs=input, outputs=x, name='vgg16_v3')


def autoencoder_v6_custom_vgg_sigmoid(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        vgg16 = get_vgg16(config)

        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='AE_v6_custom_vgg')


def autoencoder_v6_custom_vgg_v2_sigmoid(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        vgg16 = get_vgg16_v2(config)

        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='AE_v6_custom_vgg_v2')


def v10_CC_VGG16(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = config.chain_feat_dim
    K = config.K

    def get_encoder():
        vgg16 = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                         weights='imagenet')
        vgg16.trainable = True
        # x = tf.reshape(vgg16.output, [-1, vgg16.output.shape[1]*vgg16.output.shape[2]*vgg16.output.shape[3]])
        # x = layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        # x = layers.LeakyReLU()(x)
        x = layers.Conv2D(2048, [2, 2], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_classifier():
        input = layers.Input(shape=[chain_feat_dim, ])
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(K, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='classifier')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    classifier = get_classifier()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    z_chain_base = tf.squeeze(z_base[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])
    z_chain_ref = tf.squeeze(z_ref[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    chain_base = classifier(z_chain_base)
    chain_ref = classifier(z_chain_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse,
                                                             chain_base, chain_ref], name='v10_CC_VGG')


def v6_VGG16_w224(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        vgg16 = get_vgg16_v3_for_224(config)
        x = layers.Conv2D(2048, [3, 3], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim, z_dim], [3, 3], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512, 512], [3, 3], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512, 512], [3, 3]),  # (bs, 8, 8, 512)
            upsample([256, 256], [3, 3]),  # (bs, 16, 16, 256)
            upsample([128, 128], [3, 3]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='v6_VGG16_w224')


def v6_VGG16_2CH_w224(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim
    chain_feat_dim = config.chain_feat_dim
    K = config.K

    def get_encoder():
        vgg16 = get_vgg16_v3_for_224(config)
        x = layers.Conv2D(2048, [3, 3], [1, 1])(vgg16.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=vgg16.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim, z_dim], [3, 3], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512, 512], [3, 3], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512, 512], [3, 3]),  # (bs, 8, 8, 512)
            upsample([256, 256], [3, 3]),  # (bs, 16, 16, 256)
            upsample([128, 128], [3, 3]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    def get_classifier():
        input = layers.Input(shape=[chain_feat_dim, ])
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(K, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='classifier')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()
    classifier = get_classifier()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    z_chain_base = tf.squeeze(z_base[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])
    z_chain_ref = tf.squeeze(z_ref[:,:,:,age_feat_dim:age_feat_dim+chain_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    chain_base = classifier(z_chain_base)
    chain_ref = classifier(z_chain_ref)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse, chain_base, chain_ref], name='v6_VGG16_w224_2CH')


def v6_ResNet50_w224(config):   #TODO: modify network and loading params problem
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        ResNet50 = keras.applications.ResNet50(input_shape=(config.width, config.height, 3), include_top=False,
                                                weights='imagenet')
        ResNet50.trainable = True
        x = layers.Conv2D(2048, [3, 3], [1, 1])(ResNet50.output)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(z_dim, [1, 1], [1, 1])(x)
        x = layers.LeakyReLU()(x)

        return keras.Model(inputs=ResNet50.input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim, z_dim], [3, 3], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512, 512], [3, 3], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512, 512], [3, 3]),  # (bs, 8, 8, 512)
            upsample([256, 256], [3, 3]),  # (bs, 16, 16, 256)
            upsample([128, 128], [3, 3]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='sigmoid')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref, order_forward, order_reverse], name='v6_VGG16_w224')

def spherical_rebuttal(config):
    z_dim = config.feat_dim
    age_feat_dim = config.age_feat_dim

    def get_encoder():
        input = layers.Input(shape=[config.width, config.height, 3])

        down_stack = [
            downsample([64, 64], [4, 4], apply_batchnorm=False),  # (bs, 32, 32, 64)
            downsample([128, 128], [3, 3]),  # (bs, 16, 16, 128)
            downsample([256, 256], [3, 3]),  # (bs, 8, 8, 256)
            downsample([512, 512], [3, 3]),  # (bs, 4, 4, 512)
            downsample([512, 512], [3, 3]),  # (bs, 2, 2, 512)
            downsample([z_dim], [2]),  # (bs, 1, 1, 1024)
        ]

        x = input
        for ds_block in down_stack:
            x = ds_block(x)

        return keras.Model(inputs=input, outputs=x, name='encoder')

    def get_decoder():
        input = layers.Input(shape=[1, 1, z_dim])
        OUTPUT_CHANNELS = 3

        up_stack = [
            upsample([z_dim], [4], apply_dropout=True),  # (bs, 2, 2, 512)
            upsample([512], [4], apply_dropout=True),  # (bs, 4, 4, 512)
            upsample([512], [4]),  # (bs, 8, 8, 512)
            upsample([256], [4]),  # (bs, 16, 16, 256)
            upsample([128], [4]),  # (bs, 32, 32, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='tanh')

        x = input
        for us_block in up_stack:
            x = us_block(x)

        x = last(x)

        return keras.Model(inputs=input, outputs=x, name='decoder')

    def get_comparator():
        input = layers.Input(shape=[age_feat_dim*2,])
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(age_feat_dim*2, kernel_regularizer=tf.keras.regularizers.l2(0.001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.n_comparator_output, use_bias=True)(x)
        x = layers.Softmax()(x)
        return keras.Model(inputs=input, outputs=x, name='comparator')

    # -------------------- #
    #   Assemble Network   #
    # -------------------- #

    base = layers.Input(shape=[config.width, config.height, 3])
    reference = layers.Input(shape=[config.width, config.height, 3])

    encoder = get_encoder()
    decoder = get_decoder()
    comparator = get_comparator()

    # encoder
    z_base = encoder(base)
    z_ref = encoder(reference)

    # decoder
    gen_base = decoder(z_base)
    gen_ref = decoder(z_ref)

    # comparator
    z_age_base = tf.squeeze(z_base[:,:,:,:age_feat_dim], axis=[1,2])
    z_age_ref = tf.squeeze(z_ref[:,:,:,:age_feat_dim], axis=[1,2])

    concat_forward = tf.concat([z_age_base, z_age_ref], axis=-1)
    concat_reverse = tf.concat([z_age_ref, z_age_base], axis=-1)

    order_forward = comparator(concat_forward)
    order_reverse = comparator(concat_reverse)

    return tf.keras.Model(inputs=[base, reference], outputs=[gen_base, gen_ref,
                                                             tf.math.l2_normalize(tf.squeeze(z_base[:, :, :, age_feat_dim:],axis=[1, 2]), axis=-1),
                                                             tf.math.l2_normalize(tf.squeeze(z_ref[:, :, :, age_feat_dim:], axis=[1, 2]), axis=-1),
                                                             order_forward, order_reverse], name='spherical_v0')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # specify which GPU(s) to be used

    class Config():
        def __init__(self):
            self.width = 224
            self.height = 224
            self.n_comparator_output = 3
            self.feat_dim = 1024
            self.chain_feat_dim = 256
            self.age_feat_dim = 256
            self.K = 2

    config = Config()
    # m = get_vgg16(config)
    #
    backbone = keras.applications.ResNet50(input_shape=(config.width, config.height, 3), include_top=False,
                                        weights='imagenet')
    backbone = keras.applications.VGG19(input_shape=(config.width, config.height, 3), include_top=False,
                                        weights='imagenet')


    model = v6_VGG16_w224(config)
    model.summary()


    backbone = keras.applications.VGG16(input_shape=(config.width, config.height, 3), include_top=False,
                                        weights='imagenet')

    autoencoder_v6_vgg_sigmoid(config)