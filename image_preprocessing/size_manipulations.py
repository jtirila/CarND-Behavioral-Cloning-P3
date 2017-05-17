def resize_image_32_32(image):
    # https://github.com/fchollet/keras/issues/5298
    import tensorflow as tf
    return tf.image.resize_images(image, (32, 32))


def resize_image_128_128(image):
    # https://github.com/fchollet/keras/issues/5298
    import tensorflow as tf
    return tf.image.resize_images(image, (128, 128))


def resize_image_227_227(image):
    # https://github.com/fchollet/keras/issues/5298
    import tensorflow as tf
    return tf.image.resize_images(image, (227, 227))

