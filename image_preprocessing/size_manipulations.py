def my_resize_image(image, size):
    # https://github.com/fchollet/keras/issues/5298
    import tensorflow as tf
    return tf.image.resize_images(image, size)


def resize_image_32_32(image):
    return my_resize_image(image, (32, 32))


def resize_image_128_128(image):
    return my_resize_image(image, (128, 128))


