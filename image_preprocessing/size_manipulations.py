def resize_image(image):
    # https://github.com/fchollet/keras/issues/5298
    import tensorflow as tf
    return tf.image.resize_images(image, (32, 32))