def enhance_contrast(img):
    import tensorflow as tf
    return tf.image.adjust_contrast(img, 0.5)
