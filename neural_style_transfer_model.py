
import tensorflow as tf
import numpy as np

def get_model(style_layers, content_layer):
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_output = [vgg.get_layer(name).output for name in content_layer]
    model_outputs = style_outputs + content_output
    return tf.keras.models.Model(vgg.input, model_outputs)

def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base, gram_target):
    height, width, channels = base.get_shape().as_list()
    gram_style = gram_matrix(base)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, base_img, gram_style_features, content_features, num_style_layers, num_content_layer):
    style_weight, content_weight = loss_weights
    model_outputs = model(base_img)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    style_score = 0
    content_score = 0
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], gram_target=target_style)
    weight_per_content_layer = 1.0 / float(num_content_layer)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * content_loss(comb_content[0], target_content)
    style_score *= style_weight
    content_score *= content_weight
    loss = content_score + style_score
    return loss, style_score, content_score

def compute_gradients(dic):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dic)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, dic['base_img']), all_loss
