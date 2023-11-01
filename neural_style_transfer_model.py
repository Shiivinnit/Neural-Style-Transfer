""" Neural Algorithm  
This is the implementation of "A Neural Algorithm of Artistic Style" research paper proposed by Gatys et. al (2017).

You can find the paper at https://doi.org/10.48550/arXiv.1508.06576

This paper acts as a cornerstone in the usage of Artificial Intelligence in art, leveraging power of deep neural networks to create artistic images of high perceptual quality.

"""

import tensorflow as tf
import numpy as np
import IPython.display

#calling the pre-trained vgg19 model as per the research paper
def get_model(style_layers, content_layer): 
    
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False #Pre trained
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_output = [vgg.get_layer(name).output for name in content_layer]
    model_outputs = style_outputs + content_output
    return tf.keras.models.Model(vgg.input, model_outputs)

def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))
    
#creating the gram matrix
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

def compute_loss(model, ratio = 1e-4, loss_weights, base_img, gram_style_features, content_features, num_style_layers, num_content_layer):
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
    loss = (ratio*content_score) + style_score #alpha/beta (1e-3 or 1e-4)
    return loss, style_score, content_score

def compute_gradients(dic):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dic)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, dic['base_img']), all_loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations = 1800,
                       content_weight = .5e1,
                       style_weight = 1e-2):
    
    model = get_model()
                           
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    base_img = load_process_img(content_path)
    base_img = tf.Variable(base_img, dtype = tf.float32)

    opt = tf.optimizers.Adam(learning_rate= 1e-1, epsilon = 1e-4)

    iter_count = 1

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    dic = {
        'model' : model,
        'loss_weights' : loss_weights,
        'base_img' : base_img,
        'gram_style_features': gram_style_features,
        'content_features' : content_features
    }

    num_rows = 2
    num_cols = 5
    display_inteval = num_iterations / (num_rows * num_cols)


    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = - norm_means
    max_vals = 255 - norm_means

    imgs = []

    for i in range(num_iterations):
        grads, all_loss = compute_gradients(dic)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, base_img)])
        clipped = tf.clip_by_value(base_img, min_vals, max_vals)
        base_img.assign(clipped)


        if loss < best_loss:
            best_loss = loss
            best_img = invert_process(base_img.numpy())

        if i % display_inteval == 0:

            plot_img = base_img.numpy()
            plot_img = invert_process(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait = True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total Loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {: .4e}, '
                  .format(loss, style_score, content_score))

    IPython.display.clear_output(wait = True)
    plt.figure(figsize= (14,4))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss                
                  
