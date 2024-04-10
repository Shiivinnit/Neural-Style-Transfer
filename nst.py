import IPython.display 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from custom_losses import content_loss, gram_matrix, style_loss, get_feature_representations, compute_loss, compute_gradients, style_layers, content_layer, num_style_layers, num_content_layer

from process_visualise import load_process_img, invert_process


def get_model():

    vgg = tf.keras.models.load_model('vgg19_saved_model') #the saved model doesn't have fully connected layers (include_top = False)
    vgg.trainable = False #Pre-trained
    
    #debugging
    first_layer = vgg.get_layer(index=0)
    print('first_layer: {first_layer}')

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_output = [vgg.get_layer(name).output for name in content_layer]

    model_outputs = style_outputs + content_output

    return tf.keras.models.Model(vgg.input, model_outputs)


def style_transfer(content_path,
                       style_path,
                       num_iterations = 1200):
    
    #creating an input call to choose the degree of style transfer 
    print('Please choose a degree from {1,2,3}\n\n')
    input_int = int(input("Choose the degree of style transfer: "))
    
    if input_int == 1:
        alpha = 1e-3
        beta = 1
    
    elif input_int == 2:
        alpha = 1e-4
        beta = 1
    
    elif input_int == 3:
        alpha = 1e-4
        beta = 10
    
    else:
        raise ValueError("Choose a degree from {1,2,3} only!")
        
    
    model = get_model()
    for layer in model.layers:
        layer.trainable = False # we don't want to train layers, hence set to false

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    base_img = load_process_img(content_path)
    base_img = tf.Variable(base_img, dtype = tf.float32)

    opt = tf.optimizers.Adam(learning_rate= 1, epsilon = 1e-6)

    iter_count = 1

    best_loss, best_img = float('inf'), None  #setting the best_loss equal to positive infinity.

    dic = {
        'model' : model,
        'base_img' : base_img,
        'gram_style_features': gram_style_features,
        'content_features' : content_features,
        'alpha' : alpha,
        'beta' : beta
     }

    num_rows = 2
    num_cols = 3
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