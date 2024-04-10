import tensorflow as tf
from process_visualise import load_process_img

#content layer from which feature map will be pulled, set according to the research paper
content_layer = ['block4_conv2']      
                                        

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layer = len(content_layer)
num_style_layers = len(style_layers) 


def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
  # Firstly making the image channels  
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def style_loss(base, gram_target):

    height, width, channels = base.get_shape().as_list()
    gram_style = gram_matrix(base)
    
    return tf.reduce_mean(tf.square(gram_style - gram_target))    


def get_feature_representations(model, content_path, style_path):

    content_image = load_process_img(content_path)
    style_image = load_process_img(style_path)

    style_outputs = model(style_image)
    content_outputs = model(content_image)

    style_features = [style_layer for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_lyr for content_lyr in content_outputs[num_style_layers:]]

    return style_features, content_features


def compute_loss(model,base_img, gram_style_features, content_features, alpha, beta):
    
    model_outputs = model(base_img)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0.0
    content_score = 0.0
    
    #weighing equally the contribution of the loss of each layer 
    
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * style_loss(comb_style[0], target_style)

    weight_per_content_layer = 1.0 / float(num_content_layer)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score +=  weight_per_content_layer * content_loss(comb_content[0], target_content)
        
    #the ratio of alpha/beta (1e^-3 or 1e^-4) generally, this ratio dictates the output.
    loss = (alpha * content_score) + (beta * style_score)  

    return loss, style_score, content_score


def compute_gradients(dic):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dic)
    
    total_loss = all_loss[0]

    return tape.gradient(total_loss, dic['base_img']), all_loss