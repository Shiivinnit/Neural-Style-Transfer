import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path):
    
    img = Image.open(path)
    max_dim_image = max(img.size)
    max_dims = 512   #to make image compatible to a batch dimension

    scale_value = max_dims/max_dim_image

    img = img.resize((round(img.size[0]*scale_value), round(img.size[1]*scale_value)), Image.LANCZOS)
    
    img = tf.keras.utils.img_to_array(img)

    img = np.expand_dims(img, axis = 0)  #casting it in a batch shape (x,y,z)

    return img

def imshow(img, label = None):
    
    #squeezing the batch dimension of size 1
    out = np.squeeze(img, axis = 0)
    out = out.astype(np.uint8) #normalising the array entries to (0,255)
    plt.imshow(out)
    if label is not None:
        plt.title(label)
    
    plt.imshow(out)

def load_process_img(path):
    img = load_image(path)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def invert_process(processed_img):
    x = processed_img.copy()

    x = np.squeeze(x,0) #squeezing the batch_dimension
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]   #reversing the entries of the array

    x = np.clip(x,0,255).astype(np.uint8)
    return x


def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(10, 5))
  content = load_image(content_path)
  style = load_image(style_path)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')

  if show_large_final:
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()