import os
import warnings 
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import IPython.display

import tensorflow as tf


from custom_losses import content_loss, gram_matrix, style_loss, get_feature_representations, compute_loss, compute_gradients
from nst import get_model, style_transfer
from process_visualise import load_image, imshow, load_process_img, invert_process, show_results

content_path = '/users/shivinnit/downloads/mona_lisa.jpeg' #provide with your content_path
style_path = '/users/shivinnit/downloads/starry_night.jpeg' #provide with your style_path 

output_dir = '/users/shivinnit/downloads/nst_output.jpeg' #provide with your output directory


def main():
    
    print('Neural style transfer\n\n')
    #calls out the style_transfer function
    best, best_loss = style_transfer(content_path= content_path, style_path= style_path)
    #visualise the result
    show_results(best, content_path, style_path)
    #saving the output to the specified directory (output_dir)

    final = Image.fromarray(best)
    final.save(output_dir)


if __name__ == '__main__':
    main()


    
