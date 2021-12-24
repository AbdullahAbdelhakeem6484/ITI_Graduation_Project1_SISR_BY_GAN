import streamlit as st
from PIL import Image


import cv2
import numpy as np
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras

import glob
import os

import random
from numpy import asarray
from itertools import repeat

import imageio
from imageio import imread
from PIL import Image
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model

#print("Tensorflow version " + tf.__version__)
#print("Keras version " + tf.keras.__version__)

def sample_images_test(img):

    
    lr_images = []
    
    img1 = imread(img, as_gray=False, pilmode='RGB')
    img1 = img1.astype(np.float32)
    
    img1_low_resolution = imresize(img1, (64, 64, 3))
          

    # do a random horizontal flip
    if np.random.random() < 0.5:
        img1_low_resolution = np.fliplr(img1_low_resolution)
    
    lr_images.append(img1_low_resolution)
        
   
    # convert lists into numpy ndarrays
    return  np.array(lr_images)    



def save_images_test(original_image , sr_image, path):
    
    """
    Save LR, HR (original) and generated SR
    images in one panel 
    """
    
    fig, ax = plt.subplots(1,2, figsize=(10, 6))

    images = [original_image, sr_image]
    titles = ['OR','SR']

    for idx,img in enumerate(images):
        # (X + 1)/2 to scale back from [-1,1] to [0,1]
        ax[idx].imshow((img + 1)/2.0, cmap='gray')
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
        
    plt.savefig(path)
    


# @st.cache
def fetch_model():
	#import gdown
	# https://drive.google.com/file/d/116fpSp3dUBtH7GkCoZ4UymK76xRTrZLs/view?usp=sharing
	#url = 'https://drive.google.com/uc?id=116fpSp3dUBtH7GkCoZ4UymK76xRTrZLs'
	#output = 'generator_model.h5'
	#gdown.download(url, output, quiet=False)
	model =load_model("SRGAN/models/generator_5000.h5")
    #return load_model('generator_7500.h5', compile=False)
	return model

loaded_model = fetch_model()

def pred_img(img):
    model = fetch_model()
    lr_img = sample_images_test(img)
    # normalize the images
    lr_img = (lr_img / 127.5) - 1
    generated_img = model.predict_on_batch(lr_img)
    lr_images_saved_m = lr_img.reshape((64,64,3))
    generated_images_saved_m= generated_img.reshape((256,256,3))
    #save_images_test(lr_images_saved_m ,generated_images_saved_m ,"D:/02-AI Program/ALL SUBJECT/07-Deployment/02-SRGAN Deploy/predict_app/Saved/newImage")

    #image =(generated_images_saved_m + 1)/2.0
	#image = ((generated_images_saved_m + 1)/2.0)
    st.image(((generated_images_saved_m + 1)/2.0),clamp=True)


st.title('Single Image Super Resolution')
st.write('By Using Approach Called GAN : Generator and Descriminator')
st.markdown('----')

#col1, col2 = st.columns([2,1])
col1, col2 = st.columns([2,1])

with col1:
    raw_image = st.file_uploader('Upload an Image')

    if raw_image:
        st.image(raw_image)

with col2:
    #st.write('Make a Prediction')
    if st.button('Run Model'):
        pred_img(raw_image)
		#st.image(predicted_image)
        #st.write(f'Prediction:  {pred_img(raw_image)}')

		