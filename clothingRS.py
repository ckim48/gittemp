import numpy as np
import pandas as pd
import os 
import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.applications import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pathlib
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

path = '/Users/joonsunglee/Documents/LognCoding/Lecture/Metaverse DS (FRI, SUN, 2hrs)/fashion-dataset/fashion-dataset/'
dataset_path = pathlib.Path(path)
images=os.listdir(dataset_path) # ['.DS_Store', 'images.csv', 'images', 'styles', 'styles.csv']

df = pd.read_csv(path + "styles.csv", nrows=6000, error_bad_lines=False)
df['image'] = df.apply(lambda x: str(x['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)

img_width, img_height, chnl = 200, 200, 3

densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(img_width, img_height, chnl))
densenet.trainable = False

model = keras.Sequential([
    densenet,
    GlobalMaxPooling2D()
])

def img_path(img):
    return path + 'images/' + img

def model_predict(model, img_name):
    img = tf.keras.utils.load_img(img_path(img_name), target_size=(img_width, img_height))
    x   = tf.keras.utils.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)

df_copy = df
df_embedding = df_copy['image'].apply(lambda x: model_predict(model, x))
df_embedding = df_embedding.apply(pd.Series)

cosine_sim = linear_kernel(df_embedding, df_embedding)

indices = pd.Series(range(len(df)), index=df.index)

def get_recommendations(index, df, cosine_sim=cosine_sim):
    idx = indices[index]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    cloth_indices = [i[0] for i in sim_scores]
    return df['image'].iloc[cloth_indices]

def show_recommendation(chosen_img_indx):
    recommendation = get_recommendations(chosen_img_indx, df, cosine_sim)
    recommendation_list = recommendation.to_list()
    
    return recommendation_list

show_recommendation(22)