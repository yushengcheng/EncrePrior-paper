import gc

import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

model = VGG16()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)


def vgg16_emb(image):
    # prepare data
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


def c_distance(widget_list):
    e_distance_list = []
    c_distance_list = []
    for group in widget_list:
        img_np1 = group[0]
        img_np2 = group[1]
        # get image feature
        feature_1 = vgg16_emb(img_np1)
        feature_2 = vgg16_emb(img_np2)
        e_distance = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        e_distance_list.append(e_distance)
        c_distance_list.append(cosine_similarity(feature_1,feature_2)[0][0])
    return e_distance_list,c_distance_list
