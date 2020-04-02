import os, sys, json, datetime
import pandas as pd
import cv2
from PIL import Image

import numpy as np
np.set_printoptions(precision=2)
from scipy.spatial import distance

import openface

import flask
from flask import Flask, jsonify, make_response, request
from flask_jsontools import jsonapi
from flask_api import status


STATIC_PATH = '/root/openface/wrapper/app/static'
IMAGES_PATH = os.path.join(STATIC_PATH, 'resources/photos')
IMAGES_TMP_PATH = os.path.join(STATIC_PATH, 'tmp')
PATH_VECTORS = os.path.join(STATIC_PATH, 'resources/vectors.json')

TEMPLATES_PATH = 'templates'
MODEL_PATH = 'models'
DLIB_MODEL_PATH = '/root/openface/models/dlib'
OPENFACE_MODEL_DIR = '/root/openface/models/openface/'
DLIBFACE_PREDICTOR_PATH = os.path.join(DLIB_MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
ALIGN = openface.AlignDlib(DLIBFACE_PREDICTOR_PATH)
NN_MODEL_PATH = os.path.join(OPENFACE_MODEL_DIR, 'nn4.small2.v1.t7')
IMG_DIM = 96
NN_MODEL = openface.TorchNeuralNet(NN_MODEL_PATH, IMG_DIM)


app = Flask(__name__, template_folder=TEMPLATES_PATH, static_url_path=STATIC_PATH)


def get_vector(image, align=ALIGN, model=NN_MODEL, img_dim=IMG_DIM):
    rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned = align.getLargestFaceBoundingBox(rgbImg)

    alignedFace = align.align(img_dim, rgbImg, aligned, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    rep = model.forward(alignedFace)

    return rep


def save_vectors(path_photos=IMAGES_PATH, path_vectors=PATH_VECTORS):
    vectors = {}
    photos = os.listdir(path_photos)

    for photo in photos:
        if photo!='.keep':
            path = os.path.join(path_photos, photo)
            image = np.asarray(Image.open(path))
            vector = get_vector(image)
            vectors[photo] = list(vector)

    with open(path_vectors, 'w') as file:
        json.dump(vectors, file)


def get_doppelganger(target, path_vectors=PATH_VECTORS, k=6):
    with open(path_vectors) as f:
        d = f.read()
    vectors = pd.DataFrame(json.loads(d))
    matrix = np.transpose(vectors.values)
    distances = distance.cdist(target, matrix, "cosine")[0]

    idx = np.argpartition(distances, k)
    first_k_distances = distances[idx[:k]]
    first_k_idx = idx[:k]
    order = np.argsort(first_k_distances)
    first_k_idx = first_k_idx[order]
    first_k_distances = first_k_distances[order]
    first_k_photos = [str(vectors.columns.tolist()[i]) for i in first_k_idx]
    first_k_names = [x.split('.')[0] for x in first_k_photos]

    return zip(first_k_names, first_k_photos)


@app.route('/', methods=['GET'])
@jsonapi
def home():
    return status.HTTP_200_OK


@app.route('/doppelganger', methods=['GET', 'POST'])
def doppelganger():
    if request.method == 'GET':
        return flask.render_template('upload.html')

    else:
        try:
            image = Image.open(flask.request.files['image'])
            now = str(datetime.datetime.today())
            image_path = os.path.join(IMAGES_TMP_PATH, '{}.png'.format(now))
            image.save(image_path)

            image = image.convert('RGB')
            image = np.asarray(image)
            new_vector = get_vector(image, img_dim=IMG_DIM)
            new_vector = new_vector.reshape(1, len(new_vector))
            names = get_doppelganger(new_vector)

            return flask.render_template('doppelganger.html', names=names, image_path=image_path)

        except:
            return jsonify({'Error': 'Something bad happened'})



if __name__ == '__main__':
    what = sys.argv[1]

    if what=='run':
        app.run(host='0.0.0.0', debug=True)
    elif what=='get-vectors':
        save_vectors()
    else:
        raise ValueError('`python app.py run` or `python app.py get-vectors` ;)')
