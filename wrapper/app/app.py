import os, sys, json
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


modelDir = 'models'
dlibModelDir = '/root/openface/models/dlib'
openfaceModelDir = '/root/openface/models/openface/'

dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(dlibFacePredictor)

networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96
net = openface.TorchNeuralNet(networkModel, imgDim)

PATH_IMAGES = '/photos'
PATH_VECTORS = os.path.join('/vectors', 'vectors.json')


app = Flask(__name__, template_folder='templates')


def get_vector(image, imgDim=96):
    rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned = align.getLargestFaceBoundingBox(rgbImg)
    alignedFace = align.align(imgDim, rgbImg, aligned, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    rep = net.forward(alignedFace)

    return rep


def save_vectors(path_photos=PATH_IMAGES, path_vectors=PATH_VECTORS):
    vectors = {}
    photos = os.listdir(path_photos)

    for photo in photos:
        if photo!='.keep':
            path = os.path.join(path_photos, photo)
            name = photo.split('.')[0]
            image = np.asarray(Image.open(path))
            vector = get_vector(image)
            vectors[name] = list(vector)

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
    first_k_names = [vectors.columns.tolist()[i] for i in first_k_idx]

    return dict(zip(first_k_names, first_k_distances))


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
            image = Image.open(flask.request.files['image']).convert('RGB')
            image = np.asarray(image)
            new_vector = get_vector(image, imgDim)
            doppelgangers = get_doppelganger(new_vector)
            return flask.render_template('doppelganger.html', names=doppelgangers)

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
