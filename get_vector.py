import json
import cv2
from PIL import Image
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

import flask
from flask_jsontools import jsonapi
from flask_api import status
from flask import Flask, jsonify, make_response

app = Flask(__name__)



modelDir = 'models'
dlibModelDir = '/root/openface/models/dlib'
openfaceModelDir = '/root/openface/models/openface/'

dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(dlibFacePredictor)

networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96
net = openface.TorchNeuralNet(networkModel, imgDim)


def getRep(image, imgDim=96):

    bgrImg = image

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    alignedFace = align.align(imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    rep = net.forward(alignedFace)

    return rep


@app.route('/', methods=['GET'])
@jsonapi
def home():
    return status.HTTP_200_OK


@app.route('/get_vector', methods=['POST'])
@jsonapi
def get_vector():
    image = Image.open(flask.request.files['image']).convert('RGB')
    image = np.array(image)

    try:
        face = getRep(image, imgDim)
    except:
        return jsonify({'error': 'error'})

    return {'vector': json.dumps(face.tolist())}



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
