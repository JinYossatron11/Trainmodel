from os import abort

from flask import Flask, jsonify, request, render_template
from keras import applications as ka, layers as kl, models as km
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import pymongo
import  json
import tensorflow as tf
from bson.objectid import ObjectId
from keras_applications.resnet50 import preprocess_input 

def getmodel(cls=2):
    base_model = ka.nasnet.NASNetMobile(weights='imagenet', pooling='avg')
    # inceptionresnetv2, mobilenets, nasnet,resnet50
    print(base_model.summary())
    x = kl.Dense(cls, activation='softmax')(base_model.get_layer('global_average_pooling2d_1').output)
    model = km.Model(base_model.input, x)
    return model


model =getmodel()
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['acc'],
              )
model.load_weights('fishv2.h5')
graph = tf.get_default_graph()


app = Flask(__name__)
def predict(path):
    global graph
    with graph.as_default():
        img = Image.open(path).resize((224, 224))
        img = np.array(img).reshape((1, 224, 224, 3))
        img = preprocess_input(img)
#         img = img.astype('float32')
#         img /= 127.5
#         img -= 1
        return model.predict(img)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["fishAof"]
mycol = mydb["fishInfo"]
output = []

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/test', methods=['POST'])
def test():
    file = request.files['image']
    path = 'test/test.png'
    file.save(path)
    prd = predict(path)
    print(prd[0][0], prd[0][1])
    if prd[0][0] > prd[0][1]:
        return getDataByName("ChromisChrysura")
    else:
        return  getDataByName("AmphiprionClarkii")
@app.route('/fish', methods=['POST'])
def fist():
    if not request.json or not 'fishName' or not 'note' in request.json:
         abort(400)
    fishInfo = {
        "breed": request.json["breed"],
        "fishName": request.json["fishName"],
        "note": request.json["note"],
    }
    mycol.insert_one(fishInfo)
    return  jsonify({"_id":str(fishInfo["_id"]), "breed": fishInfo["breed"],"fishName" : fishInfo["fishName"], "note":fishInfo["note"]}), 200

@app.route('/fishby/<_id>', methods=['GET'])
def getFishById(_id):

    find = mycol.find_one({"_id":ObjectId(_id)})
    return jsonify({"breed": find["breed"],"fishName":find["fishName"], "note":find["note"]}), 200

def getDataByName(name):
    find = mycol.find_one({"breed": name})
    return jsonify({"breed": find["breed"],"fishName":find["fishName"], "note":find["note"]}), 200
if __name__ == '__main__':
    app.run(debug=True)
