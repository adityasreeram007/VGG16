import os
#import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import json
from flask import Flask, render_template, request, jsonify,abort
from PIL import Image

app=Flask(__name__)
UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def predict(img_path):
     labels={0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
#img_path = 'C:\\Users\\--\\Downloads\\dataset-original\\dataset-original\\metal\\'

     img = image.load_img(img_path, target_size=(300, 300))
     img = image.img_to_array(img, dtype=np.uint8)
     img=np.array(img)/255.0
#plt.imshow(img.squeeze())
     
     model = tf.keras.models.load_model("/home/aditya/Documents/TrashNet/trained_model.h5")
     p=model.predict(img[np.newaxis, ...])
     pro=np.max(p[0], axis=-1)
     print("p.shape:",p.shape)
     print("prob",pro)
     predicted_class = labels[np.argmax(p[0], axis=-1)]
     os.remove(img_path)
     print("classified label:",predicted_class)
     if predicted_class in ['Cardboard','Paper']:
          category = "Biodegradable"
          predicted_class = str(predicted_class)
          probability = str(pro)
          return category,predicted_class,probability
          #result += "Biodegradable" +'\n' + str(predicted_class) + '\n' + str(pro)
     elif predicted_class in ['Metal','Glass','Plastic']:
          category = "Non-Biodegradable"
          predicted_class = str(predicted_class)
          probability = str(pro)
          return category,predicted_class,probability
          #result += "Nonbiodegradable"+'\n'+ str(predicted_class) + '\n' + str(pro)
     else:
          category = "Categorizing Difficult"
          predicted_class = str(predicted_class)
          probability = str(pro)
          return category,predicted_class,probability
          #result = predicted_class + '\n' + str(pro)
     #return(result)
     return(str(predicted_class)+" \n Probability:"+str(pro))                   
# def predict(img_path):
#      labels={0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash'}
#      img = image.load_img(img_path, target_size=(224,224))
#      img = image.img_to_array(img, dtype=np.uint8)
#      img = np.array(img)/255.0
#      model = tf.keras.models.load_model("/home/aditya/Documents/TrashNet/trained_model.h5")
#      predicted = model.predict(img[np.newaxis, ...])
#      prob = np.max(predicted[0], axis=-1)
#      prob = prob*100
#      prob = round(prob,2)
#      prob = str(prob) + '%'
#      print("p.shape:",predicted.shape)
#      print("prob",prob)
#      predicted_class = labels[np.argmax(predicted[0], axis=-1)]
#      print("classified label:",predicted_class)
#      result=''

# #category,predicted_class,probability = predict(Image_path)
     
@app.route("/app", methods = ['POST']) #/file
def application():
     file = ""
     answer = None
     if request.method == "POST":
          file = request.files["file"]
          print(request.json)
          if file:
               f = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
               file.save(f)
               result = predict(file.filename)
               if result:
                    a = {"success":"True","result":result}
                    return json.dumps(a)
               else:
                    abort(400)
@app.route("/",methods=["GET"])
def apps():
     return '<form action="/app" method="POST" enctype="multipart/form-data"><input type="file" name="file" placeholder="file"/><input type="submit"/></form><br>'

if __name__ == "__main__":
    app.run(debug=True)
