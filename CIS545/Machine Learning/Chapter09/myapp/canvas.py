from flask import Flask, render_template, request
import base64
from scipy import misc
# import imageio
import io
import sys

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('canvas.html')
  
@app.route('/canvas', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      data_url = request.headers['PHOTO']
      content = data_url.split(';')[1]
      image_encoded = content.split(',')[1]
      # body = base64.decodebytes(image_encoded.encode('utf-8'))
      body = base64.b64decode(image_encoded)
      # image4d = imageio.imread(io.BytesIO(body))
      image4d = misc.imread(io.BytesIO(body))
      image = image4d[:,:,3]
      image = -(image - 255)
      misc.imsave('./static/imageToSave.png', image)

      return "Success"
   if request.method == 'GET':
      return "Response"
    
if __name__ == '__main__':
   app.run(debug = True)