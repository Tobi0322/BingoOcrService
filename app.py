import io
import pytesseract
import os
import PIL
import re

import numpy as np
import cv2 as cv

from flask import Flask, request
from flask_restplus import Api, Resource, fields

app = Flask(__name__)
api = Api(app)
ocr_namespace = api.namespace('ocr', description='Ocr operations')

ocr_task = api.model('ocr_task', {
    'image': fields.Raw,
})

@ocr_namespace.route('/')
class OcrTaskList(Resource):   
        
    @ocr_namespace.doc('create_ocrTask')
    @ocr_namespace.marshal_list_with(ocr_task)
    @ocr_namespace.doc(params={'image': 'The image to run ocr on'})
    def post(self):
        # Load image as binary image
        photo = request.files['image']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv.imdecode(data, cv.IMREAD_GRAYSCALE)

        # Apply OTSU threshhold
        reth, th = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        path = os.path.join("images", "1.png")
        cv.imwrite(path, th)

        # Run ocr on image
        text = pytesseract.image_to_string(PIL.Image.open(path), config='--psm 11')
        num_text = re.sub("[^0-9]", " ", text)
        num_text = '#'.join(num_text.split())
        return num_text

if __name__ == "__main__":
    app.run()
