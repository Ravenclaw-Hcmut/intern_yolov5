# # save this as app.py
# from flask import Flask, request
# from markupsafe import escape

# app = Flask(__name__)

# @app.route('/')
# def hello():
#     name = request.args.get("name", "World")
#     return f'Hello, {escape(name)}!'

"""
Convert image to 3d array
"""

#IMPORT LIBRARIES
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import numpy as np
from towhee import pipeline
from PIL import Image
from  towhee.utils.pil_utils import from_pil
from towhee.utils.ndarray_utils import to_ndarray
import json
#FILE TO PILLOW IMAGE
def read_imagefile(file) -> Image.Image:
  image = Image.open(BytesIO(file))
  return image
#PILLOW TO JSON FRIENDLY OUTPUT USING JSON AND TOWHEE
def embedding_gen(image: Image.Image) -> np.ndarray:
  image_obj=from_pil(image)
  image_emb=to_ndarray(image_obj)
  image_emb=json.dumps(image_emb.tolist())
  return image_emb
#APP
app = FastAPI()
#this code is for image based towhee api only
@app.post("/image")
#FILE INPUT AND FILE CHECKING
async def predict_api(file: UploadFile = File(...)):
  extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
  if not extension:
    return "Image must be jpg or jpeg or png format!"
#FILE PROCESSING
  image = read_imagefile(await file.read())
  emd=embedding_gen(image)
  return {"Image_Embeddings":emd}
#SERVER
if __name__ == "__main__":
  uvicorn.run(app, debug=True)