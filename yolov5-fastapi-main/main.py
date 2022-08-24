from fastapi import FastAPI, File, Path, Request
from segmentation import get_yolov5, get_image_from_bytes, get_image_from_path
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, datetime, timedelta
from updatedb import update_db

model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OKKKK')


@app.post("/path-to-json")
async def detect_path(_image_path: str):
    path_add_root = '../datasets/data_th/images/val/' + _image_path
    
    input_image = get_image_from_path(path_add_root)
    results = model(input_image)
    
    # detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    # detect_res = json.loads(detect_res)
    
    _out_folder = '../predict/'
    
    now = datetime.now()
    formatted_date = now.strftime('%y%m%d_%Hh%Mm%Ss')
    formatted_date_db = now.strftime('%Y-%m-%d %H:%M:%S')
    predict_json_file = ''.join(map(str,_image_path.split('.')[:-1])) + '-' + formatted_date + '.json'

    print(formatted_date)

    image_predict = results.pandas().xyxy[0]
    image_predict.to_json(_out_folder + predict_json_file, orient="records", indent=4)
    predict_json_str = image_predict.to_json(orient="records")
    
    # Path('../')
    
    update_db(_image_path, predict_json_str, '../predict/', results)
    
    
    
    detect_res = json.loads(predict_json_str)
    # return {"result": detect_res}
    return detect_res

    
    
    
# async def detect_food_return_json_result(file):
# async def detect_food_return_json_result(file: bytes = File(...)):
@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    # with open(file, encoding="utf8", errors='ignore') as f:

    #     my_json = file.decode('utf8').replace("'", '"')
    #     print(my_json)

    #     # Load the JSON to a Python list & dump it back out as formatted JSON
    #     data = json.loads(my_json)
    #     s = json.dumps(data, indent=4, sort_keys=True)
    #     print (s)
    #     return s
    print('fffffff')
    print(file[0:50])
    # print(type(file))
    # print(file)

    return {'asdf': 12}

    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")


@app.post("/getInformation")
async def getInformation(info : Request):
    req_info = await info.json()
    return {
        "status" : "SUCCESS",
        "data" : req_info
    }