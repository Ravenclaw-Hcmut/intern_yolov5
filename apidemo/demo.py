from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel # import class BaseModel của thư viện pydantic


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
