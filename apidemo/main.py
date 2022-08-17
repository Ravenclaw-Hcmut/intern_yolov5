from fastapi import FastAPI #import class FastAPI() từ thư viện fastapi

app = FastAPI() # gọi constructor và gán vào biến app


# @app.get("/") # giống flask, khai báo phương thức get và url
# async def root(): # do dùng ASGI nên ở đây thêm async, nếu bên thứ 3 không hỗ trợ thì bỏ async đi
#     return {"message": "Hello World"}



@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(user_id: int, item_id: str):
    item = {"item_id": item_id, "owner_id": user_id}
    return item