from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="file" type="file" >
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    f = open('./server_audio/data.wav', 'w')
    await f.write(file.file)
    f.close()
    return {"filename": file.filename}
