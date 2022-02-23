import cv2
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import StreamingResponse

import numpy as np
import io
app = FastAPI()

with open('mask_a.npy', 'rb') as f:
    mask_a = np.load(f)

with open('mask_b.npy', 'rb') as f:
    mask_b = np.load(f)

def main():
    img_1 = cv2.imread('manh.png')
    w,h,_ = img_1.shape
    thua_w = w%100+40
    thua_h = h%100+37
    img_1 = cv2.copyMakeBorder(img_1, 0, thua_w, 0, thua_h, cv2.BORDER_REFLECT
    )
    for  i in range (0,w,100):
        for j in range(0,h,100):
            image_design_tee = img_1[i:i+37,j:j+40].astype(int)*mask_a + mask_b
            image_design_tee = np.uint8(np.clip(image_design_tee, 0, 255))
            img_1[i:i+37,j:j+40]=image_design_tee
    img_1 = img_1[0:h,0:w]

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    nparr = np.fromstring(file, np.uint8)
    img_1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR )
    w,h,_ = img_1.shape
    if w <300 or h <300:
        raise HTTPException(status_code=500, detail="Iamge size >300")
    thua_w = w%100+40
    thua_h = h%100+37
    img_1 = cv2.copyMakeBorder(img_1, 0, thua_w, 0, thua_h, cv2.BORDER_REFLECT
    )
    for  i in range (0,w,100):
        for j in range(0,h,100):
            image_design_tee = img_1[i:i+37,j:j+40].astype(int)*mask_a + mask_b
            image_design_tee = np.uint8(np.clip(image_design_tee, 0, 255))
            img_1[i:i+37,j:j+40]=image_design_tee
    img_1 = img_1[0:h,0:w]
    _, im_buf_arr = cv2.imencode(".png", img_1)
    io_buf = io.BytesIO(im_buf_arr)
    return StreamingResponse(io_buf, media_type="image/png",
    headers={'Content-Disposition': 'inline; filename="manh.jpg"' })
