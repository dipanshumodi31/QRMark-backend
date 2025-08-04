from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
from embedder import embed_qr_in_cr_hl_subband
from extractor import extract_qr_from_cr_hl_subband

app = FastAPI()

# Enable CORS to allow frontend (React) communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qr-mark.up.railway.app"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "QR Embed/Extract API is live ✅"}


# @app.get("/capture")
# def capture():
#     return capture_high_res_image()

@app.post("/embed")
async def embed_qr(
    file: UploadFile = File(...),
    qr_data: str = Form(...),
    alpha: int = Form(12)
):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        output_buffer = BytesIO()

        embed_qr_in_cr_hl_subband(
            host_image_path=image,
            qr_data=qr_data,
            output_image_path=output_buffer,
            alpha=alpha
        )

        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/extract")
async def extract_qr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        decoded_data = extract_qr_from_cr_hl_subband(img_np)

        return JSONResponse(content={"data": decoded_data if decoded_data else None})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
