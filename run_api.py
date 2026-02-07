from fastapi import FastAPI
from app.api.endpoints import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
