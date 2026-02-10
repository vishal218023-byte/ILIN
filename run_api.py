import logging
import warnings

# Suppress torch warnings and inspection errors
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
logging.getLogger('torch').setLevel(logging.ERROR)

class TorchInspectionFilter(logging.Filter):
    def filter(self, record):
        return "torch.classes" not in record.getMessage()

logging.getLogger().addFilter(TorchInspectionFilter())

from fastapi import FastAPI
from app.api.endpoints import app
import uvicorn


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
