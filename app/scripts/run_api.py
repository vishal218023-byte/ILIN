import sys
import logging
import warnings
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
