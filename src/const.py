import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / "3dmem" / ".env")

# about habitat scene
INVALID_SCENE_ID = []

# about chatgpt api
END_POINT = os.getenv("END_POINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
VLM_MODEL = os.getenv("VLM_MODEL")
CG_VLM_MODEL = os.getenv("CG_VLM_MODEL")
