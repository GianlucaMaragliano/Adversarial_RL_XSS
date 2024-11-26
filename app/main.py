from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Depends, HTTPException, status, Security
from app.config import Settings
import app.controllers.endpoint_controller as endpoint_controller
from app.config import description
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.payload_validators.utils.validator_selector import ValidatorSelector

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    BASE_DIR = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))
    validator_selector = ValidatorSelector()
    validator = settings.validator
    
    app.package = {
        "templates":templates,
        "validator":validator_selector.select(validator)
    }
    yield
    
app = FastAPI(
    title="StreetView - API backend",
    description=description,
    version=settings.api_prefix,
    lifespan=lifespan
)
api_prefix = settings.api_prefix


app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    endpoint_controller.router,
    prefix=f"/{api_prefix}/endpoint",
    tags=['endpoint'],
)

