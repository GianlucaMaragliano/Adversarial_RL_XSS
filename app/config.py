from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings, case_sensitive=False):
    api_prefix: str = "vuln_backend/1.0"
    validator : str = "dummy_detector"

    model_config = SettingsConfigDict(env_file=".env")

description = ""
