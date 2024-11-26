from pydantic import BaseModel
from typing import List

class BasicRequest(BaseModel):
    payload: List[str]  