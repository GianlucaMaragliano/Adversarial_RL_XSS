from fastapi import APIRouter, status, HTTPException, Request
from app.request_models.basic_request import BasicRequest
from fastapi.responses import HTMLResponse
import logging
from app.services import payload_service
from app.exceptions.xss_exception import XSSException

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/",
    status_code = status.HTTP_200_OK
)
async def vuln_request(request: Request,params:BasicRequest):
    try:
        template = await payload_service.generate_content_from_payload(params.payload, request, request.app.package["templates"], request.app.package["validator"])

        return template
    except XSSException:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The payload contains an XSS attack")

@router.get(
    "/{payload}",
    status_code = status.HTTP_200_OK
)
async def vuln_request(request: Request,payload:str):
    try:
        template = await payload_service.generate_content_from_payload([payload], request, request.app.package["templates"], request.app.package["validator"])
        return template
    except XSSException:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The payload contains an XSS attack")