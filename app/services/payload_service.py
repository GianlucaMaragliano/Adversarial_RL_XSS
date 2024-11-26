from app.exceptions.xss_exception import XSSException

async def generate_content_from_payload(params, request, templates, validator):
    sanitized_els = []
    for param in params:
        valid, sanitized = validator.validate(param)
        if not valid:
            raise XSSException()
        sanitized_els.append(sanitized)
    return templates.TemplateResponse("basic_response.html", {"request": request, "payloads": sanitized_els})
