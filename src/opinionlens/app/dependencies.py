from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from opinionlens.common.settings import get_settings

settings = get_settings()

api_header_scheme = APIKeyHeader(name="x-key")


async def authenticate_admin(
    key: Annotated[str, Depends(api_header_scheme)],
):
    if key != settings.api.admin_key:
        raise HTTPException(status_code=401, detail="Header 'x-key' contained an invalid key.")
