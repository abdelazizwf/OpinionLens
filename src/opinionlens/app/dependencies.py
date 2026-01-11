import os
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

ADMIN_KEY = os.environ.get("ADMIN_API_KEY")

api_header_scheme = APIKeyHeader(name="x-key")


async def authenticate_admin(
    key: Annotated[str, Depends(api_header_scheme)],
):
    if ADMIN_KEY is not None and key == ADMIN_KEY:
        return
    raise HTTPException(status_code=401, detail="Header 'x-key' contained an invalid key.")
