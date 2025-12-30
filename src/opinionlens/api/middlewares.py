from fastapi import Request
from starlette.concurrency import iterate_in_threadpool

from opinionlens.common.utils import get_logger

logger = get_logger(__name__, filename="logs/api.log")


async def log_error_responses(request: Request, call_next):
    url = request.url.path
    response = await call_next(request)
    if response.status_code >= 400:
        if response.status_code == 404:
            return response

        response_body = [chunck async for chunck in response.body_iterator]
        response_text = (b"".join(response_body)).decode()
        logger.error(
            f"Bad response to {url} with status code {response.status_code}: {response_text}"
        )
        response.body_iterator = iterate_in_threadpool(iter(response_body))
    return response
