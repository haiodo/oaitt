"""
OAITT — Open AI Transformer Transcriber.

Модуль авторизации по токену.
Предоставляет функции для проверки Bearer токена в заголовке Authorization.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import AUTH_TOKEN

logger = logging.getLogger(__name__)

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    Проверяет Bearer токен из заголовка Authorization.

    Если AUTH_TOKEN не установлен (пустая строка или None),
    авторизация отключена и запрос пропускается.

    Args:
        credentials: Учетные данные из заголовка Authorization.

    Returns:
        str: Токен, если авторизация прошла успешно.

    Raises:
        HTTPException(401): Если токен отсутствует или недействителен.

    Example:
        ```python
        from fastapi import APIRouter, Depends
        from src.auth import verify_token

        router = APIRouter()

        @router.get("/protected")
        async def protected_route(token: str = Depends(verify_token)):
            return {"message": "Access granted"}
        ```
    """
    # If AUTH_TOKEN is not set, skip authentication
    if not AUTH_TOKEN:
        logger.debug("Authentication disabled (AUTH_TOKEN not set)")
        return ""

    # Check if credentials are provided
    if credentials is None:
        logger.warning("Authentication failed: no credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify the token
    if credentials.credentials != AUTH_TOKEN:
        logger.warning("Authentication failed: invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug("Authentication successful")
    return credentials.credentials
