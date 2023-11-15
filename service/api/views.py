from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from rec_sys.random_model import RandomModel
from service.api.exceptions import InvalidTokenError, ModelNotFoundError, UserNotFoundError
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


token_auth_scheme = HTTPBearer()
router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: str = Depends(token_auth_scheme),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if request.app.state.api_token != token.credentials:
        app_logger.info(f"InvalidTokenError: {token.credentials}")
        raise InvalidTokenError()

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name == "random_100":
        random_model = RandomModel()
        recommendation = random_model.predict(k_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    return RecoResponse(user_id=user_id, items=recommendation)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
