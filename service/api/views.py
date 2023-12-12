from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import InvalidTokenError, ModelNotFoundError, UserNotFoundError
from service.log import app_logger

from .init_models import extend_to_k_recs, faiss, lightfm, model_popular, random_model, ae_recommender
import numpy as np


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
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {"example": {"user_id": 1, "items": [55, 11, 72, 21, 88, 64, 45, 59, 14, 60]}}
            },
        },
        404: {
            "description": "Model / user not found",
            "content": {
                "application/json": {
                    "example": {
                        "errors": [{"error_key": "model_not_found", "error_message": "Model random_101 not found"}]
                    },
                }
            },
        },
        401: {
            "description": "Unauthorized (wrong token)",
            "content": {
                "application/json": {
                    "example": {"errors": [{"error_key": "unauthorized", "error_message": "Unauthorized"}]}
                }
            },
        },
    },
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
        reco = random_model.predict(k_recs=k_recs)
    elif model_name == "popular":
        reco = model_popular.predict([[user_id]])
    elif model_name == "faiss":
        _, reco = faiss.search(user_id)
    elif model_name == "lightfm_online":
        reco = lightfm.predict(user_id)
    elif model_name == "ae_recommender":
        reco = ae_recommender.recommend(user_id)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    if len(reco) < k_recs:
        reco = extend_to_k_recs(reco, user_id, k_recs)

    reco = reco[: min(len(reco), k_recs)]
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
