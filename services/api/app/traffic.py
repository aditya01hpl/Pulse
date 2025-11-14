from __future__ import annotations

import random
from typing import Dict, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .repository import get_traffic_weights


async def get_weights(session: AsyncSession) -> Dict[str, float]:
    db_weights = await get_traffic_weights(session)
    if db_weights:
        return db_weights
    return settings.traffic_weights


def choose_model(weights: Dict[str, float]) -> str:
    items: List[Tuple[str, float]] = list(weights.items())
    models, probs = zip(*items)
    r = random.random()
    cum = 0.0
    for m, p in zip(models, probs):
        cum += p
        if r <= cum:
            return m
    return models[-1]
