from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RecSysDataConfig:
    num_users: int = 2000
    num_items: int = 500
    interactions_per_user: int = 20
    user_feat_dim: int = 8
    item_feat_dim: int = 8
    seed: int = 42


def generate_synthetic_recsys(config: RecSysDataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)

    user_features = rng.normal(0, 1, size=(config.num_users, config.user_feat_dim))
    item_features = rng.normal(0, 1, size=(config.num_items, config.item_feat_dim))

    # Latent preference vectors
    user_pref = rng.normal(0, 1, size=(config.num_users, config.item_feat_dim))

    rows = []
    for user_id in range(config.num_users):
        # sample items for this user
        items = rng.choice(config.num_items, size=config.interactions_per_user, replace=False)
        u_feat = user_features[user_id]
        u_pref = user_pref[user_id]
        for item_id in items:
            i_feat = item_features[item_id]
            # score is dot between user preference and item features + interaction terms
            score = float(np.dot(u_pref, i_feat) + 0.2 * np.dot(u_feat, i_feat[: config.user_feat_dim]))
            # logistic with noise
            prob = 1 / (1 + np.exp(-score))
            prob = np.clip(prob, 1e-4, 1 - 1e-4)
            click = rng.binomial(1, prob)
            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                **{f"u_{i}": u_feat[i] for i in range(config.user_feat_dim)},
                **{f"v_{j}": i_feat[j] for j in range(config.item_feat_dim)},
                "score": score,
                "clicked": int(click),
            })

    df = pd.DataFrame(rows)
    return df


def split_features_labels(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith("u_") or c.startswith("v_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["clicked"].to_numpy(dtype=np.int64)
    return X, y, feature_cols
