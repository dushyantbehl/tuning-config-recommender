import asyncio
import os
import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import BackgroundTasks, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from tuning_config_recommender.adapters import FMSAdapter

app = FastAPI(title="Recommender API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def delete_files(file_paths: list[str]) -> None:
    await asyncio.sleep(600)
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")


class RecommendationsRequest(BaseModel):
    tuning_config: dict | None = None
    tuning_data_config: dict | None = None
    compute_config: dict | None = None
    accelerate_config: dict | None = None
    skip_estimator: bool | None = False


def generate_unique_stamps():
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    random_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_id}"


@app.post("/recommend")
async def recommend(
    background_tasks: BackgroundTasks,
    req: RecommendationsRequest,
):
    err_msg = (
        "Generation failed, please provide correct inputs or report it to the team!"
    )
    try:
        paths_to_delete = []
        base_dir = Path(__file__).parent
        output_dir = base_dir / "outputs" / generate_unique_stamps()

        fms_adapter = FMSAdapter(base_dir=output_dir, additional_actions=[])

        response = fms_adapter.execute(
            tuning_config=req.tuning_config,
            compute_config=req.compute_config,
            accelerate_config=req.accelerate_config,
            data_config=req.tuning_data_config,
            unique_tag="",
            paths={},
            skip_estimator=req.skip_estimator,
        )
        response.pop("patches")
        for _, path in response["paths"].items():
            paths_to_delete.append(path)

        background_tasks.add_task(delete_files, paths_to_delete)
        return response
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({"message": err_msg}),
        )
