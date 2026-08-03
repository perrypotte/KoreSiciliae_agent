from dataclasses import dataclass
import os
from typing import List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class RouteMatrixEntry:
    source_id: str
    target_id: str
    mode: str
    distance_km: float
    time_seconds: int


class RouteMatrixRepository:

    #def __init__(self, engine: Engine):
    def __init__(self):
        #self.engine = engine
        self.engine=create_engine(os.getenv("DATABASE2_URL"))

    def get_matrix(
        self,
        source_ids: List[str],
        target_ids: List[str],
        mode: Optional[str] = None,
    ) -> List[RouteMatrixEntry]:

        if not source_ids or not target_ids:
            return []

        query = """
        SELECT
            source_id,
            target_id,
            mode,
            distance_km,
            time_seconds
        FROM route_matrix
        WHERE
            source_id = ANY(:source_ids)
            AND target_id = ANY(:target_ids)
        """

        params = {
            "source_ids": source_ids,
            "target_ids": target_ids,
        }

        if mode is not None:
            query += " AND mode = :mode"
            params["mode"] = mode

        with self.engine.connect() as conn:
            rows = conn.execute(text(query), params)

            return [
                RouteMatrixEntry(
                    source_id=row.source_id,
                    target_id=row.target_id,
                    mode=row.mode,
                    distance_km=row.distance_km,
                    time_seconds=row.time_seconds,
                )
                for row in rows
            ]