import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
from app.core.config import database
import json

router = APIRouter(prefix="/evaluate", tags=["evaluate"])

VALID_FRAMEWORKS = ["deepeval", "arize_phoenix", "trulens", "opik"]


class ConversationTurn(BaseModel):
    user_input: str
    llm_output: str
    context: Optional[List[str]] = []
    expected_output: Optional[str] = None


class EvaluationPayload(BaseModel):
    session_label: Optional[str] = None
    conversation: List[ConversationTurn]
    selected_metrics: Optional[List[str]] = None
    metric_thresholds: Optional[Dict[str, float]] = None


async def run_evaluation_background(
    run_id: str,
    project_id: str,
    framework: str,
    config: dict,
    payload: EvaluationPayload,
):
    try:
        await database.execute(
            "UPDATE evaluation_runs SET status = 'running', started_at = :now WHERE id = :id",
            {"now": datetime.now(timezone.utc), "id": run_id},
        )

        if framework == "deepeval":
            from app.evaluators.deepeval_runner import run_deepeval
            dashboard_url, external_run_id = await run_deepeval(run_id, project_id, config, payload)
        elif framework == "arize_phoenix":
            from app.evaluators.arize_phoenix_runner import run_arize_phoenix
            dashboard_url, external_run_id = await run_arize_phoenix(run_id, project_id, config, payload)
        elif framework == "trulens":
            from app.evaluators.trulens_runner import run_trulens
            dashboard_url, external_run_id = await run_trulens(run_id, project_id, config, payload)
        elif framework == "opik":
            from app.evaluators.opik_runner import run_opik
            dashboard_url, external_run_id = await run_opik(run_id, project_id, config, payload)
        else:
            raise ValueError(f"Unknown framework: {framework}")

        await database.execute(
            """
            UPDATE evaluation_runs 
            SET status = 'completed', completed_at = :now,
                dashboard_url = :dashboard_url, external_run_id = :ext_id
            WHERE id = :id
            """,
            {
                "now": datetime.now(timezone.utc),
                "dashboard_url": dashboard_url,
                "ext_id": external_run_id,
                "id": run_id,
            },
        )
    except Exception as e:
        await database.execute(
            "UPDATE evaluation_runs SET status = 'failed', error_message = :err WHERE id = :id",
            {"err": str(e), "id": run_id},
        )


# ── Main evaluation endpoint: POST /evaluate/{framework}/{unique_id} ──────────
@router.post("/{framework}/{unique_id}")
async def evaluate(
    framework: str,
    unique_id: str,
    payload: EvaluationPayload,
    background_tasks: BackgroundTasks,
):
    if framework not in VALID_FRAMEWORKS:
        raise HTTPException(status_code=400, detail=f"Invalid framework: {framework}")

    # Lookup project by api_key (unique_id) and framework
    project = await database.fetch_one(
        """
        SELECT p.*, fp.framework_project_id, fp.framework_project_name
        FROM projects p
        LEFT JOIN framework_projects fp ON fp.id = p.id
        WHERE CAST(p.api_key AS VARCHAR) = :key AND p.framework = :framework AND p.status = 'active'
        """,
        {"key": unique_id, "framework": framework},
    )
    if not project:
        raise HTTPException(status_code=404, detail="Invalid endpoint. Check your framework and unique ID.")

    project = dict(project)
    run_id = str(uuid.uuid4())

    import json
    await database.execute(
        """
        INSERT INTO evaluation_runs 
            (id, project_id, run_label, status, framework, total_turns, payload_raw)
        VALUES (:id, :project_id, :label, 'pending', :framework, :turns, :raw)
        """,
        {
            "id": run_id,
            "project_id": project["id"],
            "label": payload.session_label or f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "framework": framework,
            "turns": len(payload.conversation),
            "raw": json.dumps(payload.dict()),
        },
    )

    background_tasks.add_task(
        run_evaluation_background,
        run_id=run_id,
        project_id=str(project["id"]),
        framework=framework,
        config=project,
        payload=payload,
    )

    return {
        "run_id": run_id,
        "status": "pending",
        "message": f"Evaluation started using {framework}. Poll /evaluate/run/{run_id}/status for results.",
        "framework": framework,
    }


# ── Poll run status ───────────────────────────────────────────────────────────
@router.get("/run/{run_id}/status")
async def run_status(run_id: str):
    row = await database.fetch_one(
        "SELECT * FROM evaluation_runs WHERE id = :id", {"id": run_id}
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    results = await database.fetch_all(
        "SELECT * FROM evaluation_results WHERE run_id = :id ORDER BY turn_index ASC",
        {"id": run_id}
    )

    run_data = dict(row)

    turn_details = []
    for r in results:
        r_dict = dict(r)
        raw = r_dict.get("raw_result")
        if raw:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            r_dict["metrics"]        = parsed.get("metrics", [])
            # average_score is the correct field from deepeval_runner
            r_dict["overall_score"]  = parsed.get("average_score") or parsed.get("overall_score")
            r_dict["overall_passed"] = parsed.get("overall_passed")
            r_dict["metrics_passed"] = parsed.get("metrics_passed")
            r_dict["metrics_failed"] = parsed.get("metrics_failed")
        turn_details.append(r_dict)

    run_data["turns"]         = turn_details
    run_data["dashboard_url"] = run_data.get("dashboard_url") or ""

    return run_data
