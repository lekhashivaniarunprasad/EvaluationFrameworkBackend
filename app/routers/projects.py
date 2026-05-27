import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.core.config import database, settings
from app.core.auth import get_session_user

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    language: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
   

class FrameworkSelect(BaseModel):
    framework: str  # deepeval | arize_phoenix | trulens | opik


VALID_FRAMEWORKS = ["deepeval", "arize_phoenix", "trulens", "opik"]


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC FRAMEWORK PROJECT SETUP HOOK
# Each framework runner exposes a setup_project() function.
# projects.py calls it here — no framework-specific logic lives here.
# ══════════════════════════════════════════════════════════════════════════════

async def _run_framework_project_setup(
    framework: str,
    project_id: str,
    project_name: str,
) -> dict:
    """
    Calls the framework-specific setup_project() if it exists.
    Returns a dict with framework_project_id, framework_project_name,
    and dashboard_url (all optional — empty string if not applicable).

    Each runner is responsible for:
      - Creating its own external project
      - Inserting a row into framework_projects
      - Returning the dashboard URL
    """
    result = {
        "framework_project_id":   None,
        "framework_project_name": None,
        "dashboard_url":          None,
    }

    try:
        if framework == "opik":
            from app.evaluators.opik_runner import setup_project
            result = await setup_project(project_id=project_id, project_name=project_name)

        elif framework == "deepeval":
            pass  # deepeval has no external project concept — add runner import here when ready

        elif framework == "arize_phoenix":
            from app.evaluators.arize_phoenix_runner import setup_project
            result = await setup_project(project_id=project_id, project_name=project_name)

        elif framework == "trulens":
            from app.evaluators.trulens_runner import setup_project
            result = await setup_project(project_id=project_id, project_name=project_name)

    except Exception as e:
        print(f"[Projects] ⚠️  Framework setup failed for '{framework}': {e}")
        raise

    return result


# ── Step 1: Create project ────────────────────────────────────────────────────
@router.post("/")
async def create_project(body: ProjectCreate, current_user=Depends(get_session_user)):
    project_id = str(uuid.uuid4())

    await database.execute(
        """
        INSERT INTO projects 
            (id, user_id, name, description, language, llm_provider, llm_model, framework)
        VALUES 
            (:id, :user_id, :name, :description, :language, :llm_provider, :llm_model, 'pending')
        """,
        {
            "id":           project_id,
            "user_id":      current_user["user_id"],
            "name":         body.name,
            "description":  body.description,
            "language":     body.language,
            "llm_provider": body.llm_provider,
            "llm_model":    body.llm_model,
        },
    )

    return {
        "project_id": project_id,
        "message":    "Project created. Proceed to select a framework.",
    }


# ── Step 2: Select framework ──────────────────────────────────────────────────
@router.post("/{project_id}/select-framework")
async def select_framework(
    project_id: str,
    body: FrameworkSelect,
    current_user=Depends(get_session_user),
):
    row = await database.fetch_one(
        "SELECT id, name FROM projects WHERE id = :id AND user_id = :uid",
        {"id": project_id, "uid": current_user["user_id"]},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    if body.framework not in VALID_FRAMEWORKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid framework. Choose from: {VALID_FRAMEWORKS}",
        )

    short_id      = uuid.uuid4().hex[:8]
    api_endpoint  = f"/{body.framework}/{short_id}"
    platform_name = dict(row)["name"]

    # ── Call framework-specific project setup (runner handles everything) ─────
    setup = await _run_framework_project_setup(
        framework    = body.framework,
        project_id   = project_id,
        project_name = platform_name,
    )

    # ── Update projects table ─────────────────────────────────────────────────
    await database.execute(
        """
        UPDATE projects
        SET framework     = :framework,
            api_key       = CAST(:api_key AS VARCHAR),
            status        = 'active',
            dashboard_url = :dashboard_url
        WHERE id = :id
        """,
        {
            "framework":     body.framework,
            "api_key":       short_id,
            "dashboard_url": setup.get("dashboard_url"),
            "id":            project_id,
        },
    )

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        "project_id":   project_id,
        "framework":    body.framework,
        "api_key":      short_id,
        "api_endpoint": api_endpoint,
        "full_url":     f"http://localhost:8000/evaluate{api_endpoint}",
        "message":      "Framework selected. Your unique evaluation endpoint is ready.",
    }

    # Attach framework project info if the runner returned any
    if setup.get("framework_project_id"):
        response["framework_project_id"]   = setup["framework_project_id"]
        response["framework_project_name"] = setup["framework_project_name"]
        response["dashboard_url"]          = setup["dashboard_url"]

    return response


# ── List all projects for current user ────────────────────────────────────────
@router.get("/")
async def list_projects(current_user=Depends(get_session_user)):
    rows = await database.fetch_all(
        """
        SELECT p.*,
               fp.framework_project_id,
               fp.framework_project_name,
               COUNT(er.id)       AS total_runs,
               MAX(er.created_at) AS last_run_at
        FROM projects p
        LEFT JOIN framework_projects fp ON fp.id = p.id
        LEFT JOIN evaluation_runs er    ON er.project_id = p.id
        WHERE p.user_id = :uid
        GROUP BY p.id, fp.framework_project_id, fp.framework_project_name
        ORDER BY p.created_at DESC
        """,
        {"uid": current_user["user_id"]},
    )
    return [dict(r) for r in rows]


# ── Get single project ────────────────────────────────────────────────────────
@router.get("/{project_id}")
async def get_project(project_id: str, current_user=Depends(get_session_user)):
    row = await database.fetch_one(
        """
        SELECT p.*, fp.framework_project_id, fp.framework_project_name
        FROM projects p
        LEFT JOIN framework_projects fp ON fp.id = p.id
        WHERE p.id = :id AND p.user_id = :uid
        """,
        {"id": project_id, "uid": current_user["user_id"]},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    runs = await database.fetch_all(
        "SELECT * FROM evaluation_runs WHERE project_id = :pid ORDER BY created_at DESC LIMIT 10",
        {"pid": project_id},
    )

    project = dict(row)

    if project.get("api_key") and project.get("framework") and project["framework"] != "pending":
        project["api_endpoint"] = f"/{project['framework']}/{project['api_key']}"
        project["full_url"]     = f"http://localhost:8000/evaluate{project['api_endpoint']}"

    # Build dashboard URL via the framework runner
    if project.get("framework") == "opik" and project.get("framework_project_id"):
        from app.evaluators.opik_runner import get_dashboard_url
        project["dashboard_url"] = get_dashboard_url(project["framework_project_id"])
    elif project.get("framework") == "arize_phoenix" and project.get("framework_project_id"):
        from app.evaluators.arize_phoenix_runner import get_dashboard_url
        project["dashboard_url"] = get_dashboard_url(project["framework_project_id"])

    if project.get("framework") == "trulens":
        from app.evaluators.trulens_runner import get_dashboard_url as tru_get_url
        project["dashboard_url"] = tru_get_url(project.get("framework_project_id", ""))

    return {
        **project,
        "recent_runs": [dict(r) for r in runs],
    }


# ── Delete project ────────────────────────────────────────────────────────────
@router.delete("/{project_id}")
async def delete_project(project_id: str, current_user=Depends(get_session_user)):
    row = await database.fetch_one(
        "SELECT id FROM projects WHERE id = :id AND user_id = :uid",
        {"id": project_id, "uid": current_user["user_id"]},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    # framework_projects row deleted automatically via ON DELETE CASCADE
    await database.execute("DELETE FROM projects WHERE id = :id", {"id": project_id})
    return {"message": "Project deleted"}


@router.get("/{project_id}/dashboard-url")
async def get_dashboard_url(project_id: str, current_user=Depends(get_session_user)):
    row = await database.fetch_one(
        "SELECT dashboard_url FROM projects WHERE id = :id AND user_id = :uid",
        {"id": project_id, "uid": current_user["user_id"]},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project_id":    project_id,
        "dashboard_url": dict(row).get("dashboard_url") or "",
    }
