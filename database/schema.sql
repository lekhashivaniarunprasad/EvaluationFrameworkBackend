-- ============================================================
-- LLM Evaluation Platform - Full Database Schema
-- Database: evaluation_db
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- USERS
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name       VARCHAR(255) NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    company         VARCHAR(255),
    role            VARCHAR(100),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- SESSIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token   TEXT UNIQUE NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- PROJECTS
-- ============================================================
CREATE TABLE IF NOT EXISTS projects (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    language        TEXT,
    llm_provider    TEXT,
    llm_model       TEXT,
    framework       VARCHAR(50) NOT NULL DEFAULT 'pending',
    api_key         VARCHAR(20) UNIQUE,        -- short 8-char unique id
    status          VARCHAR(50) DEFAULT 'draft', -- draft | active | archived
    dashboard_url   TEXT,                        -- framework dashboard URL (e.g. Confident AI)
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- EVALUATION RUNS
-- ============================================================
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_label       TEXT,
    status          VARCHAR(50) DEFAULT 'pending', -- pending | running | completed | failed
    framework       VARCHAR(50) NOT NULL,
    dashboard_url   TEXT,
    external_run_id TEXT,
    total_turns     INTEGER DEFAULT 0,
    payload_raw     JSONB,
    results         TEXT,                        -- full JSON results
    score           FLOAT,                       -- overall average score
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- EVALUATION RESULTS (per turn)
-- ============================================================
CREATE TABLE IF NOT EXISTS evaluation_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    turn_index      INTEGER NOT NULL,
    user_input      TEXT NOT NULL,
    llm_output      TEXT NOT NULL,
    context         TEXT[],
    expected_output TEXT,
    raw_result      JSONB,
    passed          BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_sessions_token        ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user         ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_user         ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_api_key      ON projects(api_key);
CREATE INDEX IF NOT EXISTS idx_eval_runs_project     ON evaluation_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_run      ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_project  ON evaluation_results(project_id);

-- ============================================================
-- AUTO-UPDATE updated_at TRIGGER
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
