from app.core.config import engine
from sqlalchemy import text


def create_tables():
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto";'))

        conn.execute(text("""
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
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_token   TEXT UNIQUE NOT NULL,
                expires_at      TIMESTAMPTZ NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS projects (
                id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name            VARCHAR(255) NOT NULL,
                description     TEXT,
                language        VARCHAR(100),
                llm_provider    VARCHAR(100),
                llm_model       VARCHAR(100),
                framework       VARCHAR(50) NOT NULL DEFAULT 'pending',
                api_key         VARCHAR(20) UNIQUE,
                status          VARCHAR(50) DEFAULT 'draft',
                dashboard_url   TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                run_label       VARCHAR(255),
                status          VARCHAR(50) DEFAULT 'pending',
                framework       VARCHAR(50) NOT NULL,
                dashboard_url   TEXT,
                external_run_id TEXT,
                total_turns     INTEGER DEFAULT 0,
                payload_raw     JSONB,
                error_message   TEXT,
                started_at      TIMESTAMPTZ,
                completed_at    TIMESTAMPTZ,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """))

        conn.execute(text("""
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
        """))

        # Indexes
        # Safe migration — add dashboard_url to projects if it doesn't exist yet
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='projects' AND column_name='dashboard_url'
                ) THEN
                    ALTER TABLE projects ADD COLUMN dashboard_url TEXT;
                END IF;
            END$$;
        """))

        # Safe migration — add results and score to evaluation_runs if missing
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='evaluation_runs' AND column_name='results'
                ) THEN
                    ALTER TABLE evaluation_runs ADD COLUMN results TEXT;
                END IF;
            END$$;
        """))
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='evaluation_runs' AND column_name='score'
                ) THEN
                    ALTER TABLE evaluation_runs ADD COLUMN score FLOAT;
                END IF;
            END$$;
        """))

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_token       ON sessions(session_token);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user        ON sessions(user_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_projects_user        ON projects(user_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_projects_api_key     ON projects(api_key);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eval_runs_project    ON evaluation_runs(project_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eval_results_run     ON evaluation_results(run_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eval_results_project ON evaluation_results(project_id);"))

        # Trigger function
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION update_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """))

        conn.execute(text("""
            DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
            CREATE TRIGGER trg_users_updated_at
                BEFORE UPDATE ON users
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """))

        conn.execute(text("""
            DROP TRIGGER IF EXISTS trg_projects_updated_at ON projects;
            CREATE TRIGGER trg_projects_updated_at
                BEFORE UPDATE ON projects
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS framework_projects (
                id                   UUID PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                framework_project_id   TEXT NOT NULL,
                framework_project_name VARCHAR(255) NOT NULL
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_framework_projects_id ON framework_projects(id);"))

        # Safe migrations - widen legacy VARCHAR columns created in older schemas
        conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'evaluation_runs'
                      AND column_name = 'run_label'
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE evaluation_runs ALTER COLUMN run_label TYPE TEXT;
                END IF;
            END$$;
        """))

        conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'framework_projects'
                      AND column_name = 'framework_project_id'
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE framework_projects ALTER COLUMN framework_project_id TYPE TEXT;
                END IF;

                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'framework_projects'
                      AND column_name = 'framework_project_name'
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE framework_projects ALTER COLUMN framework_project_name TYPE TEXT;
                END IF;
            END$$;
        """))

        conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'projects'
                      AND column_name = 'language'
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE projects ALTER COLUMN language TYPE TEXT;
                END IF;

                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'projects'
                      AND column_name = 'llm_provider'
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE projects ALTER COLUMN llm_provider TYPE TEXT;
                END IF;

                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'projects'
                      AND column_name = 'llm_model'
                      AND data_type = 'character varying'
                ) THEN
                    ALTER TABLE projects ALTER COLUMN llm_model TYPE TEXT;
                END IF;
            END$$;
        """))
        conn.commit()
        print("✅ Database tables verified / created successfully.")





