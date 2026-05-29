"""
app/evaluators/opik_runner.py

Self-contained Opik evaluation runner.

Public API (called by projects.py):
    setup_project(project_id, project_name) -> dict
    get_dashboard_url(framework_project_id) -> str

Internal entry point (called by evaluate.py):
    run_opik(run_id, project_id, config, payload) -> (dashboard_url, external_run_id)
"""

import os
import re
import json
import uuid
import ssl
import asyncio
import traceback
import warnings

# ── MUST be set before any opik import so httpx client picks it up ────────────
os.environ["OPIK_CHECK_TLS_CERTIFICATE"] = "false"

import urllib3
import requests

# ── Corporate SSL bypass ──────────────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
os.environ["CURL_CA_BUNDLE"]      = ""
os.environ["REQUESTS_CA_BUNDLE"]  = ""
os.environ["PYTHONHTTPSVERIFY"]   = "0"
ssl._create_default_https_context = ssl._create_unverified_context

from app.core.config import database, settings
from app.evaluators.metric_config import (
    get_metric_label,
    resolve_metric_id,
    resolve_metric_ids,
    resolve_metric_thresholds,
)
from app.routers.evaluate import EvaluationPayload


# ── Available metrics ─────────────────────────────────────────────────────────
DEFAULT_METRICS = [
    "exact_match",
    "contains",
    "regex_match",
    "sentence_bleu",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "rouge_lsum",
    "hallucination",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "moderation",
    "usefulness",
    "meaning_match",
    "summarization_consistency",
    "summarization_coherence",
    "dialogue_helpfulness",
    "compliance_risk",
    "prompt_uncertainty",
    "g_eval",
    "trajectory_accuracy",
    "task_completion",
    "tool_correctness",
    "conversational_thread_metrics",
    "custom_conversation_metric",
    "custom_metric",
]


def _normalize_context(context):
    if context is None:
        return []
    if isinstance(context, str):
        return [context] if context.strip() else []
    if isinstance(context, list):
        return [str(item).strip() for item in context if str(item).strip()]
    return [str(context).strip()] if str(context).strip() else []


def _print_retrieved_context(framework_name: str, question: str, contexts: list) -> None:
    print(f"\n[{framework_name}] Retrieved context for evaluation input: {question}", flush=True)
    if contexts:
        for index, chunk in enumerate(contexts, start=1):
            print(f"[{framework_name}] Context chunk {index}:\n{chunk}\n", flush=True)
    else:
        print(f"[{framework_name}] No context provided for this turn.\n", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — called by projects.py
# ══════════════════════════════════════════════════════════════════════════════

async def setup_project(project_id: str, project_name: str) -> dict:
    """
    Called by projects.py when a user selects the opik framework.
    1. Creates a new isolated Opik project
    2. Inserts a row into framework_projects table
    3. Returns framework_project_id, framework_project_name, dashboard_url
    """
    if not settings.OPIK_API_KEY or not settings.OPIK_WORKSPACE:
        raise ValueError("OPIK_API_KEY and OPIK_WORKSPACE must be configured.")

    opik_project_name, opik_project_id = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _create_opik_project(project_id, project_name),
    )

    existing = await database.fetch_one(
        "SELECT id FROM framework_projects WHERE id = :id",
        {"id": project_id},
    )

    if existing:
        await database.execute(
            """
            UPDATE framework_projects
            SET framework_project_id   = :framework_project_id,
                framework_project_name = :framework_project_name
            WHERE id = :id
            """,
            {
                "framework_project_id":   opik_project_id,
                "framework_project_name": opik_project_name,
                "id":                     project_id,
            },
        )
    else:
        await database.execute(
            """
            INSERT INTO framework_projects
                (id, framework_project_id, framework_project_name)
            VALUES
                (:id, :framework_project_id, :framework_project_name)
            """,
            {
                "id":                     project_id,
                "framework_project_id":   opik_project_id,
                "framework_project_name": opik_project_name,
            },
        )

    print(f"[OpikRunner] ✅ framework_projects row saved for project {project_id}")

    return {
        "framework_project_id":   opik_project_id,
        "framework_project_name": opik_project_name,
        "dashboard_url":          get_dashboard_url(opik_project_id),
    }


def get_dashboard_url(framework_project_id: str) -> str:
    """Builds the correct Opik dashboard URL from the stored project ID."""
    workspace = settings.OPIK_WORKSPACE
    if workspace and framework_project_id:
        return (
            f"https://www.comet.com/opik/{workspace}/projects/"
            f"{framework_project_id}/traces?time_range=past30days"
        )
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE — Opik project creation via REST API
# ══════════════════════════════════════════════════════════════════════════════

def _create_opik_project(platform_project_id: str, platform_project_name: str) -> tuple:
    """
    Creates a new Opik project via REST API.
    Project name = user's project name + short ID for uniqueness.
    Returns (opik_project_name, opik_project_id)
    """
    import opik

    short_id     = platform_project_id[:8]
    project_name = f"{platform_project_name} ({short_id})"

    opik.configure(
        api_key   = settings.OPIK_API_KEY,
        workspace = settings.OPIK_WORKSPACE,
        use_local = False,
        force     = True,
    )

    base_url = "https://www.comet.com/opik/api"
    headers  = {
        "Content-Type":    "application/json",
        "authorization":   settings.OPIK_API_KEY,
        "Comet-Workspace": settings.OPIK_WORKSPACE,
    }

    def _fetch_project_id(name: str):
        resp = requests.get(
            f"{base_url}/v1/private/projects",
            headers = headers,
            params  = {"page": 1, "size": 100},
            timeout = 30,
            verify  = False,
        )
        if resp.status_code == 200 and resp.text.strip():
            data  = resp.json()
            items = data.get("content", data.get("projects", []))
            for p in items:
                if p.get("name") == name:
                    return p.get("id")
        return None

    # Check if already exists
    existing_id = _fetch_project_id(project_name)
    if existing_id:
        print(f"  [Opik] ✅ Found existing project: '{project_name}' id={existing_id}")
        return project_name, str(existing_id)

    # Create
    resp = requests.post(
        f"{base_url}/v1/private/projects",
        headers = headers,
        json    = {"name": project_name},
        timeout = 30,
        verify  = False,
    )

    print(f"  [Opik] Create response: {resp.status_code} | body: '{resp.text[:100]}'")

    if resp.status_code not in (200, 201, 204, 409):
        raise RuntimeError(f"Opik project creation failed: {resp.status_code} {resp.text[:200]}")

    if resp.text.strip():
        try:
            data = resp.json()
            pid  = data.get("id") or data.get("projectId")
            if pid:
                print(f"  [Opik] ✅ Created project: '{project_name}' id={pid}")
                return project_name, str(pid)
        except Exception:
            pass

    project_id = _fetch_project_id(project_name)
    if project_id:
        print(f"  [Opik] ✅ Created + fetched: '{project_name}' id={project_id}")
        return project_name, str(project_id)

    raise RuntimeError(f"Created Opik project '{project_name}' but could not retrieve its ID.")


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE — Opik client for trace logging
# ══════════════════════════════════════════════════════════════════════════════

def _get_opik_client(opik_project_name: str):
    """
    Connects to Opik cloud. Returns (client, project_name) or (None, None).
    """
    try:
        import opik
        opik.configure(
            api_key   = settings.OPIK_API_KEY,
            workspace = settings.OPIK_WORKSPACE,
            use_local = False,
            force     = True,
        )
        client = opik.Opik(project_name=opik_project_name)
        print(f"  [Opik] ✅ Connected — project: '{opik_project_name}'")
        return client, opik_project_name
    except Exception as e:
        print(f"  [Opik] ❌ Could not connect: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# PwC GENAI CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class PwCGenAIClient:
    def __init__(self):
        self.api_key  = settings.PWC_GENAI_API_KEY
        self.base_url = settings.PWC_GENAI_BASE_URL.rstrip("/")
        self.model    = settings.PWC_GENAI_MODEL

        if not self.api_key:
            raise ValueError(
                "PWC_GENAI_API_KEY is not set in environment. "
                "Please set it and restart the server."
            )
        print(f"  [PwC] Client initialised — model: {self.model}")

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/completions"
        headers = {
            "accept":        "application/json",
            "Content-Type":  "application/json",
            "API-Key":       self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model":            self.model,
            "prompt":           prompt,
            "temperature":      0.0,
            "top_p":            1,
            "presence_penalty": 0,
            "seed":             42,
            "stream":           False,
            "max_tokens":       1000,
        }
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.PWC_GENAI_TIMEOUT_SECONDS,
            verify=False,
        )
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "text" in choice:
                return choice["text"].strip()
            if "message" in choice:
                return choice["message"].get("content", "").strip()

        raise ValueError(f"Unexpected PwC response format: {json.dumps(data)[:300]}")


# ══════════════════════════════════════════════════════════════════════════════
# BASE METRIC
# ══════════════════════════════════════════════════════════════════════════════

class BaseRAGMetric:
    name = "BaseMetric"

    def __init__(self, client: PwCGenAIClient):
        self.client = client

    def _extract_score(self, text: str) -> tuple:
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data   = json.loads(match.group())
                score  = float(data.get("score", 0.0))
                reason = str(data.get("reason", ""))
                return round(max(0.0, min(1.0, score)), 4), reason
        except Exception:
            pass

        match = re.search(r'\b(0\.\d+|1\.0|0|1)\b', text)
        if match:
            score = float(match.group())
            return round(max(0.0, min(1.0, score)), 4), text.strip()

        return 0.0, f"Could not parse score from: {text[:200]}"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

class HallucinationMetric(BaseRAGMetric):
    name = "Hallucination"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        context_text = "\n".join(context) if context else "No context provided."
        prompt = f"""You are an expert evaluator assessing whether an AI response contains hallucinations.

QUESTION: {input}

RETRIEVED CONTEXT:
{context_text}

AI RESPONSE: {output}

TASK: Evaluate if the AI response contains information that is NOT supported by the retrieved context.

Scoring:
- 1.0 = Response is fully supported by the context, no hallucinations
- 0.5 = Response is partially supported, some unsupported claims
- 0.0 = Response contains significant hallucinations not in the context

Respond ONLY with a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

        try:
            raw = self.client.complete(prompt)
            print(f"  [Hallucination] Raw: {raw[:150]}")
            score, reason = self._extract_score(raw)
            return {"score": score, "reason": reason}
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {str(e)}"}


class AnswerRelevanceMetric(BaseRAGMetric):
    name = "Answer Relevance"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        prompt = f"""You are an expert evaluator assessing the relevance of an AI response.

QUESTION: {input}

AI RESPONSE: {output}

{"EXPECTED ANSWER: " + expected_output if expected_output else ""}

TASK: Evaluate how relevant and on-topic the AI response is to the question.

Scoring:
- 1.0 = Response directly and completely answers the question
- 0.5 = Response is partially relevant but misses key aspects
- 0.0 = Response is completely irrelevant to the question

Respond ONLY with a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

        try:
            raw = self.client.complete(prompt)
            print(f"  [AnswerRelevance] Raw: {raw[:150]}")
            score, reason = self._extract_score(raw)
            return {"score": score, "reason": reason}
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {str(e)}"}


class AnswerCorrectnessMetric(BaseRAGMetric):
    name = "Answer Correctness"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        if not expected_output.strip():
            return {"score": 0.0, "reason": "Expected answer is required for answer correctness."}

        prompt = f"""You are an expert evaluator assessing whether an AI response matches the expected answer.

QUESTION: {input}

EXPECTED ANSWER:
{expected_output}

AI RESPONSE:
{output}

TASK: Score semantic correctness of the AI response against the EXPECTED ANSWER.
Do not score based only on relevance to the question. If the AI response contradicts
or differs materially from the expected answer, the score must be 0.0.

Scoring:
- 1.0 = AI response is semantically equivalent to the expected answer
- 0.5 = AI response is partially correct but incomplete
- 0.0 = AI response is incorrect, contradicts, or does not contain the expected answer

Respond ONLY with a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

        try:
            raw = self.client.complete(prompt)
            print(f"  [AnswerCorrectness] Raw: {raw[:150]}")
            score, reason = self._extract_score(raw)
            return {"score": score, "reason": reason}
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {str(e)}"}


class ContextPrecisionMetric(BaseRAGMetric):
    name = "Context Precision"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        context_text = "\n".join(
            [f"[Chunk {i+1}]: {c}" for i, c in enumerate(context)]
        ) if context else "No context provided."

        prompt = f"""You are an expert evaluator assessing the precision of retrieved context for RAG systems.

QUESTION: {input}

RETRIEVED CONTEXT:
{context_text}

AI RESPONSE: {output}

TASK: Evaluate whether the retrieved context chunks are relevant and useful for answering the question.

Scoring:
- 1.0 = All retrieved context is highly relevant and useful
- 0.5 = Some context is relevant, but there is significant noise
- 0.0 = Retrieved context is completely irrelevant to the question

Respond ONLY with a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

        try:
            raw = self.client.complete(prompt)
            print(f"  [ContextPrecision] Raw: {raw[:150]}")
            score, reason = self._extract_score(raw)
            return {"score": score, "reason": reason}
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {str(e)}"}


class ContextRecallMetric(BaseRAGMetric):
    name = "Context Recall"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        context_text = "\n".join(context) if context else "No context provided."

        prompt = f"""You are an expert evaluator assessing the recall of retrieved context for RAG systems.

QUESTION: {input}

RETRIEVED CONTEXT:
{context_text}

{"EXPECTED ANSWER: " + expected_output if expected_output else ""}

AI RESPONSE: {output}

TASK: Evaluate whether the retrieved context contains ALL the information needed to produce the EXPECTED ANSWER.
If the EXPECTED ANSWER is not supported by the retrieved context, score 0.0 even if the context answers the question differently.

Scoring:
- 1.0 = Context contains all information needed to fully answer the question
- 0.5 = Context contains some relevant information but is missing key details
- 0.0 = Context is missing almost all information needed to answer the question

Respond ONLY with a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

        try:
            raw = self.client.complete(prompt)
            print(f"  [ContextRecall] Raw: {raw[:150]}")
            score, reason = self._extract_score(raw)
            return {"score": score, "reason": reason}
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# METRIC BUILDER
# ══════════════════════════════════════════════════════════════════════════════

GENERIC_METRIC_CRITERIA = {
    "moderation": "Flag content that violates safety or policy guidelines.",
    "usefulness": "Evaluate overall helpfulness of the response to the user.",
    "meaning_match": "Check semantic equivalence against the expected or reference answer.",
    "summarization_consistency": "Check that the summary does not contradict the source context.",
    "summarization_coherence": "Check that the summary is logically structured and readable.",
    "dialogue_helpfulness": "Evaluate conversational quality across a support-style interaction.",
    "compliance_risk": "Evaluate regulatory, legal, privacy, or policy risk in the output.",
    "prompt_uncertainty": "Assess ambiguity or low-confidence signals in the prompt and response.",
    "g_eval": "Apply a general LLM-as-judge quality rubric using the available input, context, output, and reference.",
    "trajectory_accuracy": "Check if the agent followed the correct execution path; score 0 if trace/tool path data is unavailable.",
    "task_completion": "Evaluate whether the agent successfully completed the assigned task.",
    "tool_correctness": "Check if the correct tool was selected and invoked; score 0 if tool-call data is unavailable.",
    "conversational_thread_metrics": "Evaluate quality across the available conversation turn/thread.",
    "custom_conversation_metric": "Apply a general conversation-level quality rubric to the available turn/thread.",
    "custom_metric": "Apply a general custom quality rubric to the available evaluation fields.",
}


class GenericLLMJudgeMetric(BaseRAGMetric):
    def __init__(self, client: PwCGenAIClient, metric_id: str, name: str, criteria: str):
        super().__init__(client)
        self.metric_id = metric_id
        self.name = name
        self.criteria = criteria

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        context_text = "\n".join(context) if context else "No context provided."
        expected = expected_output or "No expected output provided."
        prompt = f"""You are an expert evaluator scoring the metric: {self.name}.

QUESTION:
{input}

RETRIEVED CONTEXT:
{context_text}

EXPECTED OUTPUT / REFERENCE:
{expected}

AI RESPONSE:
{output}

TASK:
{self.criteria}

If this metric requires data that is not present in the fields above, give score 0.0 and explain what is missing.

Respond ONLY with a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""
        try:
            raw = self.client.complete(prompt)
            print(f"  [{self.name}] Raw: {raw[:150]}")
            score, reason = self._extract_score(raw)
            return {"score": score, "reason": reason}
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {str(e)}"}


class ExactMatchMetric(BaseRAGMetric):
    name = "Exact Match"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        if not expected_output:
            return {"score": 0.0, "reason": "Expected output is required for exact match."}
        passed = output.strip() == expected_output.strip()
        return {"score": 1.0 if passed else 0.0, "reason": "Output exactly matches expected output." if passed else "Output does not exactly match expected output."}


class ContainsMetric(BaseRAGMetric):
    name = "Contains"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        if not expected_output:
            return {"score": 0.0, "reason": "Expected output must contain the required substring."}
        passed = expected_output.lower() in output.lower()
        return {"score": 1.0 if passed else 0.0, "reason": "Output contains required substring." if passed else "Output does not contain required substring."}


class RegexMatchMetric(BaseRAGMetric):
    name = "Regex Match"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        if not expected_output:
            return {"score": 0.0, "reason": "Expected output must contain the regex pattern."}
        try:
            passed = re.search(expected_output, output) is not None
            return {"score": 1.0 if passed else 0.0, "reason": "Output matches regex pattern." if passed else "Output does not match regex pattern."}
        except re.error as e:
            return {"score": 0.0, "reason": f"Invalid regex pattern: {e}"}


class OverlapMetric(BaseRAGMetric):
    def __init__(self, client: PwCGenAIClient, metric_id: str, name: str):
        super().__init__(client)
        self.metric_id = metric_id
        self.name = name

    @staticmethod
    def _tokens(text: str) -> list:
        return re.findall(r"\w+", str(text or "").lower())

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        reference = expected_output or " ".join(context)
        if not reference:
            return {"score": 0.0, "reason": "Expected output or context is required for overlap metrics."}
        out_tokens = self._tokens(output)
        ref_tokens = self._tokens(reference)
        if not out_tokens or not ref_tokens:
            return {"score": 0.0, "reason": "Output or reference has no comparable tokens."}
        out_set = set(out_tokens)
        ref_set = set(ref_tokens)
        precision = len(out_set & ref_set) / len(out_set)
        recall = len(out_set & ref_set) / len(ref_set)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {"score": round(f1, 4), "reason": f"Approximate overlap score with precision={precision:.4f}, recall={recall:.4f}."}


def _normalise(s: str) -> str:
    return s.lower().strip().replace(" ", "").replace("_", "").replace("-", "")


CLASS_MAP = {
    "hallucination":    HallucinationMetric,
    "answerrelevance":  AnswerRelevanceMetric,
    "answerrelevancy":  AnswerRelevanceMetric,
    "answercorrectness": AnswerCorrectnessMetric,
    "faithfulness":     AnswerRelevanceMetric,
    "contextprecision": ContextPrecisionMetric,
    "contextrecall":    ContextRecallMetric,
    "exactmatch":       ExactMatchMetric,
    "contains":         ContainsMetric,
    "regexmatch":       RegexMatchMetric,
    "sentencebleu":     OverlapMetric,
    "corpusbleu":       OverlapMetric,
    "rouge1":           OverlapMetric,
    "rouge2":           OverlapMetric,
    "rougel":           OverlapMetric,
    "rougelsum":        OverlapMetric,
}


def _build_metrics(selected: list, client: PwCGenAIClient) -> list:
    metrics = []
    for name in selected:
        key = _normalise(name)
        cls = CLASS_MAP.get(key)
        if cls:
            try:
                metrics.append(cls(client=client))
                print(f"  [Metrics] ✅ {name} → {cls.__name__}")
            except Exception as e:
                print(f"  [Metrics] ❌ Failed '{name}': {e}")
        else:
            print(f"  [Metrics] ⚠️  Unknown metric: '{name}'")
    return metrics


def _build_metrics(selected: list, client: PwCGenAIClient) -> list:
    metrics = []
    for name in selected:
        key = _normalise(name)
        cls = CLASS_MAP.get(key)
        metric_id = resolve_metric_id("opik", name) or name
        if cls:
            try:
                if cls is OverlapMetric:
                    metrics.append(cls(
                        client=client,
                        metric_id=metric_id,
                        name=get_metric_label("opik", metric_id),
                    ))
                else:
                    metrics.append(cls(client=client))
                print(f"  [Metrics] built {name} with {cls.__name__}")
            except Exception as e:
                print(f"  [Metrics] failed '{name}': {e}")
        else:
            criteria = GENERIC_METRIC_CRITERIA.get(metric_id)
            if criteria:
                metrics.append(GenericLLMJudgeMetric(
                    client=client,
                    metric_id=metric_id,
                    name=get_metric_label("opik", metric_id),
                    criteria=criteria,
                ))
                print(f"  [Metrics] built {name} with GenericLLMJudgeMetric")
            else:
                print(f"  [Metrics] unknown metric: '{name}'")
    return metrics


def _resolve_metrics(config: dict) -> list:
    raw = config.get("opik_metrics") or config.get("metrics")
    if not raw:
        return DEFAULT_METRICS
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return DEFAULT_METRICS


# ══════════════════════════════════════════════════════════════════════════════
# OPIK TRACE LOGGER
# Uses opik.Opik().trace() — confirmed working pattern from Dify/production code
# ══════════════════════════════════════════════════════════════════════════════

def _log_trace_to_opik(
    opik_client,
    project_name: str,
    run_id: str,
    session_label: str,
    turn_index: int,
    turn,
    metric_results: list,
    overall_score: float,
):
    trace_name = f"{session_label} | Turn {turn_index + 1}"
    trace_id   = None

    try:
        contexts = _normalize_context(turn.context)
        # Confirmed API: opik.Opik().trace(**kwargs) — returns a Trace object
        trace = opik_client.trace(
            name         = trace_name,
            project_name = project_name,
            input        = {"user_input": turn.user_input, "context": contexts},
            output       = {"llm_output": turn.llm_output, "expected_output": turn.expected_output or ""},
            metadata     = {
                "run_id":        run_id,
                "session_label": session_label,
                "turn_index":    turn_index,
                "overall_score": overall_score,
                "framework":     "opik",
            },
            tags = ["rag-evaluation", "evalforge"],
        )

        trace_id = trace.id
        print(f"  [Opik] ✅ Trace created: '{trace_name}' id={trace_id}")

    except Exception as e:
        print(f"  [Opik] ❌ Trace creation failed: {e}")
        traceback.print_exc()
        return None

    # Log feedback scores
    for metric in metric_results:
        try:
            trace.log_feedback_score(
                name   = metric["name"],
                value  = float(metric["score"]),
                reason = metric.get("reason", ""),
            )
            print(f"  [Opik] ✅ Feedback: {metric['name']} = {metric['score']}")
        except Exception as e:
            print(f"  [Opik] ⚠️  Feedback failed '{metric['name']}': {e}")

    # End trace — finalises it and marks it ready for upload
    try:
        trace.end()
    except Exception as e:
        print(f"  [Opik] ⚠️  trace.end() warning: {e}")

    # Flush — sends all queued traces to Comet immediately
    try:
        opik_client.flush()
        print(f"  [Opik] ✅ Flushed: '{trace_name}'")
    except Exception as e:
        print(f"  [Opik] ⚠️  Flush warning: {e}")

    return trace_id


# ══════════════════════════════════════════════════════════════════════════════
# TURN RUNNER + DB PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _run_turn(turn, metrics: list, metric_thresholds: dict) -> dict:
    results = []
    contexts = _normalize_context(turn.context)
    _print_retrieved_context("OpikRunner", turn.user_input, contexts)
    for metric in metrics:
        print(f"\n  [Eval] Scoring: {metric.name}...")
        metric_id = resolve_metric_id("opik", metric.name) or metric.name
        threshold = metric_thresholds.get(metric_id, 0.5)
        try:
            result = metric.score(
                input           = turn.user_input,
                output          = turn.llm_output,
                expected_output = turn.expected_output or "",
                context         = contexts,
            )
            score  = float(result.get("score", 0.0))
            reason = result.get("reason", "")
            print(f"  [Eval] OK {metric.name} = {score:.4f}")
            results.append({
                "metric": metric_id,
                "name":   metric.name,
                "score":  round(score, 4),
                "reason": reason,
                "threshold": threshold,
                "passed": score >= threshold,
            })
        except Exception as e:
            print(f"  [Eval] FAIL {metric.name}: {e}")
            traceback.print_exc()
            results.append({
                "metric": metric_id,
                "name":   metric.name,
                "score":  0.0,
                "reason": f"Error: {str(e)}",
                "threshold": threshold,
                "passed": False,
            })

    valid   = [r["score"] for r in results if r["score"] > 0]
    overall = round(sum(valid) / len(valid), 4) if valid else 0.0

    return {
        "overall_score":  overall,
        "overall_passed": all(r["passed"] for r in results) if results else None,
        "metrics_passed": sum(1 for r in results if r["passed"]),
        "metrics_failed": sum(1 for r in results if not r["passed"]),
        "metrics":        results,
        "context":        contexts,
        "status":         "success" if valid else "error",
    }


async def _persist_turn_result(run_id, project_id, turn_index, turn, result):
    passed = result.get("overall_passed", False)
    await database.execute(
        """
        INSERT INTO evaluation_results
            (id, run_id, project_id, turn_index,
             user_input, llm_output, context, expected_output,
             raw_result, passed)
        VALUES
            (:id, :run_id, :project_id, :turn_index,
             :user_input, :llm_output, :context, :expected_output,
             :raw_result, :passed)
        """,
        {
            "id":              str(uuid.uuid4()),
            "run_id":          run_id,
            "project_id":      project_id,
            "turn_index":      turn_index,
            "user_input":      turn.user_input,
            "llm_output":      turn.llm_output,
            "context":         result.get("context") or _normalize_context(turn.context),
            "expected_output": turn.expected_output or "",
            "raw_result":      json.dumps(result),
            "passed":          passed,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — called by evaluate.py
# ══════════════════════════════════════════════════════════════════════════════

async def run_opik(
    run_id:     str,
    project_id: str,
    config:     dict,
    payload:    EvaluationPayload,
) -> tuple:
    """
    Evaluates every turn using PwC Gemini, logs traces to the project-specific
    Opik project, and returns (dashboard_url, external_run_id).
    """
    metrics_to_run = resolve_metric_ids(
        "opik",
        getattr(payload, "selected_metrics", None),
        config.get("opik_metrics") or config.get("metrics"),
    )
    metric_thresholds = resolve_metric_thresholds(
        "opik",
        metrics_to_run,
        getattr(payload, "metric_thresholds", None),
    )
    session_label     = payload.session_label or f"Run {run_id[:8]}"
    opik_project_name = config.get("framework_project_name") or settings.OPIK_PROJECT_NAME
    opik_project_id   = config.get("framework_project_id")
    dashboard_url     = get_dashboard_url(opik_project_id) if opik_project_id else ""

    print(f"\n[OpikRunner] run_id={run_id}")
    print(f"[OpikRunner] turns={len(payload.conversation)}, metrics={metrics_to_run}")
    print(f"[OpikRunner] project='{opik_project_name}' id={opik_project_id}")
    print(f"[OpikRunner] dashboard={dashboard_url}")

    # ── 1. Init PwC client ────────────────────────────────────────────────────
    try:
        client = PwCGenAIClient()
    except ValueError as exc:
        raise RuntimeError(f"PwC GenAI configuration error: {exc}") from exc

    # ── 2. Build metrics ──────────────────────────────────────────────────────
    metric_objects = _build_metrics(metrics_to_run, client)
    if not metric_objects:
        raise RuntimeError(f"No valid metrics could be built from: {metrics_to_run}")

    # ── 3. Connect to Opik ────────────────────────────────────────────────────
    opik_client, opik_project_name = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _get_opik_client(opik_project_name)
    )

    # ── 4. Evaluate each turn ─────────────────────────────────────────────────
    first_trace_id = None
    all_results    = []

    for idx, turn in enumerate(payload.conversation):
        print(f"\n[OpikRunner] ── Turn {idx + 1}/{len(payload.conversation)} ──")

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda t=turn: _run_turn(t, metric_objects, metric_thresholds)
        )

        all_results.append(result)
        await _persist_turn_result(run_id, project_id, idx, turn, result)

        if opik_client:
            trace_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda t=turn, r=result: _log_trace_to_opik(
                    opik_client    = opik_client,
                    project_name   = opik_project_name,
                    run_id         = run_id,
                    session_label  = session_label,
                    turn_index     = idx,
                    turn           = t,
                    metric_results = r["metrics"],
                    overall_score  = r["overall_score"],
                ),
            )
            if trace_id and first_trace_id is None:
                first_trace_id = trace_id

        print(f"[OpikRunner] Turn {idx + 1} — overall={result.get('overall_score', 0):.4f}")

    print(f"\n[OpikRunner] ✅ Complete — dashboard: {dashboard_url}")
    return dashboard_url, (first_trace_id or run_id)
