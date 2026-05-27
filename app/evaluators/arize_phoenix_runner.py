"""
app/evaluators/arize_phoenix_runner.py

Arize AX evaluation runner.

Public API (called by projects.py):
    setup_project(project_id, project_name) -> dict
    get_dashboard_url(framework_project_id) -> str

Internal entry point (called by evaluate.py):
    run_arize_phoenix(run_id, project_id, config, payload) -> (dashboard_url, external_run_id)

Arize AX OTLP endpoint : https://otlp.arize.com/v1/traces
Auth headers           : space_id, api_key
Setup                  : arize.otel.register(space_id, api_key, project_name)
Projects               : created automatically on first trace
"""

import os
import re
import json
import uuid
import ssl
import asyncio
import traceback
import warnings

import urllib3
import requests
from dotenv import load_dotenv

load_dotenv()

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
    "hallucination",
    "faithfulness",
    "qa_correctness",
    "rag_relevance",
    "summarization",
    "toxicity",
    "correctness",
    "conciseness",
    "document_relevance",
    "refusal_detection",
    "user_frustration",
    "sql_generation",
    "audio_emotion_detection",
    "tool_selection",
    "tool_invocation",
    "tool_response_handling",
    "function_calling_eval",
    "path_convergence",
    "agent_planning",
    "agent_reflection",
    "code_generation",
    "exact_match",
    "matches_regex",
    "precision",
    "recall",
    "f1",
]

# ── Arize AX constants ────────────────────────────────────────────────────────
ARIZE_BASE_URL = "https://app.arize.com"


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
    Called by projects.py when a user selects arize_phoenix framework.

    Arize AX creates projects automatically on first trace — no explicit
    project creation API call needed. We generate a unique project name,
    store it in framework_projects, and return the dashboard URL.

    Returns:
        {
            "framework_project_id":   str,
            "framework_project_name": str,
            "dashboard_url":          str,
        }
    """
    if not settings.ARIZE_SPACE_ID or not settings.ARIZE_API_KEY:
        raise ValueError("ARIZE_SPACE_ID and ARIZE_API_KEY must be configured in settings.")

    short_id      = project_id[:8]
    arize_project = f"{project_name} ({short_id})"

    # Insert or update framework_projects table
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
                "framework_project_id":   arize_project,
                "framework_project_name": arize_project,
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
                "framework_project_id":   arize_project,
                "framework_project_name": arize_project,
            },
        )

    print(f"[ArizeRunner] ✅ framework_projects row saved: '{arize_project}'")

    return {
        "framework_project_id":   arize_project,
        "framework_project_name": arize_project,
        "dashboard_url":          get_dashboard_url(arize_project),
    }


def get_dashboard_url(framework_project_id: str) -> str:
    """
    Builds the Arize AX dashboard URL for a project.
    Format: https://app.arize.com/organizations/-/spaces/{space_id}/models
    Projects appear in the UI automatically after first trace is sent.
    """
    space_id = settings.ARIZE_SPACE_ID
    if space_id and framework_project_id:
        import urllib.parse
        encoded = urllib.parse.quote(framework_project_id)
        return f"{ARIZE_BASE_URL}/organizations/-/spaces/{space_id}/projects/{encoded}/traces"
    return ARIZE_BASE_URL


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE — Arize AX tracer setup using arize-otel
# ══════════════════════════════════════════════════════════════════════════════

def _setup_arize_tracer(project_name: str):
    """
    Sets up OpenTelemetry tracer using arize.otel.register().
    This is the official Arize AX setup pattern from their docs.
    Returns (tracer_provider, tracer) or (None, None) on failure.
    """
    try:
        from arize.otel import register

        tracer_provider = register(
            space_id     = settings.ARIZE_SPACE_ID,
            api_key      = settings.ARIZE_API_KEY,
            project_name = project_name,
        )

        tracer = tracer_provider.get_tracer(__name__)
        print(f"  [Arize] ✅ Tracer registered — project: '{project_name}'")
        return tracer_provider, tracer

    except Exception as e:
        print(f"  [Arize] ❌ Tracer setup failed: {e}")
        traceback.print_exc()
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
                "PWC_GENAI_API_KEY is not set in .env. "
                "Please add it and restart the server."
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
    "faithfulness": "Check whether the response is grounded in and supported by the retrieved context.",
    "qa_correctness": "Check whether the response accurately matches the expected or reference answer.",
    "rag_relevance": "Check whether retrieved documents are pertinent to the query and useful for generation.",
    "summarization": "Check completeness, accuracy, and non-contradiction of the summary against the source context.",
    "toxicity": "Detect hateful, threatening, discriminatory, abusive, or otherwise toxic content.",
    "correctness": "Check general factual and semantic correctness against the expected answer when provided.",
    "conciseness": "Check whether the response is appropriately brief without omitting important information.",
    "document_relevance": "Evaluate retrieved document relevance at the span or chunk level.",
    "refusal_detection": "Check whether the model appropriately refused unsafe or out-of-scope requests.",
    "user_frustration": "Detect signals that the user experience is unresolved, confusing, or frustrating.",
    "sql_generation": "Evaluate generated SQL for correctness, safety, and alignment with the user request.",
    "audio_emotion_detection": "Evaluate whether emotional tone inferred from transcript text is appropriate and supported.",
    "tool_selection": "Check whether the agent selected the correct tool for the task.",
    "tool_invocation": "Check whether tool/function calls use correct and complete arguments.",
    "tool_response_handling": "Check whether the agent correctly interpreted and acted on tool output.",
    "function_calling_eval": "Evaluate function call format, argument names, values, and completeness.",
    "path_convergence": "Check whether the agent reached the expected execution path or final state.",
    "agent_planning": "Evaluate the quality and feasibility of the agent's multi-step plan.",
    "agent_reflection": "Evaluate whether the agent self-corrected effectively after errors.",
    "code_generation": "Evaluate generated code for functional correctness, safety, and maintainability.",
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


class MatchesRegexMetric(BaseRAGMetric):
    name = "Matches Regex"

    def score(self, input: str, output: str, expected_output: str = "", context: list = []) -> dict:
        if not expected_output:
            return {"score": 0.0, "reason": "Expected output must contain the regex pattern."}
        try:
            passed = re.search(expected_output, output) is not None
            return {"score": 1.0 if passed else 0.0, "reason": "Output matches regex pattern." if passed else "Output does not match regex pattern."}
        except re.error as e:
            return {"score": 0.0, "reason": f"Invalid regex pattern: {e}"}


class TokenOverlapMetric(BaseRAGMetric):
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
            return {"score": 0.0, "reason": "Expected output or context is required for token overlap metrics."}
        out_tokens = self._tokens(output)
        ref_tokens = self._tokens(reference)
        if not out_tokens or not ref_tokens:
            return {"score": 0.0, "reason": "Output or reference has no comparable tokens."}
        out_set = set(out_tokens)
        ref_set = set(ref_tokens)
        precision = len(out_set & ref_set) / len(out_set)
        recall = len(out_set & ref_set) / len(ref_set)
        score = precision if self.metric_id == "precision" else recall if self.metric_id == "recall" else 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {"score": round(score, 4), "reason": f"Token overlap precision={precision:.4f}, recall={recall:.4f}."}


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
    "matchesregex":     MatchesRegexMetric,
    "precision":        TokenOverlapMetric,
    "recall":           TokenOverlapMetric,
    "f1":               TokenOverlapMetric,
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
        metric_id = resolve_metric_id("arize_phoenix", name) or name
        if cls:
            try:
                if cls is TokenOverlapMetric:
                    metrics.append(cls(
                        client=client,
                        metric_id=metric_id,
                        name=get_metric_label("arize_phoenix", metric_id),
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
                    name=get_metric_label("arize_phoenix", metric_id),
                    criteria=criteria,
                ))
                print(f"  [Metrics] built {name} with GenericLLMJudgeMetric")
            else:
                print(f"  [Metrics] unknown metric: '{name}'")
    return metrics


def _resolve_metrics(config: dict) -> list:
    raw = config.get("arize_metrics") or config.get("metrics")
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
# ARIZE AX TRACE LOGGER
# Uses OpenInference semantic conventions for best Arize AX integration
# ══════════════════════════════════════════════════════════════════════════════

def _log_trace_to_arize(
    tracer,
    project_name: str,
    run_id: str,
    session_label: str,
    turn_index: int,
    turn,
    metric_results: list,
    overall_score: float,
) -> str | None:
    """
    Logs a single conversation turn as an OpenTelemetry span to Arize AX.
    Uses OpenInference semantic conventions for span kind and attributes.
    """
    trace_id = None

    try:
        from opentelemetry.trace import StatusCode

        trace_name = f"{session_label} | Turn {turn_index + 1}"

        with tracer.start_as_current_span(trace_name) as span:

            # ── OpenInference span kind — CHAIN for RAG pipeline ──────────────
            span.set_attribute("openinference.span.kind", "CHAIN")

            # ── Core input / output ───────────────────────────────────────────
            span.set_attribute("input.value",     turn.user_input)
            span.set_attribute("output.value",    turn.llm_output)

            # ── Session / metadata ────────────────────────────────────────────
            span.set_attribute("session.id",      run_id)
            span.set_attribute("tag.session_label", session_label)
            span.set_attribute("tag.turn_index",    turn_index)
            span.set_attribute("tag.project_name",  project_name)
            span.set_attribute("tag.framework",     "arize_phoenix")

            # ── Expected output ───────────────────────────────────────────────
            if turn.expected_output:
                span.set_attribute("tag.expected_output", turn.expected_output)

            # ── Retrieved context chunks ──────────────────────────────────────
            for i, ctx in enumerate(_normalize_context(turn.context)):
                span.set_attribute(f"retrieval.documents.{i}.document.content", ctx)

            # ── Overall eval score ────────────────────────────────────────────
            span.set_attribute("eval.overall_score",  overall_score)
            span.set_attribute("eval.overall_passed",  overall_score >= 0.5)

            # ── Per-metric scores ─────────────────────────────────────────────
            for metric in metric_results:
                safe = metric["name"].lower().replace(" ", "_")
                span.set_attribute(f"eval.{safe}.score",  metric["score"])
                span.set_attribute(f"eval.{safe}.passed", metric["passed"])
                span.set_attribute(f"eval.{safe}.reason", metric.get("reason", ""))

            span.set_status(StatusCode.OK)

            ctx    = span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")

        print(f"  [Arize] ✅ Span logged: '{trace_name}' trace_id={trace_id[:16]}...")
        return trace_id

    except Exception as e:
        print(f"  [Arize] ❌ Span logging failed: {e}")
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TURN RUNNER + DB PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _run_turn(turn, metrics: list, metric_thresholds: dict) -> dict:
    results = []
    contexts = _normalize_context(turn.context)
    _print_retrieved_context("ArizeRunner", turn.user_input, contexts)
    for metric in metrics:
        print(f"\n  [Eval] Scoring: {metric.name}...")
        metric_id = resolve_metric_id("arize_phoenix", metric.name) or metric.name
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

async def run_arize_phoenix(
    run_id:     str,
    project_id: str,
    config:     dict,
    payload:    EvaluationPayload,
) -> tuple:
    """
    Evaluates every turn using PwC Gemini, logs spans to Arize AX,
    and returns (dashboard_url, external_run_id).
    """
    metrics_to_run = resolve_metric_ids(
        "arize_phoenix",
        getattr(payload, "selected_metrics", None),
        config.get("arize_metrics") or config.get("metrics"),
    )
    metric_thresholds = resolve_metric_thresholds(
        "arize_phoenix",
        metrics_to_run,
        getattr(payload, "metric_thresholds", None),
    )
    session_label  = payload.session_label or f"Run {run_id[:8]}"
    arize_project  = config.get("framework_project_name") or "Default Project"
    proj_id        = config.get("framework_project_id") or arize_project
    dashboard_url  = get_dashboard_url(proj_id)

    print(f"\n[ArizeRunner] run_id={run_id}")
    print(f"[ArizeRunner] turns={len(payload.conversation)}, metrics={metrics_to_run}")
    print(f"[ArizeRunner] project='{arize_project}'")
    print(f"[ArizeRunner] dashboard={dashboard_url}")

    # ── 1. Init PwC client ────────────────────────────────────────────────────
    try:
        client = PwCGenAIClient()
    except ValueError as exc:
        raise RuntimeError(f"PwC GenAI configuration error: {exc}") from exc

    # ── 2. Build metrics ──────────────────────────────────────────────────────
    metric_objects = _build_metrics(metrics_to_run, client)
    if not metric_objects:
        raise RuntimeError(f"No valid metrics could be built from: {metrics_to_run}")

    # ── 3. Set up Arize AX tracer using arize.otel.register() ────────────────
    tracer_provider, tracer = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _setup_arize_tracer(arize_project)
    )

    # ── 4. Evaluate each turn ─────────────────────────────────────────────────
    first_trace_id = None

    for idx, turn in enumerate(payload.conversation):
        print(f"\n[ArizeRunner] ── Turn {idx + 1}/{len(payload.conversation)} ──")

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda t=turn: _run_turn(t, metric_objects, metric_thresholds)
        )

        await _persist_turn_result(run_id, project_id, idx, turn, result)

        if tracer:
            trace_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda t=turn, r=result: _log_trace_to_arize(
                    tracer         = tracer,
                    project_name   = arize_project,
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

        print(f"[ArizeRunner] Turn {idx + 1} — overall={result.get('overall_score', 0):.4f}")

    # ── 5. Shutdown tracer — flushes all pending spans to Arize AX ───────────
    if tracer_provider:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, tracer_provider.shutdown
            )
            print("[ArizeRunner] ✅ Tracer shut down — all spans flushed to Arize AX")
        except Exception as e:
            print(f"[ArizeRunner] ⚠️  Shutdown warning: {e}")

    print(f"\n[ArizeRunner] ✅ Complete — dashboard: {dashboard_url}")
    return dashboard_url, (first_trace_id or run_id)
