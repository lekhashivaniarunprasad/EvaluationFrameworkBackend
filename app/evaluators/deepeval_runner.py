"""
app/evaluators/deepeval_runner.py
----------------------------------
DeepEval RAG Evaluation Engine.
Called by app/routers/evaluate.py as a background task.

Entry point signature (matches evaluate.py):
    run_deepeval(run_id, project_id, config, payload) -> (dashboard_url, external_run_id)

Payload structure (from evaluate.py EvaluationPayload):
    payload.conversation = List[ConversationTurn]
    Each ConversationTurn:
        user_input      : str           → input / question
        llm_output      : str           → actual_output / chatbot response
        context         : List[str]     → retrieval_context
        expected_output : Optional[str] → expected_output (required for deepeval)

config = project row from DB (dict), contains:
    config["id"]          → project_id
    config["framework"]   → "deepeval"
    config["api_key"]     → short_id (unique endpoint)
    config["llm_model"]   → optional model info

Results are stored in evaluation_runs table via database.execute().

Environment variables required:
    VERTEX_API_KEY=your_key
    VERTEX_API_BASE=https://your-proxy-url.com
"""

from __future__ import annotations

import json
import re
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch
import os
import traceback


import asyncio
from concurrent.futures import ThreadPoolExecutor

import httpx

# Keep DeepEval from auto-opening Confident AI in a browser after evaluation.
os.environ.setdefault('CONFIDENT_OPEN_BROWSER', '0')

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    PIILeakageMetric,
    SummarizationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

from app.core.config import database, settings
from app.evaluators.metric_config import (
    get_metric_label,
    resolve_metric_ids,
    resolve_metric_thresholds,
)

# ── Confident AI — set API key using DeepEval's KeyFileHandler ───────────────
# DeepEval stores the key via KeyFileHandler.write_key(KeyValues.CONFIDENT_API_KEY)
# Secrets are deprecated in the file — preferred way is via environment variable
_confident_api_key = os.getenv("CONFIDENT_AI_API_KEY")
if _confident_api_key:
    try:
        from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
        # Set in environment (preferred by DeepEval)
        os.environ["CONFIDENT_API_KEY"] = _confident_api_key
        # Also write via KeyFileHandler for CLI compatibility
        KEY_FILE_HANDLER.write_key(KeyValues.CONFIDENT_API_KEY, _confident_api_key)
        print(f"[DeepEval] Confident AI API key set successfully")
    except Exception as _e:
        print(f"[DeepEval] Confident AI login warning: {_e}")
else:
    print("[DeepEval] Warning: CONFIDENT_AI_API_KEY not set in environment — dashboard URL will be null")


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Proxy Model
# ─────────────────────────────────────────────────────────────────────────────

class VertexProxyModel(DeepEvalBaseLLM):
    """
    Calls the Vertex AI proxy directly. No OpenAI SDK needed.

    Headers: accept, API-Key, Authorization: Bearer, Content-Type
    Body:    model, prompt, temperature=0.0, seed=42
    """

    def __init__(self):
        self.api_key  = os.getenv("VERTEX_API_KEY")
        self.api_base = os.getenv("VERTEX_API_BASE")
        if not self.api_key:
            raise ValueError("VERTEX_API_KEY not set in environment")
        if not self.api_base:
            raise ValueError("VERTEX_API_BASE not set in environment")

    def get_model_name(self) -> str:
        return "vertex_ai.gemini-2.0-flash"

    def load_model(self):
        return self

    def _call_proxy(self, prompt: str) -> str:
        response = httpx.post(
            url     = self.api_base,
            headers = {
                "accept"        : "application/json",
                "API-Key"       : self.api_key,
                "Authorization" : f"Bearer {self.api_key}",
                "Content-Type"  : "application/json",
            },
            json = {
                "model"            : "vertex_ai.gemini-2.0-flash",
                "prompt"           : prompt,
                "temperature"      : 0.0,
                "top_p"            : 1,
                "presence_penalty" : 0,
                "seed"             : 42,
                "stream"           : False,
                "max_tokens"       : 2048,
            },
            timeout = settings.PWC_GENAI_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, data: dict) -> str:
        if "choices" in data:
            choice = data["choices"][0]
            if "message" in choice: return choice["message"]["content"]
            if "text"    in choice: return choice["text"]
        if "text"     in data: return data["text"]
        if "content"  in data: return data["content"]
        if "response" in data: return data["response"]
        if "output"   in data: return data["output"]
        return str(data)

    @staticmethod
    def _strip_markdown_fence(raw: str) -> str:
        cleaned = str(raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
            cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    @staticmethod
    def _schema_required_keys(schema) -> set:
        schema_json = schema.model_json_schema() if hasattr(schema, "model_json_schema") else {}
        return set(schema_json.get("required", []))

    @staticmethod
    def _matches_schema(data: Any, schema) -> bool:
        required = VertexProxyModel._schema_required_keys(schema)
        if not required:
            return True
        return isinstance(data, dict) and required.issubset(data.keys())

    @staticmethod
    def _extract_json_object(raw: str, schema=None) -> str | None:
        cleaned = VertexProxyModel._strip_markdown_fence(raw)
        try:
            parsed = json.loads(cleaned)
            if VertexProxyModel._matches_schema(parsed, schema):
                return json.dumps(parsed)
        except Exception:
            pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\{\[]", cleaned):
            try:
                parsed, _ = decoder.raw_decode(cleaned[match.start():])
                if VertexProxyModel._matches_schema(parsed, schema):
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _fallback_json_for_schema(raw: str, schema) -> str:
        schema_json = schema.model_json_schema() if hasattr(schema, "model_json_schema") else {}
        properties = schema_json.get("properties", {})
        required = schema_json.get("required", [])
        raw_text = str(raw or "").strip()

        if "steps" in required or "steps" in properties:
            lines = [
                line.strip(" -0123456789.)")
                for line in raw_text.splitlines()
                if line.strip()
            ]
            steps = [line for line in lines if line][:5] or [
                "Assess the response against the metric criteria.",
                "Compare the response with the input, context, and expected output.",
                "Return a score with a concise reason.",
            ]
            return json.dumps({"steps": steps})

        if "truths" in required or "truths" in properties:
            return json.dumps({"truths": [raw_text[:500]] if raw_text else []})

        if "claims" in required or "claims" in properties:
            return json.dumps({"claims": [raw_text[:500]] if raw_text else []})

        if "verdicts" in required or "verdicts" in properties:
            return json.dumps({
                "verdicts": [{
                    "verdict": "idk",
                    "reason": "The model did not return verdicts in the expected JSON format.",
                }]
            })

        if "score" in required or "score" in properties:
            match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw_text)
            score = float(match.group(0)) if match else 0.0
            return json.dumps({
                "score": max(0.0, min(1.0, score)),
                "reason": raw_text[:500] or "Model did not return a parseable reason.",
            })

        fallback = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type")
            if field_type == "array":
                fallback[field_name] = []
            elif field_type in {"number", "integer"}:
                fallback[field_name] = 0
            elif field_type == "boolean":
                fallback[field_name] = False
            else:
                fallback[field_name] = raw_text[:500] if field_name == "reason" else ""
        return json.dumps(fallback or {"reason": raw_text[:500]})

    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            prompt = (
                "CRITICAL INSTRUCTION: Your response must be a valid JSON object ONLY.\n"
                "Do NOT include ```json``` or any markdown formatting.\n"
                "Do NOT include any explanation or extra text before or after.\n"
                "Your entire response must start with { and end with }.\n"
                "Example of correct format: {\"key\": \"value\"}\n\n"
                + prompt
            )
            # Retry up to 3 times if response is not valid JSON
            last_error = None
            last_raw = ""
            for attempt in range(3):
                try:
                    raw = self._call_proxy(prompt)
                    last_raw = raw
                    cleaned = self._extract_json_object(raw, schema)
                    if cleaned is None:
                        raise ValueError(f"No valid JSON found in model response: {raw[:300]}")
                    return cleaned
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        prompt = (
                            "YOUR PREVIOUS RESPONSE WAS NOT VALID JSON. TRY AGAIN.\n"
                            "CRITICAL INSTRUCTION: Your response must be a valid JSON object ONLY.\n"
                            "Do NOT include ```json``` or any markdown formatting.\n"
                            "Your entire response must start with { and end with }.\n\n"
                            + prompt
                        )
            print(f"[DeepEval] Warning: model did not return valid JSON; using schema fallback. Last error: {last_error}")
            return self._fallback_json_for_schema(last_raw, schema)
        return self._call_proxy(prompt)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)


# ── Initialise once — reused across all metrics and all turns ─────────────────
llm_model = VertexProxyModel()

DEFAULT_METRIC_IDS = [
    "faithfulness",
    "answer_relevancy",
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
    # "ragas",
    "g_eval",
    "dag_metric",
    "hallucination",
    "summarization",
    "toxicity",
    "bias",
    "task_completion",
    "tool_correctness",
    "plan_quality",
    "plan_adherence",
    "knowledge_retention",
    "conversation_completeness",
    "role_adherence",
    "custom_metric",
    "answer_correctness",
    "data_security",
]

METRIC_CLASS_MAP = {
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
    "data_security": PIILeakageMetric,
    "hallucination": HallucinationMetric,
    "summarization": SummarizationMetric,
    "toxicity": ToxicityMetric,
    "bias": BiasMetric,
}

CONTEXT_REQUIRED_METRIC_IDS = {
    "faithfulness",
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
    # "ragas",
    "hallucination",
}


DEEPEVAL_GEVAL_CRITERIA = {
    # "ragas": "Produce a composite RAG score averaging answer relevancy, faithfulness, contextual precision, and contextual recall from the available fields.",
    "g_eval": "Apply a general LLM-as-judge quality rubric using the input, context, expected output, and actual output.",
    "dag_metric": "Approximate deterministic DAG evaluation by checking clear decision criteria in sequence and scoring the final outcome.",
    "task_completion": "Evaluate whether the agent or model successfully completed the user's assigned task.",
    "tool_correctness": "Check whether the correct tools were selected and invoked. If tool-call data is absent, score 0 and explain what is missing.",
    "plan_quality": "Evaluate the quality, feasibility, and completeness of the plan or reasoning.",
    "plan_adherence": "Check whether the response follows its plan. If no explicit plan is present, score based on observable adherence and explain the limitation.",
    "knowledge_retention": "Check whether the chatbot retains relevant facts across the available conversation context.",
    "conversation_completeness": "Evaluate whether the conversation response satisfies the user's need completely.",
    "role_adherence": "Check whether the model stays within the assigned role or persona. If no role is specified, score based on the visible prompt constraints.",
    "custom_metric": "Apply a general custom quality rubric to the available evaluation fields.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal metric runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_metric(metric_instance, test_case, metric_name: str, threshold: float) -> Dict[str, Any]:
    """Run a single metric and return a result dict."""
    try:
        metric_instance.measure(test_case)
        score = getattr(metric_instance, "score", None)
        return {
            "metric"    : metric_name,
            "name"      : get_metric_label("deepeval", metric_name),
            "score"     : round(score, 4) if score is not None else None,
            "passed"    : getattr(metric_instance, "is_successful", lambda: None)(),
            "reason"    : getattr(metric_instance, "reason", None),
            "threshold" : threshold,
            "error"     : None,
        }
    except Exception as exc:
        return {
            "metric"    : metric_name,
            "name"      : get_metric_label("deepeval", metric_name),
            "score"     : None,
            "passed"    : None,
            "reason"    : None,
            "threshold" : threshold,
            "error"     : f"{type(exc).__name__}: {exc}",
        }


def _build_turn_result(
    test_case       : LLMTestCase,
    metrics_results : list,
    dashboard_url   : Optional[str],
) -> Dict[str, Any]:
    """
    Build a turn result dict from an already-measured LLMTestCase + metrics.
    Called after batch deepeval_evaluate() completes.
    """
    scores         = [m["score"] for m in metrics_results if m["score"] is not None]
    average_score  = round(sum(scores) / len(scores), 4) if scores else None
    overall_passed = all(m["passed"] is True for m in metrics_results) if metrics_results else None

    return {
        "input"          : test_case.input,
        "actual_output"  : test_case.actual_output,
        "context"        : test_case.retrieval_context,
        "expected_output": test_case.expected_output,
        "overall_passed" : overall_passed,
        "average_score"  : average_score,
        "metrics_passed" : sum(1 for m in metrics_results if m["passed"] is True),
        "metrics_failed" : sum(1 for m in metrics_results if m["passed"] is False),
        "metrics"        : metrics_results,
        "dashboard_url"  : dashboard_url,
}


def _normalize_context(context: Any) -> List[str]:
    if context is None:
        return []
    if isinstance(context, str):
        return [context] if context.strip() else []
    if isinstance(context, list):
        return [str(item).strip() for item in context if str(item).strip()]
    return [str(context).strip()] if str(context).strip() else []


def _make_metric(metric_id: str, threshold: float):
    if metric_id == "answer_correctness":
        return GEval(
            name="Answer Correctness",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Compare the actual output against the expected output for the same input.",
                "Score 1 only when the actual output is semantically equivalent to the expected output.",
                "Score 0 when the actual output contradicts the expected output, gives a different factual answer, or misses the expected answer.",
                "Do not reward an answer only because it is relevant to the question; correctness against expected output is required.",
            ],
            threshold=threshold,
            model=llm_model,
            async_mode=False,
        )

    if metric_id in DEEPEVAL_GEVAL_CRITERIA:
        return GEval(
            name=get_metric_label("deepeval", metric_id),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            criteria=DEEPEVAL_GEVAL_CRITERIA[metric_id],
            threshold=threshold,
            model=llm_model,
            async_mode=False,
        )

    metric_cls = METRIC_CLASS_MAP.get(metric_id)
    if not metric_cls:
        return None
    return metric_cls(
        threshold=threshold,
        model=llm_model,
        include_reason=True,
        async_mode=False,
    )


def _run_all_turns(
    conversation : list,
    selected_metric_ids: list,
    metric_thresholds: dict,
    run_label    : str,
) -> tuple[list, Optional[str]]:
    """
    Builds ALL test cases, calls deepeval_evaluate() ONCE for all turns together.
    This means all turns share a single Confident AI dashboard URL.
    Runs synchronously inside a ThreadPoolExecutor (avoids uvloop patch error).
    """
    if not selected_metric_ids:
        raise ValueError("At least one DeepEval metric must be selected.")

    # ── Build one LLMTestCase per turn. DeepEval contextual metrics read
    # retrieval_context directly from this test case.
    test_cases = []
    for turn in conversation:
        retrieval_context = _normalize_context(getattr(turn, "context", None))
        print(
            f"\n[DeepEval] Retrieved context for evaluation input: {turn.user_input}",
            flush=True,
        )
        if retrieval_context:
            for context_index, context_chunk in enumerate(retrieval_context, start=1):
                print(
                    f"[DeepEval] Context chunk {context_index}:\n{context_chunk}\n",
                    flush=True,
                )
        else:
            print("[DeepEval] No context provided for this turn.\n", flush=True)

        if CONTEXT_REQUIRED_METRIC_IDS.intersection(selected_metric_ids) and not retrieval_context:
            print(
                "[DeepEval] Warning: contextual metrics selected but no context "
                f"was provided for input: {turn.user_input[:120]}",
                flush=True,
            )

        tc = LLMTestCase(
            input             = turn.user_input,
            actual_output     = turn.llm_output,
            expected_output   = turn.expected_output or "",
            context           = retrieval_context,
            retrieval_context = retrieval_context,
        )
        test_cases.append(tc)

    all_metrics = [
        _make_metric(metric_id, metric_thresholds.get(metric_id, 0.5))
        for metric_id in selected_metric_ids
    ]
    all_metrics = [metric for metric in all_metrics if metric is not None]

    # ── Single deepeval_evaluate() call → single Confident AI URL ─────────────
    eval_result = None
    try:
        with patch.object(webbrowser, "open", return_value=False), \
             patch.object(webbrowser, "open_new", return_value=False), \
             patch.object(webbrowser, "open_new_tab", return_value=False):
            eval_result = deepeval_evaluate(
                test_cases    = test_cases,
                metrics       = all_metrics,
                identifier    = run_label,
                async_config  = AsyncConfig(run_async=False),
                display_config= DisplayConfig(show_indicator=False, print_results=False),
            )
    except Exception as exc:
        print(
            "[DeepEval] Warning: dashboard batch evaluation failed; "
            f"continuing with local metric results. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
    # Get dashboard URL — try confident_link first, then LAST_TEST_RUN_LINK from key handler
    dashboard_url = getattr(eval_result, "confident_link", None) if eval_result else None

    if not dashboard_url:
        # Try LAST_TEST_RUN_LINK — DeepEval stores the last run URL here after push
        try:
            from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
            dashboard_url = KEY_FILE_HANDLER.fetch_data(KeyValues.LAST_TEST_RUN_LINK)
            if dashboard_url:
                print(f"[DeepEval] Got dashboard URL from LAST_TEST_RUN_LINK: {dashboard_url}")
        except Exception:
            pass

    if not dashboard_url:
        # Build from test_run_id as final fallback
        test_run_id = getattr(eval_result, "test_run_id", None) if eval_result else None
        if test_run_id:
            dashboard_url = f"https://app.confident-ai.com/test-runs/{test_run_id}"
            print(f"[DeepEval] Built dashboard URL from test_run_id: {dashboard_url}")
        else:
            print(f"[DeepEval] Warning: No dashboard URL found — check CONFIDENT_AI_API_KEY in environment")

    # ── Measure each turn explicitly using that turn's retrieval_context.
    # DeepEval metric objects store score/reason state, so each metric instance
    # must be fresh for each test case.
    turn_results = []
    for i, tc in enumerate(test_cases):
        metrics_results = []
        for metric_id in selected_metric_ids:
            metric = _make_metric(metric_id, metric_thresholds.get(metric_id, 0.5))
            if metric is None:
                continue
            metrics_results.append(
                _run_metric(
                    metric_instance = metric,
                    test_case       = tc,
                    metric_name     = metric_id,
                    threshold       = metric_thresholds.get(metric_id, 0.5),
                )
            )

        turn_result = _build_turn_result(
            test_case     = tc,
            metrics_results = metrics_results,
            dashboard_url = dashboard_url,   # same URL for all turns
        )
        turn_result["turn_index"] = i + 1
        turn_results.append(turn_result)

    return turn_results, dashboard_url


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — called by evaluate.py background task
# ─────────────────────────────────────────────────────────────────────────────

async def run_deepeval(
    run_id     : str,
    project_id : str,
    config     : dict,
    payload,                  # EvaluationPayload from evaluate.py
) -> Tuple[str, str]:
    """
    Called by evaluate.py's run_evaluation_background().

    Iterates over payload.conversation (List[ConversationTurn]),
    runs all 5 DeepEval metrics on each turn, saves results to DB,
    and returns (dashboard_url, external_run_id).

    Parameters
    ----------
    run_id     : evaluation_runs.id (UUID)
    project_id : projects.id (UUID)
    config     : full project row as dict — contains api_key, framework, etc.
    payload    : EvaluationPayload with .conversation list

    Returns
    -------
    dashboard_url    : str — URL to view results (internal status endpoint)
    external_run_id  : str — same as run_id for internal runs
    """
    turn_results = []
    selected_metric_ids = resolve_metric_ids(
        "deepeval",
        getattr(payload, "selected_metrics", None),
        DEFAULT_METRIC_IDS,
    )
    metric_thresholds = resolve_metric_thresholds(
        "deepeval",
        selected_metric_ids,
        getattr(payload, "metric_thresholds", None),
    )

    # Append run_id suffix so every run appears separately on Confident AI
    # Even if session_label is reused, each run gets a unique identifier
    session_label = getattr(payload, "session_label", None) or "Eval Run"
    run_label     = f"{session_label} | {run_id[:8]}"

    # ── Run ALL turns in one batch — single Confident AI dashboard URL ─────────
    # DeepEval can't patch uvloop so we run in a thread executor
    loop     = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()

    turn_results, confident_dashboard_url = await loop.run_in_executor(
        executor,
        lambda: _run_all_turns(
            conversation = payload.conversation,
            selected_metric_ids = selected_metric_ids,
            metric_thresholds = metric_thresholds,
            run_label    = run_label,
        ),
    )

    # ── Aggregate across all turns ─────────────────────────────────────────────
    all_scores   = [t["average_score"] for t in turn_results if t["average_score"] is not None]
    overall_avg  = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    turns_passed = sum(1 for t in turn_results if t["overall_passed"] is True)
    turns_failed = sum(1 for t in turn_results if t["overall_passed"] is False)

    final_result = {
        "framework"             : "deepeval",
        "project_id"            : project_id,
        "run_id"                : run_id,
        "endpoint_id"           : config.get("api_key"),
        "total_turns"           : len(turn_results),
        "turns_passed"          : turns_passed,
        "turns_failed"          : turns_failed,
        "overall_average_score" : overall_avg,
        "overall_passed"        : turns_failed == 0 if turn_results else None,
        "selected_metrics"      : selected_metric_ids,
        "metric_thresholds"     : metric_thresholds,
        "confident_dashboard_url": confident_dashboard_url,
        "turn_results"          : turn_results,
    }

    # ── Save summary to evaluation_runs (including dashboard URL) ─────────────
    await database.execute(
        """
        UPDATE evaluation_runs
        SET results      = :results,
            score        = :score,
            dashboard_url = :dashboard_url
        WHERE id = :id
        """,
        {
            "results"       : json.dumps(final_result),
            "score"         : overall_avg,
            "dashboard_url" : confident_dashboard_url,
            "id"            : run_id,
        },
    )

    # ── Save one row per turn into evaluation_results ─────────────────────────
    # Columns: id, run_id, project_id, turn_index, user_input, llm_output,
    #          context (text[]), expected_output, raw_result (jsonb), passed, created_at
    import uuid as _uuid
    from datetime import datetime, timezone

    for turn in turn_results:
        await database.execute(
            """
            INSERT INTO evaluation_results
                (id, run_id, project_id, turn_index, user_input, llm_output,
                 context, expected_output, raw_result, passed, created_at)
            VALUES
                (:id, :run_id, :project_id, :turn_index, :user_input, :llm_output,
                 :context, :expected_output, :raw_result, :passed, :created_at)
            """,
            {
                "id"              : str(_uuid.uuid4()),
                "run_id"          : run_id,
                "project_id"      : project_id,
                "turn_index"      : turn["turn_index"],
                "user_input"      : turn["input"],
                "llm_output"      : turn["actual_output"],
                "context"         : turn.get("context", []),
                "expected_output" : turn.get("expected_output"),
                "raw_result"      : json.dumps(turn),
                "passed"          : turn["overall_passed"],
                "created_at"      : datetime.now(timezone.utc),
            },
        )

    # Return Confident AI URL if available, otherwise fall back to internal status URL
    returned_dashboard_url = confident_dashboard_url or f"https://etlab-projects.pwc.in/evalforge-be/evaluate/run/{run_id}/status"
    external_run_id        = run_id

    return returned_dashboard_url, external_run_id
