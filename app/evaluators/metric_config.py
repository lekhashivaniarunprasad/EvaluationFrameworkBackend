import json
import re

DEFAULT_THRESHOLD = 0.5


def _label_to_id(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").lower()).strip("_")


def _metric(label: str, metric_id: str = None, aliases=None) -> dict:
    metric_id = metric_id or _label_to_id(label)
    base_aliases = {
        metric_id,
        metric_id.replace("_", " "),
        label,
        label.lower(),
    }
    if aliases:
        base_aliases.update(aliases)
    return {
        "id": metric_id,
        "label": label,
        "aliases": sorted(base_aliases),
    }


FRAMEWORK_METRICS = {
    "deepeval": [
        _metric("Faithfulness"),
        _metric("Answer Relevancy", aliases=["answer relevance"]),
        _metric("Contextual Relevancy", aliases=["context relevance"]),
        _metric("Contextual Precision", aliases=["context precision"]),
        _metric("Contextual Recall", aliases=["context recall"]),
        # _metric("RAGAS"),
        _metric("G-Eval", "g_eval", aliases=["geval"]),
        _metric("DAG Metric", "dag_metric", aliases=["dag"]),
        _metric("Hallucination"),
        _metric("Summarization"),
        _metric("Toxicity"),
        _metric("Bias"),
        _metric("Task Completion"),
        _metric("Tool Correctness"),
        _metric("Plan Quality"),
        _metric("Plan Adherence"),
        _metric("Knowledge Retention"),
        _metric("Conversation Completeness"),
        _metric("Role Adherence"),
        # _metric("Custom Metric"),
        _metric("Answer Correctness", aliases=["correctness"]),
        _metric("Data Security", aliases=["pii_leakage", "pii leakage", "privacy", "privacy leakage"]),
    ],
    "trulens": [
        _metric("Context Relevance"),
        _metric("Groundedness"),
        _metric("Answer Relevance"),
        _metric("Sentiment"),
        _metric("Language Match"),
        _metric("Toxicity"),
        _metric("Moderation"),
        _metric("Coherence"),
        _metric("Goal Alignment"),
        _metric("Plan Quality"),
        _metric("Action Correctness"),
        _metric("Logical Consistency"),
        _metric("Ground Truth Agreement"),
        _metric("Helpfulness"),
        _metric("Conciseness"),
        _metric("Stereotyping"),
        _metric("Comprehensiveness"),
        _metric("Answer Correctness", aliases=["correctness"]),
    ],
    "arize_phoenix": [
        _metric("Hallucination"),
        _metric("Faithfulness"),
        _metric("Q&A Correctness", "qa_correctness", aliases=["q&a correctness", "qa correctness"]),
        _metric("RAG Relevance"),
        _metric("Summarization"),
        _metric("Toxicity"),
        _metric("Correctness"),
        _metric("Conciseness"),
        _metric("Document Relevance"),
        _metric("Refusal Detection"),
        _metric("User Frustration"),
        _metric("SQL Generation"),
        _metric("Audio Emotion Detection"),
        _metric("Tool Selection"),
        _metric("Tool Invocation"),
        _metric("Tool Response Handling"),
        _metric("Function Calling Eval"),
        _metric("Path Convergence"),
        _metric("Agent Planning"),
        _metric("Agent Reflection"),
        _metric("Code Generation"),
        _metric("Exact Match"),
        _metric("Matches Regex", aliases=["regex match"]),
        _metric("Precision"),
        _metric("Recall"),
        _metric("F1"),
        _metric("Answer Relevance"),
        _metric("Answer Correctness", aliases=["correctness"]),
        _metric("Context Precision"),
        _metric("Context Recall"),
    ],
    "opik": [
        _metric("Exact Match"),
        _metric("Contains"),
        _metric("Regex Match", aliases=["matches regex"]),
        _metric("Sentence BLEU"),
        _metric("ROUGE-1", "rouge_1", aliases=["rouge1"]),
        _metric("ROUGE-2", "rouge_2", aliases=["rouge2"]),
        _metric("ROUGE-L", "rouge_l", aliases=["rougel"]),
        _metric("ROUGE-Lsum", "rouge_lsum", aliases=["rougelsum"]),
        _metric("Hallucination"),
        _metric("Answer Relevance"),
        _metric("Context Precision"),
        _metric("Context Recall"),
        _metric("Moderation"),
        _metric("Usefulness"),
        _metric("Meaning Match"),
        _metric("Summarization Consistency"),
        _metric("Summarization Coherence"),
        _metric("Dialogue Helpfulness"),
        _metric("Compliance Risk"),
        _metric("Prompt Uncertainty"),
        _metric("G-Eval", "g_eval", aliases=["geval"]),
        _metric("Trajectory Accuracy"),
        _metric("Task Completion"),
        _metric("Tool Correctness"),
        _metric("Conversational Thread Metrics"),
        _metric("Custom Conversation Metric"),
        # _metric("Custom Metric"),
        _metric("Answer Correctness", aliases=["correctness"]),
    ],
}


def normalize_metric_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _get_catalog(framework: str) -> list:
    return FRAMEWORK_METRICS.get(framework, [])


def _get_alias_map(framework: str) -> dict:
    alias_map = {}
    for metric in _get_catalog(framework):
        for alias in [metric["id"], metric["label"], *metric.get("aliases", [])]:
            alias_map[normalize_metric_name(alias)] = metric["id"]
    return alias_map


def get_metric_label(framework: str, metric_id: str) -> str:
    for metric in _get_catalog(framework):
        if metric["id"] == metric_id:
            return metric["label"]
    return metric_id


def resolve_metric_id(framework: str, metric_name: str):
    return _get_alias_map(framework).get(normalize_metric_name(metric_name))


def resolve_metric_ids(framework: str, payload_selected_metrics=None, config_raw=None) -> list:
    alias_map = _get_alias_map(framework)
    default_ids = [metric["id"] for metric in _get_catalog(framework)]

    source = payload_selected_metrics
    if not source:
        source = config_raw

    if isinstance(source, str):
        try:
            source = json.loads(source)
        except (json.JSONDecodeError, TypeError):
            source = [source]

    if not isinstance(source, list):
        return default_ids

    resolved = []
    seen = set()
    for item in source:
        metric_id = alias_map.get(normalize_metric_name(item))
        if metric_id and metric_id not in seen:
            seen.add(metric_id)
            resolved.append(metric_id)

    return resolved or default_ids


def resolve_metric_thresholds(framework: str, selected_metric_ids: list, payload_thresholds=None) -> dict:
    thresholds = {metric_id: DEFAULT_THRESHOLD for metric_id in selected_metric_ids}
    alias_map = _get_alias_map(framework)

    if not isinstance(payload_thresholds, dict):
        return thresholds

    for raw_key, raw_value in payload_thresholds.items():
        metric_id = alias_map.get(normalize_metric_name(raw_key))
        if not metric_id or metric_id not in thresholds:
            continue
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            continue
        thresholds[metric_id] = max(0.0, min(1.0, parsed))

    return thresholds
