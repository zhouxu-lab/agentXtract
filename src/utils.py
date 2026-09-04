"""Utility functions: API wrappers, config loading, retry logic, cost tracking, checkpoint DB."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json as _json
import logging
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
import openai
import yaml
from dotenv import load_dotenv
from google import genai

from src.schema import CostEntry, PaperCheckpoint, PipelineStatus

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -- Config loading ----------------------------------------------------------

def load_config(*yaml_paths: str | Path) -> dict:
    """Load and merge multiple YAML config files. Later files override earlier ones."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    merged: dict = {}
    for p in yaml_paths:
        path = Path(p)
        if not path.is_absolute() and not path.exists():
            path = PROJECT_ROOT / path
        if not path.exists():
            continue
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def write_json_atomic(path: str | Path, value: object) -> None:
    """Write JSON beside its destination and atomically replace on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        _json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


# -- Pricing / cost ----------------------------------------------------------

def _get_pricing(config: dict, model: str) -> dict:
    pricing = config.get("pricing", {})
    return pricing.get(model, {"input_per_mtok": 0.0, "output_per_mtok": 0.0})


def compute_cost(model: str, input_tokens: int, output_tokens: int, config: dict) -> float:
    p = _get_pricing(config, model)
    return (input_tokens * p["input_per_mtok"] + output_tokens * p["output_per_mtok"]) / 1_000_000


# -- Anthropic API wrapper ---------------------------------------------------

async def call_anthropic(
    prompt: str,
    system: str = "",
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    images: list[dict] | None = None,
    config: dict | None = None,
) -> tuple[str, CostEntry]:
    """Call Anthropic Messages API (async, non-streaming). Returns (response_text, cost_entry)."""
    client = anthropic.AsyncAnthropic()

    content: list[dict] = []
    if images:
        for img in images:
            img_path = Path(img["path"])
            media_type = img.get("media_type", "image/png")
            image_bytes = await asyncio.to_thread(img_path.read_bytes)
            b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })
    content.append({"type": "text", "text": prompt})

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    # Anthropic SDK 1.3 removed ``temperature`` from the realtime Messages
    # signature. Retain compatibility with older SDKs without turning a local
    # argument mismatch into five pointless API retries.
    try:
        supports_temperature = "temperature" in inspect.signature(
            client.messages.stream
        ).parameters
    except (TypeError, ValueError):
        supports_temperature = False
    if supports_temperature:
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system

    logger.debug(f"Calling Anthropic {model}, prompt length ~{len(prompt)} chars")
    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            async with client.messages.stream(**kwargs) as stream:
                response = await stream.get_final_message()
            break
        except anthropic.RateLimitError as e:
            # Respect Retry-After header if present; otherwise use exponential backoff
            retry_after = 30
            if hasattr(e, "response") and e.response is not None:
                retry_after = int(e.response.headers.get("retry-after", 30))
            wait = min(retry_after, 120)
            logger.warning(f"  Rate limit (attempt {attempt + 1}/{max_retries + 1}) — waiting {wait}s")
            await asyncio.sleep(wait)
            if attempt == max_retries:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code in (500, 529) and attempt < max_retries:
                wait = 2 ** (attempt + 1)
                logger.warning(f"  Anthropic server error {e.status_code} (attempt {attempt + 1}/{max_retries + 1}) — retrying in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"  API error (attempt {attempt + 1}/{max_retries + 1}): {e} — retrying in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise
    text = response.content[0].text
    usage = response.usage
    cfg = config or {}
    cost = compute_cost(model, usage.input_tokens, usage.output_tokens, cfg)

    cost_entry = CostEntry(
        stage="",
        model=model,
        doi="",
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
    )
    logger.debug(f"Anthropic response: {usage.input_tokens} in, {usage.output_tokens} out, ${cost:.4f}")
    return text, cost_entry


# -- OpenAI API wrapper ------------------------------------------------------

async def call_openai(
    prompt: str,
    system: str = "",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    config: dict | None = None,
) -> tuple[str, CostEntry]:
    """Call OpenAI Chat Completions API (async). Returns (response_text, cost_entry)."""
    client = openai.AsyncOpenAI()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    logger.debug(f"Calling OpenAI {model}, prompt length ~{len(prompt)} chars")
    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            # GPT-5+ models require max_completion_tokens and don't support temperature
            is_gpt5 = model.startswith("gpt-5")
            extra = {"max_completion_tokens": max_tokens} if is_gpt5 else {"max_tokens": max_tokens, "temperature": temperature}
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **extra,
            )
            break
        except openai.RateLimitError as e:
            retry_after = 30
            if hasattr(e, "response") and e.response is not None:
                retry_after = int(e.response.headers.get("retry-after", 30))
            wait = min(retry_after, 120)
            logger.warning(f"  OpenAI rate limit (attempt {attempt + 1}/{max_retries + 1}) — waiting {wait}s")
            await asyncio.sleep(wait)
            if attempt == max_retries:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"  OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e} — retrying in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise

    text = response.choices[0].message.content
    usage = response.usage
    cfg = config or {}
    cost = compute_cost(model, usage.prompt_tokens, usage.completion_tokens, cfg)

    cost_entry = CostEntry(
        stage="",
        model=model,
        doi="",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
    )
    logger.debug(f"OpenAI response: {usage.prompt_tokens} in, {usage.completion_tokens} out, ${cost:.4f}")
    return text, cost_entry


# -- Gemini API wrapper ------------------------------------------------------

def _is_gemini_rate_limit(err_str: str) -> bool:
    """Detect a Gemini rate-limit error without matching words like generate."""
    err_str = err_str.lower()
    return (
        "429" in err_str
        or "resource_exhausted" in err_str
        or "quota" in err_str
        or re.search(r"\brate.?limit", err_str) is not None
    )

async def call_gemini(
    prompt: str,
    system: str = "",
    model: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    config: dict | None = None,
) -> tuple[str, CostEntry]:
    """Call Google Gemini API (async). Returns (response_text, cost_entry)."""
    client = genai.Client()

    kwargs = {
        "model": model,
        "contents": prompt,
        "config": genai.types.GenerateContentConfig(
            system_instruction=system if system else None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    }

    logger.debug(f"Calling Gemini {model}, prompt length ~{len(prompt)} chars")
    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            response = await client.aio.models.generate_content(**kwargs)
            break
        except Exception as e:
            # Detect rate limit (429) vs transient error.
            is_rate_limit = _is_gemini_rate_limit(str(e))
            wait = 30 if is_rate_limit else 2 ** attempt
            if attempt < max_retries:
                logger.warning(f"  Gemini API error (attempt {attempt + 1}/{max_retries + 1}): {e} — retrying in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise

    text = response.text or ""
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0
    cfg = config or {}
    cost = compute_cost(model, input_tokens, output_tokens, cfg)

    cost_entry = CostEntry(
        stage="",
        model=model,
        doi="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )
    logger.debug(f"Gemini response: {input_tokens} in, {output_tokens} out, ${cost:.4f}")
    return text, cost_entry


# -- Unified LLM dispatcher -------------------------------------------------

_PROVIDER_PREFIXES = {
    "claude-": "anthropic",
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini-": "gemini",
}


def _detect_provider(model: str) -> str:
    """Detect provider from model name prefix."""
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer a provider for model '{model}'. Use a supported model "
        f"prefix ({', '.join(sorted(_PROVIDER_PREFIXES))}) instead of "
        "silently sending document content to the wrong provider."
    )


async def call_llm(
    prompt: str,
    system: str = "",
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    images: list[dict] | None = None,
    config: dict | None = None,
    provider: str | None = None,
) -> tuple[str, CostEntry]:
    """Unified LLM dispatcher. Routes to correct provider based on model name."""
    if provider is None:
        provider = _detect_provider(model)

    # Only the Anthropic wrapper forwards images. Silently dropping them
    # would make a vision call look like a failed text call, which is how the
    # vision fallback came to look enabled while never actually running.
    if images and provider != "anthropic":
        raise ValueError(
            f"Image input requested for model '{model}' (provider "
            f"'{provider}'), but only the Anthropic wrapper forwards images. "
            f"Configure a vision-capable Anthropic model, or extend "
            f"call_openai/call_gemini to accept images."
        )

    if provider == "anthropic":
        return await call_anthropic(prompt, system, model, max_tokens, temperature, images, config)
    elif provider == "openai":
        return await call_openai(prompt, system, model, max_tokens, temperature, config)
    elif provider == "gemini":
        return await call_gemini(prompt, system, model, max_tokens, temperature, config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# -- Batch API wrappers ------------------------------------------------------

def _sanitize_custom_id(cid: str) -> str:
    """Sanitize custom_id to match Anthropic's ^[a-zA-Z0-9_-]{1,64}$ pattern."""
    # Replace any non-alphanumeric/underscore/hyphen chars with underscore
    cid = str(cid)
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', cid)
    # Replacement itself can collide (``a/b`` and ``a b`` both become
    # ``a_b``), not only truncation. Add a stable suffix whenever information
    # was removed so result routing cannot attach one paper's response to
    # another paper.
    if sanitized != cid or len(sanitized) > 64 or not sanitized:
        digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()[:9]
        sanitized = sanitized[:54] + "_" + digest
    return sanitized


async def submit_anthropic_batch(
    requests: list[dict],
    config: dict | None = None,
) -> str:
    """Submit a batch of Anthropic Messages API requests.

    Each request dict must have:
      - custom_id: str  (unique identifier for matching results)
      - model: str
      - max_tokens: int
      - messages: list[dict]
      - system: str (optional)

    Returns batch_id.
    """
    client = anthropic.Anthropic()

    try:
        supports_temperature = "temperature" in inspect.signature(
            client.messages.create
        ).parameters
    except (AttributeError, TypeError, ValueError):
        supports_temperature = False

    batch_requests = []
    for req in requests:
        params = {
            "model": req["model"],
            "max_tokens": req["max_tokens"],
            "messages": req["messages"],
        }
        if req.get("system"):
            params["system"] = req["system"]
        if supports_temperature and req.get("temperature") is not None:
            params["temperature"] = req["temperature"]

        batch_requests.append({
            "custom_id": _sanitize_custom_id(req["custom_id"]),
            "params": params,
        })

    logger.info(f"Submitting Anthropic batch with {len(batch_requests)} requests...")
    batch = client.messages.batches.create(requests=batch_requests)
    logger.info(f"  Batch created: {batch.id} (status: {batch.processing_status})")
    return batch.id


async def poll_anthropic_batch(
    batch_id: str,
    poll_interval: float = 60.0,
    config: dict | None = None,
) -> dict[str, dict]:
    """Poll an Anthropic batch until completion. Returns {custom_id: result_dict}.

    result_dict has keys: text, input_tokens, output_tokens, error (if failed).
    """
    client = anthropic.Anthropic()
    timeout_seconds = float((config or {}).get("batch_timeout_seconds", 86400))
    started = asyncio.get_running_loop().time()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        logger.info(
            f"  Batch {batch_id}: {status} "
            f"(succeeded={counts.succeeded}, errored={counts.errored}, "
            f"processing={counts.processing}, canceled={counts.canceled})"
        )

        if status == "ended":
            break

        if asyncio.get_running_loop().time() - started >= timeout_seconds:
            raise TimeoutError(
                f"Anthropic batch {batch_id} did not finish within "
                f"{timeout_seconds:g} seconds"
            )

        await asyncio.sleep(poll_interval)

    # Retrieve results
    results: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            message = result.result.message
            text = message.content[0].text if message.content else ""
            results[custom_id] = {
                "text": text,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
        else:
            error_msg = str(result.result) if result.result else "Unknown error"
            logger.warning(f"  Batch request {custom_id} failed: {error_msg}")
            results[custom_id] = {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": error_msg,
            }

    logger.info(f"  Batch {batch_id}: retrieved {len(results)} results")
    return results


async def submit_openai_batch(
    requests: list[dict],
    config: dict | None = None,
) -> str:
    """Submit a batch of OpenAI Chat Completions API requests.

    Each request dict must have:
      - custom_id: str
      - model: str
      - max_tokens: int
      - messages: list[dict]

    Returns batch_id.
    """
    client = openai.OpenAI()

    # Build JSONL content
    lines = []
    for req in requests:
        messages = list(req["messages"])
        if req.get("system"):
            messages = [
                {"role": "system", "content": req["system"]},
                *messages,
            ]
        body = {
            "model": req["model"],
            "messages": messages,
        }
        is_gpt5 = req["model"].startswith("gpt-5")
        if is_gpt5:
            body["max_completion_tokens"] = req["max_tokens"]
        else:
            body["max_tokens"] = req["max_tokens"]
            if req.get("temperature") is not None:
                body["temperature"] = req["temperature"]

        line = {
            "custom_id": _sanitize_custom_id(req["custom_id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        lines.append(_json.dumps(line))

    jsonl_content = "\n".join(lines)

    # Upload JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(jsonl_content)
        temp_path = f.name

    def _upload_and_submit():
        try:
            with open(temp_path, "rb") as file_handle:
                file_obj = client.files.create(file=file_handle, purpose="batch")
            logger.info(
                "Submitting OpenAI batch with %d requests (file: %s)...",
                len(lines),
                file_obj.id,
            )
            return client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        finally:
            # Cleanup belongs to the worker that owns the file handle. This
            # avoids deleting the file while an upload continues after caller
            # cancellation.
            Path(temp_path).unlink(missing_ok=True)

    operation = asyncio.create_task(asyncio.to_thread(_upload_and_submit))
    try:
        batch = await asyncio.shield(operation)
    except asyncio.CancelledError:
        # The remote request may already be committed. Settle it before
        # propagating cancellation so callers cannot unknowingly overlap it
        # with a retry.
        try:
            await operation
        # Worker errors are secondary to the caller's cancellation request.
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenAI batch submission failed during cancellation: %s", exc)
        raise

    logger.info(f"  Batch created: {batch.id} (status: {batch.status})")
    return batch.id


async def poll_openai_batch(
    batch_id: str,
    poll_interval: float = 60.0,
    config: dict | None = None,
) -> dict[str, dict]:
    """Poll an OpenAI batch until completion. Returns {custom_id: result_dict}."""
    client = openai.OpenAI()
    timeout_seconds = float((config or {}).get("batch_timeout_seconds", 86400))
    started = asyncio.get_running_loop().time()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        logger.info(
            f"  Batch {batch_id}: {status} "
            f"(completed={batch.request_counts.completed}, "
            f"failed={batch.request_counts.failed}, "
            f"total={batch.request_counts.total})"
        )

        if status in ("completed", "failed", "expired", "cancelled"):
            break

        if asyncio.get_running_loop().time() - started >= timeout_seconds:
            raise TimeoutError(
                f"OpenAI batch {batch_id} did not finish within "
                f"{timeout_seconds:g} seconds"
            )

        await asyncio.sleep(poll_interval)

    if not batch.output_file_id:
        logger.error(f"  Batch {batch_id} has no output file (status: {status})")
        return {}

    # Download results
    content = client.files.content(batch.output_file_id)
    results: dict[str, dict] = {}
    for line in content.text.strip().split("\n"):
        if not line.strip():
            continue
        entry = _json.loads(line)
        custom_id = entry["custom_id"]
        response = entry.get("response", {})

        if response.get("status_code") == 200:
            body = response["body"]
            text = body["choices"][0]["message"]["content"]
            usage = body["usage"]
            results[custom_id] = {
                "text": text,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        else:
            error_msg = str(response.get("error", "Unknown error"))
            logger.warning(f"  Batch request {custom_id} failed: {error_msg}")
            results[custom_id] = {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": error_msg,
            }

    logger.info(f"  Batch {batch_id}: retrieved {len(results)} results")
    return results


async def run_batch(
    requests: list[dict],
    config: dict | None = None,
    poll_interval: float = 60.0,
) -> dict[str, tuple[str, CostEntry]]:
    """Submit a batch, poll until done, return {custom_id: (text, cost_entry)}.

    Automatically routes to Anthropic or OpenAI based on model name. Mixed
    provider lists are partitioned into provider-native batches.

    Each request dict must have: custom_id, model, max_tokens, messages,
    and optionally: system, temperature.
    """
    if not requests:
        return {}

    by_provider: dict[str, list[dict]] = {}
    for request in requests:
        provider = _detect_provider(request["model"])
        by_provider.setdefault(provider, []).append(request)

    # A table pass may deliberately use (for example) Anthropic for ordinary
    # tables and OpenAI for equation tables. Submit one provider-native batch
    # per group; routing the whole list according to its first model leaks data
    # to the wrong service and fails with an invalid model name.
    if len(by_provider) > 1:
        grouped_results = await asyncio.gather(*(
            run_batch(group, config=config, poll_interval=poll_interval)
            for _provider, group in sorted(by_provider.items())
        ))
        combined: dict[str, tuple[str, CostEntry]] = {}
        for result in grouped_results:
            overlap = combined.keys() & result.keys()
            if overlap:
                raise ValueError(f"Duplicate batch custom_id(s): {sorted(overlap)}")
            combined.update(result)
        return combined

    provider, provider_requests = next(iter(by_provider.items()))
    cfg = config or {}
    batch_discount = cfg.get("batch_discount", 0.5)

    if provider == "anthropic":
        batch_id = await submit_anthropic_batch(provider_requests, config)
        raw_results = await poll_anthropic_batch(batch_id, poll_interval, config)
    elif provider == "openai":
        batch_id = await submit_openai_batch(provider_requests, config)
        raw_results = await poll_openai_batch(batch_id, poll_interval, config)
    else:
        raise ValueError(f"Batch API not supported for provider: {provider}")

    # Build reverse mapping: sanitized_id -> original_id
    request_map = {
        _sanitize_custom_id(req["custom_id"]): req for req in provider_requests
    }
    if len(request_map) != len(provider_requests):
        raise ValueError("Batch custom_id values collide after sanitization")

    # Convert to (text, CostEntry) pairs with batch discount applied
    results: dict[str, tuple[str, CostEntry]] = {}
    for sanitized_id, raw in raw_results.items():
        request = request_map.get(sanitized_id)
        if request is None:
            logger.warning(f"  Ignoring unrecognized batch result id: {sanitized_id}")
            continue
        request_model = request["model"]
        cost = compute_cost(
            request_model, raw["input_tokens"], raw["output_tokens"], cfg
        )
        cost *= batch_discount  # 50% batch discount

        cost_entry = CostEntry(
            stage="",
            model=request_model,
            doi="",
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            cost_usd=cost,
        )

        if raw.get("error"):
            logger.warning(f"  Batch result {sanitized_id} had error: {raw['error']}")

        # Map back to original custom_id so callers can look up by their original key
        original_id = request["custom_id"]
        results[original_id] = (raw.get("text", ""), cost_entry)

    return results


# -- JSON helpers ------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    """Extract JSON from LLM response, stripping reasoning text and code fences."""
    text = text.strip()

    # Remove markdown code fences
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
        else:
            text = text[start:].strip()
    elif "```" in text:
        start = text.index("```") + 3
        if "\n" in text[start:start+5]:
            start = text.index("\n", start) + 1
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()

    # Extract an object or a bare array, whichever starts first.
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        last_bracket = text.rfind("]")
        text = text[first_bracket:last_bracket + 1] if last_bracket > first_bracket else text[first_bracket:]
    elif first_brace != -1:
        last_brace = text.rfind("}")
        text = text[first_brace:last_brace + 1] if last_brace > first_brace else text[first_brace:]

    return text.strip()


def parse_json_safe(text: str) -> dict:
    """Parse JSON, attempting to salvage records from truncated responses.
    
    Key insight: Sonnet often generates correct JSON records but the response
    gets truncated at max_tokens. Instead of discarding everything, we extract
    whatever complete records exist before the truncation point.
    """
    import json
    import re

    # Try clean parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed["_response_complete"] = True
        return parsed
    except json.JSONDecodeError:
        pass

    # Try standard repair: add closing brackets
    for suffix in [']}', ']\n}', '"]}', '"}]}', 'null}]}']:
        for trim in range(0, min(200, len(text)), 1):
            candidate = text[:len(text) - trim] if trim > 0 else text
            try:
                parsed = json.loads(candidate + suffix)
                if isinstance(parsed, dict):
                    parsed["_response_complete"] = False
                return parsed
            except json.JSONDecodeError:
                continue

    # Salvage mode: extract individual complete objects from a records OR
    # equations array. Equation responses often finish the valuable array and
    # are truncated only inside a trailing notes string; discarding the whole
    # coefficient table in that case loses otherwise valid source data.
    records_match = re.search(r'"records"\s*:\s*\[', text)
    equations_match = re.search(r'"equations"\s*:\s*\[', text)
    array_match = records_match or equations_match
    array_key = "records" if records_match else "equations"
    if not array_match:
        logger.warning(
            "Could not repair model JSON (length=%d, sha256=%s); raw "
            "document-derived output is not written to logs",
            len(text),
            hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12],
        )
        return {
            "records": [],
            "notes": "JSON was truncated and could not be repaired.",
            "_response_complete": False,
        }

    # Extract individual record objects {...} from the array
    array_start = array_match.end()
    salvaged_objects = []
    depth = 0
    obj_start = None
    in_string = False
    escaped = False

    for i in range(array_start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_text = text[obj_start:i + 1]
                try:
                    salvaged_objects.append(json.loads(obj_text))
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif c == ']' and depth == 0:
            break  # End of records array

    if salvaged_objects:
        logger.info(
            f"    Salvaged {len(salvaged_objects)} {array_key} "
            "from truncated JSON"
        )
        return {
            array_key: salvaged_objects,
            "notes": "Partially salvaged from truncated response.",
            "_response_complete": False,
        }

    logger.warning(
        "Could not repair model JSON (length=%d, sha256=%s); raw "
        "document-derived output is not written to logs",
        len(text),
        hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12],
    )
    return {
        "records": [],
        "notes": "JSON was truncated and could not be repaired.",
        "_response_complete": False,
    }


# -- Prompt loading ----------------------------------------------------------

def load_skill(skill_name: str) -> str:
    """Load a prompt template from skills/ directory."""
    skill_path = PROJECT_ROOT / "skills" / f"{skill_name}.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")
    return skill_path.read_text(encoding="utf-8")


# -- Checkpoint DB -----------------------------------------------------------

class CheckpointDB:
    """Lightweight checkpoint store backed by SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                doi TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'discovered',
                pdf_path TEXT DEFAULT '',
                parse_time TEXT,
                screen_time TEXT,
                extract_time TEXT,
                assemble_time TEXT,
                error_message TEXT,
                api_cost_usd REAL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                model TEXT NOT NULL,
                doi TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def upsert(self, cp: PaperCheckpoint) -> None:
        self._conn.execute(
            """INSERT INTO papers (doi, status, pdf_path, parse_time, screen_time,
                   extract_time, assemble_time, error_message, api_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doi) DO UPDATE SET
                   status=excluded.status,
                   pdf_path=CASE WHEN excluded.pdf_path IS NULL OR excluded.pdf_path=''
                                 THEN papers.pdf_path ELSE excluded.pdf_path END,
                   parse_time=COALESCE(excluded.parse_time, parse_time),
                   screen_time=COALESCE(excluded.screen_time, screen_time),
                   extract_time=COALESCE(excluded.extract_time, extract_time),
                   assemble_time=COALESCE(excluded.assemble_time, assemble_time),
                   error_message=COALESCE(excluded.error_message, papers.error_message),
                   api_cost_usd=api_cost_usd + excluded.api_cost_usd
            """,
            (
                cp.doi, cp.status.value, cp.pdf_path,
                cp.parse_time.isoformat() if cp.parse_time else None,
                cp.screen_time.isoformat() if cp.screen_time else None,
                cp.extract_time.isoformat() if cp.extract_time else None,
                cp.assemble_time.isoformat() if cp.assemble_time else None,
                cp.error_message, cp.api_cost_usd,
            ),
        )
        self._conn.commit()

    def get(self, doi: str) -> PaperCheckpoint | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def get_by_status(self, status: PipelineStatus) -> list[PaperCheckpoint]:
        rows = self._conn.execute(
            "SELECT * FROM papers WHERE status = ?", (status.value,)
        ).fetchall()
        return [self._row_to_checkpoint(r) for r in rows]

    def summary(self) -> dict[PipelineStatus, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM papers GROUP BY status"
        ).fetchall()
        result: dict[PipelineStatus, int] = {}
        for row in rows:
            try:
                result[PipelineStatus(row["status"])] = row["cnt"]
            except ValueError:
                pass
        return result

    def add_cost(self, entry: CostEntry) -> None:
        self._conn.execute(
            """INSERT INTO costs (stage, model, doi, input_tokens, output_tokens,
                   cost_usd, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry.stage, entry.model, entry.doi, entry.input_tokens,
             entry.output_tokens, entry.cost_usd, entry.timestamp.isoformat()),
        )
        self._conn.commit()

    def total_cost(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM costs"
        ).fetchone()
        return row["total"]

    def close(self):
        self._conn.close()

    @staticmethod
    def _row_to_checkpoint(row) -> PaperCheckpoint:
        return PaperCheckpoint(
            doi=row["doi"],
            status=PipelineStatus(row["status"]),
            pdf_path=row["pdf_path"] or "",
            parse_time=datetime.fromisoformat(row["parse_time"]) if row["parse_time"] else None,
            screen_time=datetime.fromisoformat(row["screen_time"]) if row["screen_time"] else None,
            extract_time=datetime.fromisoformat(row["extract_time"]) if row["extract_time"] else None,
            assemble_time=datetime.fromisoformat(row["assemble_time"]) if row["assemble_time"] else None,
            error_message=row["error_message"],
            api_cost_usd=row["api_cost_usd"] or 0.0,
        )
