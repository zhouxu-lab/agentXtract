#!/usr/bin/env python3
"""Benchmark configured LLMs on the dielectric extraction pipeline.

Runs extract + assemble stages with each model, then evaluates against gold standard.
Parse and screen stages are reused (model-independent).

Usage:
    python benchmark.py                                    # All configured models
    python benchmark.py --models haiku sonnet              # Specific models
    python benchmark.py --resume                           # Resume interrupted run
    python benchmark.py --concurrency 10                   # Higher parallelism
    python benchmark.py --no-gold                          # Skip evaluation (no gold standard)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def extracted_filename(doi: str) -> str:
    """Per-paper extraction JSON filename written by src.table_extractor."""
    return f"{doi.replace('/', '_')}_table.json"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    # Pricing below is the standard realtime API list price in USD per million
    # tokens, verified 2026-09-04. Refresh it before publishing cost results.
    # Anthropic
    "haiku": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    },
    "sonnet": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    },
    "opus": {
        "model": "claude-opus-4-6",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    },
    # OpenAI
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
    },
    "gpt-4.1": {
        "model": "gpt-4.1",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    },
    "gpt-5-mini": {
        "model": "gpt-5-mini",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 0.25, "output_per_mtok": 2.00},
    },
    "gpt-5.4": {
        "model": "gpt-5.4",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 2.50, "output_per_mtok": 15.00},
    },
    # Google Gemini
    "gemini-flash": {
        "model": "gemini-2.5-flash",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 0.30, "output_per_mtok": 2.50},
    },
    "gemini-3-flash": {
        "model": "gemini-3-flash-preview",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 0.50, "output_per_mtok": 3.00},
    },
    "gemini-3.1-pro": {
        "model": "gemini-3.1-pro-preview",
        "max_tokens": 8192,
        "temperature": 0.0,
        "pricing": {"input_per_mtok": 2.00, "output_per_mtok": 12.00},
    },
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_base_config() -> dict:
    from src.utils import load_config
    config = load_config(
        "configs/test_mode.yaml", "configs/models.yaml", "configs/thresholds.yaml"
    )
    for key in ("input_dir", "output_dir"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((PROJECT_ROOT / config[key]).resolve())
    return config


def _patch_config_for_model(config: dict, model_key: str, concurrency: int | None = None) -> dict:
    m = MODELS[model_key]
    model_config = {
        "model": m["model"],
        "max_tokens": m["max_tokens"],
        "temperature": m["temperature"],
    }
    # A benchmark candidate must own every active model-dependent extraction
    # path. Leaving text or equation settings at their defaults silently mixes
    # another model's output into the candidate run.
    config["table_extractor"] = dict(model_config)
    config["text_extractor"] = dict(model_config)
    config["equation_extractor"] = dict(model_config)
    # Image payloads are not yet implemented consistently across all provider
    # wrappers. Disable vision for every candidate so the cross-provider score
    # compares the same input modality rather than advantaging one provider.
    config.setdefault("vision_fallback", {})["enabled"] = False
    config.setdefault("pricing", {})
    config["pricing"][m["model"]] = m["pricing"]
    # Disable batch API for benchmarking (avoids duplicate ID collisions
    # and works with all providers including Gemini)
    config["use_batch_api"] = False
    if concurrency:
        config["default_concurrency"] = concurrency
    return config


def _requires_text_extraction(paper) -> bool:
    """Use the production pipeline's prose-only text-extraction rule."""
    return not bool(paper.metadata.data_tables)


def _benchmark_artifact_complete(paper, extracted_dir: Path) -> bool:
    """Check the route-specific artifact required to resume one paper."""
    from src.pipeline import _already_extracted

    return _already_extracted(
        paper.paper_uid,
        extracted_dir,
        require_text=_requires_text_extraction(paper),
    )


# ---------------------------------------------------------------------------
# Pipeline run (extract + assemble only)
# ---------------------------------------------------------------------------

async def run_extract_and_assemble(
    config: dict, model_key: str, run_dir: Path, resume: bool = False,
    paper_dois: set[str] | None = None,
):
    """Run a benchmark with guaranteed config restoration and DB cleanup."""
    from src.utils import CheckpointDB

    checkpoint_db = CheckpointDB(run_dir / "pipeline.db")
    try:
        return await _run_extract_and_assemble(
            copy.deepcopy(config), model_key, run_dir, checkpoint_db,
            resume=resume, paper_dois=paper_dois,
        )
    finally:
        checkpoint_db.close()


async def _run_extract_and_assemble(
    config: dict, model_key: str, run_dir: Path, checkpoint_db,
    resume: bool = False, paper_dois: set[str] | None = None,
):
    """Run extract + assemble stages, writing outputs to run_dir.

    If resume=True, extraction is resumed per paper: papers whose extraction
    JSON already exists are skipped, only the missing ones are extracted, and
    assembly merges everything found in the extracted dir.
    """
    from src.assembler import run as assemble_run
    from src.pipeline import (
        _clean_stale_text,
        load_extraction_results,
        load_parsed_papers,
        load_screener_results,
        pair_papers_with_screener,
    )
    from src.table_extractor import run as table_run
    from src.text_extractor import run as text_run

    parsed_dir = Path(config["output_dir"]) / "parsed"
    extracted_dir = run_dir / "extracted"
    database_dir = run_dir / "database"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    database_dir.mkdir(parents=True, exist_ok=True)

    original_output = config["output_dir"]

    # Use the same manifest-aware loaders as the production pipeline. Loading
    # every JSON in parsed/ used to benchmark stale copies of the same paper.
    papers = load_parsed_papers(parsed_dir)
    screener_results = load_screener_results(parsed_dir)

    # Filter to specific papers if requested
    if paper_dois is not None:
        papers = [
            p for p in papers
            if p.doi in paper_dois or p.metadata.doi in paper_dois
        ]
        logger.info(f"[{model_key}] Filtered to {len(papers)} papers (gold standard subset)")

    paired = pair_papers_with_screener(papers, screener_results)

    # Scope resume and assembly to exactly this benchmark corpus. Without a
    # run-local manifest, artifacts left by an earlier benchmark invocation
    # were silently included in the new model score.
    from src.paper_id import Manifest
    run_manifest = Manifest(run_dir / "manifest.json", strict=True)
    run_manifest.entries = {}
    for paper in papers:
        if paper.paper_uid:
            run_manifest.upsert(
                paper.paper_uid,
                pdf_path=paper.pdf_path,
                slug=Path(paper.pdf_path).stem if paper.pdf_path else paper.doi,
                doi=paper.metadata.doi or paper.doi,
                title=paper.metadata.title,
                year=paper.metadata.year,
            )
    run_manifest.save()

    config["output_dir"] = str(run_dir)

    # Ensure dirs exist
    (run_dir / "parsed" / "table_images").mkdir(parents=True, exist_ok=True)
    src_table_images = parsed_dir / "table_images"
    dst_table_images = run_dir / "parsed" / "table_images"
    if src_table_images.exists():
        for img in src_table_images.glob("*"):
            dst = dst_table_images / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    # Per-paper resume. A paper's JSON is written only after that paper
    # finishes, so an existing file means the paper is done. When resuming we
    # extract only the papers still missing a JSON, instead of skipping
    # extraction entirely (the old all-or-nothing behaviour resumed a
    # partially-completed run straight to assembly, producing garbage recall).
    completed_uids = {
        paper.paper_uid
        for paper, _screener in paired
        if _benchmark_artifact_complete(paper, extracted_dir)
    }
    if resume and completed_uids:
        missing = [
            (p, sr) for (p, sr) in paired
            if p.paper_uid not in completed_uids
        ]
        logger.info(f"[{model_key}] Resume: {len(completed_uids)} papers done, "
                    f"{len(missing)} still missing")
        paired = missing

    if resume and completed_uids and not paired:
        logger.info(f"[{model_key}] All papers already extracted, skipping extraction")
        extract_time = 0.0
    else:
        logger.info(f"[{model_key}] Starting extraction with {MODELS[model_key]['model']} ({len(paired)} papers)")

        t0 = time.time()
        text_pairs = [
            (paper, screener_result)
            for paper, screener_result in paired
            if _requires_text_extraction(paper)
        ]
        for paper, _screener_result in paired:
            if paper.metadata.data_tables:
                _clean_stale_text(paper, extracted_dir)

        text_results = (
            await text_run(
                text_pairs, config, checkpoint_db, progress=None
            )
            if text_pairs else []
        )
        table_results = await table_run(paired, config, checkpoint_db, progress=None)
        incomplete = [
            result for result in [*text_results, *table_results]
            if not result.complete
        ]
        if incomplete:
            raise RuntimeError(
                f"[{model_key}] {len(incomplete)} extraction artifact(s) incomplete"
            )
        extract_time = time.time() - t0
        logger.info(
            f"[{model_key}] Extraction calls completed in {extract_time:.1f}s"
        )

    # Run assembly
    extraction_results = load_extraction_results(
        extracted_dir, parsed_papers=papers
    )
    incomplete = [result for result in extraction_results if not result.complete]
    if incomplete:
        raise RuntimeError(
            f"[{model_key}] {len(incomplete)} extraction artifact(s) incomplete"
        )
    total_records = sum(len(result.records) for result in extraction_results)

    papers_dict = {
        (p.paper_uid or f"{p.doi}#{index}"): p
        for index, p in enumerate(papers)
    }
    t1 = time.time()
    assemble_run(
        extraction_results=extraction_results,
        parsed_papers=papers_dict,
        output_dir=database_dir,
        config=config,
    )
    assemble_time = time.time() - t1

    config["output_dir"] = original_output

    # Collect cost info
    total_cost = checkpoint_db.total_cost()
    cost_rows = checkpoint_db._conn.execute(
        """SELECT doi, SUM(input_tokens) AS inp, SUM(output_tokens) AS out,
                  SUM(cost_usd) AS cost
           FROM costs GROUP BY doi ORDER BY doi"""
    ).fetchall()

    per_paper_costs = {}
    total_input_tokens = 0
    total_output_tokens = 0
    for r in cost_rows:
        per_paper_costs[r["doi"]] = {
            "input_tokens": r["inp"], "output_tokens": r["out"], "cost": r["cost"]
        }
        total_input_tokens += r["inp"]
        total_output_tokens += r["out"]

    return {
        "model_key": model_key,
        "model": MODELS[model_key]["model"],
        "extract_time": extract_time,
        "assemble_time": assemble_time,
        "total_time": extract_time + assemble_time,
        "total_cost": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_records": total_records,
        "n_papers": len(papers),
        "per_paper_costs": per_paper_costs,
        "pred_csv": str(database_dir / "dielectric_properties.csv"),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_run(pred_csv: str, gold_path: str) -> dict:
    """Run evaluation and return metrics dict."""
    from evaluate import (
        compute_metrics,
        filter_pred_by_gold_frequencies,
        filter_pred_by_gold_paper_ids,
        load_gold,
        load_pred,
        match_records,
        per_paper_breakdown,
    )

    if not Path(pred_csv).exists():
        return {"error": f"Prediction file not found: {pred_csv}"}

    gold = load_gold(gold_path)
    pred = load_pred(pred_csv)

    pred = filter_pred_by_gold_paper_ids(gold, pred)
    pred = filter_pred_by_gold_frequencies(gold, pred)

    tp_pairs, fp_indices, fn_indices = match_records(gold, pred)
    metrics = compute_metrics(gold, pred, tp_pairs, fp_indices, fn_indices)
    breakdown = per_paper_breakdown(gold, pred, tp_pairs, fp_indices, fn_indices)

    return {
        "gold_count": len(gold),
        "pred_count": len(pred),
        **metrics,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]):
    """Print a comparison table of all model runs."""
    W = 90
    print(f"\n{'='*W}")
    print("  MODEL COMPARISON BENCHMARK")
    print(f"{'='*W}")

    # Summary table
    header = f"  {'Model':<12} {'VA':>6} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>4} {'FN':>4} {'Cost':>8} {'Time':>7} {'$/paper':>8} {'s/paper':>8}"
    print(header)
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            print(f"  {r['model_key']:<12} ERROR: {m['error']}")
            continue
        run = r.get("run", {})
        n_papers = max(run.get("n_papers", 1), 1)
        cost = run.get("total_cost", 0)
        total_time = run.get("total_time", 0)
        print(
            f"  {r['model_key']:<12} "
            f"{m.get('value_accuracy', 0):>5.1%} "
            f"{m.get('precision', 0):>5.1%} "
            f"{m.get('recall', 0):>5.1%} "
            f"{m.get('f1', 0):>5.3f} "
            f"{m.get('tp', 0):>5} "
            f"{m.get('fp', 0):>4} "
            f"{m.get('fn', 0):>4} "
            f"${cost:>7.4f} "
            f"{total_time:>6.1f}s "
            f"${cost/n_papers:>7.4f} "
            f"{total_time/n_papers:>6.1f}s"
        )

    # Per-paper breakdown
    print("\n  PER-PAPER BREAKDOWN")
    print(f"  {'Model':<12} {'Paper':<40} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'Cost':>8}")
    print(f"  {'-'*12} {'-'*40} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*6} {'-'*8}")

    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        breakdown = m.get("breakdown", {})
        per_paper_costs = r.get("run", {}).get("per_paper_costs", {})
        for paper_doi in sorted(breakdown.keys()):
            b = breakdown[paper_doi]
            label = paper_doi.split("/")[-1][:40]
            paper_cost = per_paper_costs.get(paper_doi, {}).get("cost", 0)
            print(
                f"  {r['model_key']:<12} {label:<40} "
                f"{b['tp']:>4} {b['fp']:>4} {b['fn']:>4} "
                f"{b['precision']:>5.1%} {b['recall']:>5.1%} "
                f"${paper_cost:>7.4f}"
            )

    # Token usage
    print("\n  TOKEN USAGE")
    print(f"  {'Model':<12} {'Input tokens':>14} {'Output tokens':>14} {'Total cost':>10}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*10}")
    for r in results:
        run = r.get("run", {})
        if not run:
            continue
        print(
            f"  {r['model_key']:<12} "
            f"{run.get('total_input_tokens', 0):>14,} "
            f"{run.get('total_output_tokens', 0):>14,} "
            f"${run.get('total_cost', 0):>9.4f}"
        )

    print(f"\n{'='*W}")


def save_results_json(results: list[dict], output_path: Path):
    """Save benchmark results to JSON for later analysis."""
    serializable = []
    for r in results:
        per_paper_raw = r.get("metrics", {}).get("breakdown", {})
        per_paper = {}
        for k, v in per_paper_raw.items():
            per_paper[str(k)] = {sk: (float(sv) if isinstance(sv, (int, float)) else str(sv)) for sk, sv in v.items()}

        metrics_raw = r.get("metrics", {})
        metrics_safe = {}
        for k, v in metrics_raw.items():
            if k == "breakdown":
                continue
            if isinstance(v, (int, float, str, bool, type(None))):
                metrics_safe[k] = v
            else:
                metrics_safe[k] = str(v)

        run = r.get("run", {})
        entry = {
            "model_key": r["model_key"],
            "model": run.get("model", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_papers": run.get("n_papers", 0),
            "metrics": metrics_safe,
            "per_paper": per_paper,
            "cost": {
                "total": run.get("total_cost", 0),
                "per_paper": run.get("total_cost", 0) / max(run.get("n_papers", 1), 1),
                "input_tokens": run.get("total_input_tokens", 0),
                "output_tokens": run.get("total_output_tokens", 0),
            },
            "time": {
                "extract_s": run.get("extract_time", 0),
                "assemble_s": run.get("assemble_time", 0),
                "total_s": run.get("total_time", 0),
                "per_paper_s": run.get("total_time", 0) / max(run.get("n_papers", 1), 1),
            },
        }
        serializable.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def save_results_csv(results: list[dict], output_path: Path):
    """Save benchmark summary as CSV for easy spreadsheet import."""
    rows = []
    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        run = r.get("run", {})
        n_papers = max(run.get("n_papers", 1), 1)
        rows.append({
            "model": r["model_key"],
            "model_id": run.get("model", ""),
            "n_papers": n_papers,
            "value_accuracy": round(m.get("value_accuracy", 0), 4),
            "precision": round(m.get("precision", 0), 4),
            "recall": round(m.get("recall", 0), 4),
            "f1": round(m.get("f1", 0), 4),
            "tp": m.get("tp", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
            "value_errors": m.get("value_errors", 0),
            "total_cost_usd": round(run.get("total_cost", 0), 4),
            "cost_per_paper_usd": round(run.get("total_cost", 0) / n_papers, 4),
            "total_time_s": round(run.get("total_time", 0), 1),
            "time_per_paper_s": round(run.get("total_time", 0) / n_papers, 1),
            "input_tokens": run.get("total_input_tokens", 0),
            "output_tokens": run.get("total_output_tokens", 0),
        })

    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV summary saved to {output_path}")

    # Also save per-paper breakdown CSV
    paper_rows = []
    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        breakdown = m.get("breakdown", {})
        per_paper_costs = r.get("run", {}).get("per_paper_costs", {})
        for paper_doi in sorted(breakdown.keys()):
            b = breakdown[paper_doi]
            pc = per_paper_costs.get(paper_doi, {})
            paper_rows.append({
                "model": r["model_key"],
                "doi": paper_doi,
                "paper": paper_doi.split("/")[-1],
                "tp": b["tp"],
                "fp": b["fp"],
                "fn": b["fn"],
                "precision": round(b["precision"], 4),
                "recall": round(b["recall"], 4),
                "cost_usd": round(pc.get("cost", 0), 4),
                "input_tokens": pc.get("input_tokens", 0),
                "output_tokens": pc.get("output_tokens", 0),
            })

    if paper_rows:
        paper_csv = output_path.parent / output_path.name.replace(".csv", "_per_paper.csv")
        with open(paper_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=paper_rows[0].keys())
            writer.writeheader()
            writer.writerows(paper_rows)
        print(f"Per-paper CSV saved to {paper_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    _load_dotenv(PROJECT_ROOT / ".env", override=False)

    model_keys = args.models
    gold_path = args.gold
    benchmark_dir = Path(args.output)
    resume = args.resume
    skip_eval = args.no_gold

    if not skip_eval and not gold_path:
        print("WARNING: No reference file supplied; running without evaluation")
        print("         Use --gold PATH to evaluate or --no-gold to suppress this warning")
        skip_eval = True
    elif not skip_eval and not Path(gold_path).exists():
        print(f"WARNING: Reference data not found: {gold_path}")
        print("         Running without evaluation")
        skip_eval = True

    # Ensure parse + screen have been run
    base_config = _load_base_config()
    parsed_dir = Path(base_config["output_dir"]) / "parsed"
    if not parsed_dir.exists() or not list(parsed_dir.glob("*.json")):
        print("ERROR: No parsed data found. Run the pipeline first:")
        print("  python -m src.pipeline --stage parse")
        print("  python -m src.pipeline --stage screen")
        return

    # Determine paper subset (gold-only mode filters to gold standard DOIs)
    paper_dois = None
    if getattr(args, "gold_only", False) and not skip_eval:
        # Load parsed papers and match DOIs to gold standard paper_ids
        import csv as _csv

        from evaluate import load_gold
        gold_df = load_gold(gold_path)
        gold_pids = set(gold_df["paper_id"].unique())
        # Use the main pipeline CSV (if exists) to get correct paper_id -> DOI mapping
        main_csv = Path(base_config["output_dir"]) / "database" / "dielectric_properties.csv"
        paper_dois = set()
        mapped_pids = set()
        if main_csv.exists():
            def _read_doi_mappings():
                matched_dois = set()
                matched_pids = set()
                with open(main_csv, encoding="utf-8") as handle:
                    for row in _csv.DictReader(handle):
                        if row["paper_id"] in gold_pids and row.get("doi"):
                            matched_dois.add(row["doi"])
                            matched_pids.add(row["paper_id"])
                return matched_dois, matched_pids

            discovered_dois, discovered_pids = await asyncio.to_thread(
                _read_doi_mappings
            )
            paper_dois.update(discovered_dois)
            mapped_pids.update(discovered_pids)
        # Also scan parsed papers for DOIs matching gold paper_ids by stem
        for f in sorted(parsed_dir.glob("*.json")):
            if f.name.startswith("screener_"):
                continue
            stem = f.stem.lower().replace("_", " ").replace(".", " ")
            parsed_text = await asyncio.to_thread(f.read_text, encoding="utf-8")
            doi = json.loads(parsed_text).get("doi", "")
            for pid in gold_pids:
                if pid.lower() == stem or pid.lower().replace(" ", "") == stem.replace(" ", ""):
                    paper_dois.add(doi)
                    if doi:
                        mapped_pids.add(pid)
        paper_dois.discard("")
        unmapped_pids = sorted(gold_pids - mapped_pids)
        if unmapped_pids:
            banner = "!" * 70
            print(banner)
            print(f"WARNING: {len(unmapped_pids)} gold paper_id(s) could not be mapped to a "
                  f"DOI and will NOT be extracted, but they STAY in the gold set —")
            print("every one of their gold records will count as a false negative:")
            for pid in unmapped_pids:
                print(f"  - {pid}")
            print(banner)
            logger.warning(f"Gold-only mode: {len(unmapped_pids)} unmapped gold paper_id(s): "
                           f"{unmapped_pids}")
        print(f"Gold-only mode: {len(paper_dois)} papers mapped, "
              f"{len(unmapped_pids)} gold paper_id(s) unmapped")

    n_papers = (
        len(paper_dois)
        if paper_dois is not None
        else len([
            f for f in parsed_dir.glob("*.json")
            if not f.name.startswith("screener_")
        ])
    )
    print(f"Benchmark: {len(model_keys)} models x {n_papers} papers")

    results = []

    for model_key in model_keys:
        print(f"\n{'#'*60}")
        print(f"  BENCHMARKING: {model_key.upper()} ({MODELS[model_key]['model']})")
        print(f"{'#'*60}")

        run_dir = benchmark_dir / model_key

        if not resume:
            if run_dir.exists():
                # Close any lingering SQLite connections before deleting
                db_file = run_dir / "pipeline.db"
                if db_file.exists():
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(db_file))
                        conn.close()
                    except sqlite3.Error as exc:
                        logger.debug("Could not close benchmark database: %s", exc)
                shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        config = _load_base_config()
        config = _patch_config_for_model(config, model_key, concurrency=args.concurrency)

        try:
            run_info = await run_extract_and_assemble(config, model_key, run_dir, resume=resume, paper_dois=paper_dois)

            if skip_eval:
                metrics = {"note": "No gold standard — evaluation skipped"}
            else:
                metrics = evaluate_run(run_info["pred_csv"], gold_path)

            results.append({
                "model_key": model_key,
                "run": run_info,
                "metrics": metrics,
            })
        # A failed provider/model must not prevent the remaining benchmark
        # candidates from running and being reported.
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({
                "model_key": model_key,
                "run": {},
                "metrics": {"error": str(e)},
            })

    # Print comparison (only if eval was run)
    if not skip_eval:
        print_comparison(results)

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_results_json(results, benchmark_dir / f"benchmark_{timestamp}.json")
    if not skip_eval:
        save_results_csv(results, benchmark_dir / f"benchmark_{timestamp}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models on the agentXtract pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --no-gold                # All configured models
  python benchmark.py --gold reference.csv     # Evaluate against tidy records
  python benchmark.py --models haiku sonnet    # Only Haiku and Sonnet
  python benchmark.py --resume                 # Resume interrupted run
  python benchmark.py --concurrency 10         # More parallel API calls
  python benchmark.py --no-gold                # Skip evaluation
        """,
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help=f"Models to benchmark (default: all {len(MODELS)})",
    )
    parser.add_argument("--gold", default=None,
                        help="Tidy reference CSV, TSV, or Excel path")
    parser.add_argument("--output", default="data/benchmark",
                        help="Output directory for benchmark runs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing extracted files (skip extraction)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override API concurrency (default: from config)")
    parser.add_argument("--no-gold", action="store_true",
                        help="Skip evaluation (no gold standard required)")
    parser.add_argument("--gold-only", action="store_true",
                        help="Only extract papers in the gold standard (saves cost)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
