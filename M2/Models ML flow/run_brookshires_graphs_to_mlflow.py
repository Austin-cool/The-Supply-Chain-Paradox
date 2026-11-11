"""Run Brookshire's MI/comparison scripts and plotting notebooks; log outputs to MLflow.

Place this helper and the three .ipynb files in M2/Models/, with:
  M2/Models/results/merged_all_stores.csv
  M2/Models/make_all_variables_mi_matrix_updated.py
  M2/Models/compare_stores_stockouts_and_profit.py

The two Plots_and_trends notebook files have identical code. By default the
redundant generic checkpoint is skipped. Include it with --include-checkpoint.

Use --dry-run to generate outputs without contacting MLflow (for troubleshooting).
This logs descriptive analysis artifacts; it does not train/validate a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR
INPUT_DEFAULT = MODEL_DIR / "results" / "merged_all_stores.csv"
MI_SCRIPT = MODEL_DIR / "make_all_variables_mi_matrix_updated.py"
COMPARE_SCRIPT = MODEL_DIR / "compare_stores_stockouts_and_profit.py"
NOTEBOOKS = [
    "Plots_and_trends.ipynb",
    "Plots_and_trends-checkpoint-DiegoSegura.ipynb",
    "Plots_and_trends-checkpoint.ipynb",
]


def lookup_notebook(name: str) -> Path:
    for parent in [MODEL_DIR, MODEL_DIR / "notebooks"]:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {name}. Put the notebook in {MODEL_DIR} (or its notebooks/ folder).")


def run_command(command: list[str], description: str) -> None:
    print(f"\nRunning {description}...", flush=True)
    subprocess.run(command, check=True, cwd=MODEL_DIR)


def prepare_notebook(source_path: Path, input_path: Path, chart_dir: Path):
    """Adapt uploaded notebooks in memory; do not modify originals in GitHub."""
    import nbformat

    notebook = nbformat.read(source_path, as_version=4)
    chart_dir.mkdir(parents=True, exist_ok=True)
    preamble = (
        "from pathlib import Path\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        f"INPUT_CSV = Path({str(input_path)!r})\n"
        f"PLOTS_OUT = Path({str(chart_dir)!r})\n"
        "PLOTS_OUT.mkdir(parents=True, exist_ok=True)\n"
    )
    notebook.cells.insert(0, nbformat.v4.new_code_cell(preamble))
    replaced_read = 0
    for cell in notebook.cells[1:]:
        if cell.cell_type != "code":
            continue
        src = cell.source
        for original in ("pd.read_csv('../merged_all_stores.csv')", "pd.read_csv('merged_all_stores.csv')"):
            if original in src:
                src = src.replace(original, "pd.read_csv(INPUT_CSV)")
                replaced_read += 1
        src = src.replace(
            'f"Profit vs zero rate per department in {store}.svg",',
            'PLOTS_OUT / f"Profit vs zero rate per department in {store}.svg",',
        )
        src = src.replace(
            'f"Profit vs sale type per department in {store}.svg",',
            'PLOTS_OUT / f"Profit vs sale type per department in {store}.svg",',
        )
        src = src.replace("Path('/path/to/output_folder')", "PLOTS_OUT")
        # The original final notebook cell only DISPLAYED a chart, rather than saving it.
        # Save it instead so MLflow can log it. Do not rewrite commented-out plt.show().
        if "plt.show()" in src and not all(line.strip().startswith("#") for line in src.splitlines() if "plt.show()" in line):
            lines = src.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "plt.show()":
                    lines[i] = (
                        'fig.savefig(PLOTS_OUT / "store_131_profit_vs_sale_type_summary.png", '
                        'dpi=150, bbox_inches="tight")\nplt.close(fig)'
                    )
            src = "\n".join(lines)
        cell.source = src

    if replaced_read != 1:
        raise ValueError(f"Notebook {source_path.name} had {replaced_read} recognized CSV load(s), expected 1. Check its structure.")
    return notebook


def execute_notebook(source_path: Path, input_path: Path, output_dir: Path, timeout: int) -> dict:
    """Run Jupyter cells on the VM and save both generated graphs and executed notebook."""
    import nbformat
    from nbclient import NotebookClient

    charts = output_dir / "graphs"
    notebook = prepare_notebook(source_path, input_path, charts)
    print(f"\nExecuting notebook {source_path.name}...", flush=True)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(MODEL_DIR)}},
    )
    executed = client.execute()
    output_dir.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, output_dir / "executed.ipynb")
    charts_found = sorted(p for p in charts.iterdir() if p.is_file() and p.suffix.lower() in (".svg", ".png", ".pdf"))
    if not charts_found:
        raise RuntimeError(f"Notebook {source_path.name} ran but generated no graph files in {charts}.")
    print(f"  Created {len(charts_found)} plots in {charts}", flush=True)
    return {"notebook": source_path.name, "plot_count": len(charts_found),
            "plots": [p.name for p in charts_found], "output_folder": str(output_dir)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=INPUT_DEFAULT, help="Merged store/day CSV path")
    p.add_argument("--train-end", default="2025-03-14", help="Shared training cutoff for MI and comparisons")
    p.add_argument("--output-root", type=Path, default=MODEL_DIR / "results" / "mlflow_graphs")
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"),
                   help="Your lab MLflow tracking URI, or set MLFLOW_TRACKING_URI")
    p.add_argument("--experiment", default="Brookshires_M2")
    p.add_argument("--include-checkpoint", action="store_true", help="Run the duplicate generic checkpoint as well")
    p.add_argument("--notebooks-only", action="store_true", help="Skip MI and store comparison scripts")
    p.add_argument("--analysis-only", action="store_true", help="Skip the notebooks")
    p.add_argument("--dry-run", action="store_true", help="Generate local outputs without MLflow upload")
    p.add_argument("--cell-timeout", type=int, default=900, help="Maximum time in seconds per notebook cell")
    a = p.parse_args()
    if a.notebooks_only and a.analysis_only:
        p.error("--notebooks-only and --analysis-only cannot be used together")
    if not a.dry_run and not a.tracking_uri:
        p.error("Set MLFLOW_TRACKING_URI or pass --tracking-uri (or use --dry-run to test locally)")
    return a


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing merged input CSV: {input_path}. Push/transfer it to the lab VM first.")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {"input": str(input_path), "train_end": args.train_end,
                "notebooks": [], "analysis_scripts": [], "source_sha256": {},
                "note": "Descriptive graphs; does not perform model training or unseen-date validation."}

    if not args.notebooks_only:
        for script in (MI_SCRIPT, COMPARE_SCRIPT):
            if not script.is_file():
                raise FileNotFoundError(f"Missing analysis script: {script}")
        analysis = [
            (MI_SCRIPT, output_root / "mi_matrix", ["--exclude-christmas"]),
            (COMPARE_SCRIPT, output_root / "store_comparisons", []),
        ]
        for script, result_dir, extra in analysis:
            cmd = [sys.executable, str(script), "--input", str(input_path),
                   "--train-end", args.train_end, "--output-dir", str(result_dir)] + extra
            run_command(cmd, script.name)
            manifest["analysis_scripts"].append(script.name)
            manifest["source_sha256"][script.name] = hashlib.sha256(script.read_bytes()).hexdigest()

    if not args.analysis_only:
        names = NOTEBOOKS if args.include_checkpoint else NOTEBOOKS[:-1]
        for index, name in enumerate(names):
            source = lookup_notebook(name)
            label = "plots_and_trends" if index == 0 else (
                "diego_checkpoint" if index == 1 else "generic_checkpoint")
            result = execute_notebook(source, input_path, output_root / "notebooks" / label, args.cell_timeout)
            manifest["notebooks"].append(result)
            manifest["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()

    # A manifest helps the professor reproduce exactly what each run executed.
    manifest_file = output_root / "run_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"\nDry run complete. Generated results in {output_root} (nothing uploaded to MLflow).")
        return

    # Lazy import allows local --dry-run without MLflow installed.
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="MI_comparisons_and_plotting_notebooks") as run:
        mlflow.log_param("dataset", input_path.name)
        mlflow.log_param("train_end", args.train_end)
        mlflow.log_param("excluded_christmas_MI", not args.notebooks_only)
        mlflow.log_param("notebooks_executed", len(manifest["notebooks"]))
        mlflow.log_param("analysis_programs_executed", len(manifest["analysis_scripts"]))
        mlflow.log_metric("notebook_graphs_generated", sum(item["plot_count"] for item in manifest["notebooks"]))
        folders_to_log = []
        if not args.notebooks_only:
            folders_to_log.extend(["mi_matrix", "store_comparisons"])
        if not args.analysis_only:
            folders_to_log.append("notebooks")
        for subfolder in folders_to_log:
            folder = output_root / subfolder
            if folder.is_dir() and any(p.is_file() for p in folder.rglob("*")):
                mlflow.log_artifacts(str(folder), artifact_path=subfolder)
        mlflow.log_artifact(str(manifest_file))
        print(f"\nMLflow run: {run.info.run_id}")
        print(f"Tracking server: {args.tracking_uri}")
        print("All graphs and CSVs are under this run's Artifacts tab.")
        print("This logs descriptive analysis only, not R7/R8 predictive model validation.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"ERROR: {exc}")
