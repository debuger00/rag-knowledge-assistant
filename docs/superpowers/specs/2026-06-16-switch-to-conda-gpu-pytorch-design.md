# Design: Switch from venv (CPU PyTorch) to Conda pytorch251 (GPU PyTorch)

**Date:** 2026-06-16
**Status:** Approved

## Motivation

Current `.venv` uses PyTorch 2.12.0+cpu — inference is slow. The conda environment
`pytorch251` has PyTorch 2.5.1+cu121 with CUDA 12.1 and GPU available.

## Current vs Target

| | Current (.venv) | Target (pytorch251) |
|---|---|---|
| Python | 3.13.13 | 3.10.20 |
| PyTorch | 2.12.0+cpu | 2.5.1+cu121 |
| CUDA | Not available | Available (1 GPU) |
| Location | `<project>/.venv` | `E:\aaa_SpecializedSoftware\MiniConda\envs\pytorch251` |

## Changes

### Single file modified

**`pyproject.toml`** — Lower Python version requirement:

```diff
-requires-python = ">=3.11"
+requires-python = ">=3.10"
```

pytorch251 has Python 3.10.20 which satisfies `>=3.10`.

### Dependencies installed into pytorch251

Run after activating the conda environment:

```bash
conda activate pytorch251
pip install -e .
```

This installs all project dependencies: fastapi, uvicorn, typer, rich, langchain-*,
chromadb, sentence-transformers, watchdog, pyyaml, python-dotenv, httpx, sse-starlette.

### No code changes needed

- Project code uses no Python 3.11+ exclusive syntax
- All dependencies are compatible with Python 3.10

## Verification

```bash
conda activate pytorch251
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK')"
rag ask "什么是RAG?"
```

## Rollback

If issues arise:
1. Revert `pyproject.toml` to `requires-python = ">=3.11"`
2. Activate the old venv: `.venv\Scripts\activate`
