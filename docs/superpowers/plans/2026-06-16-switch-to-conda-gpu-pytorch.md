# Switch to Conda GPU PyTorch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the project from `.venv` (CPU PyTorch) to conda `pytorch251` (GPU PyTorch) environment.

**Architecture:** Single-file config change (`pyproject.toml`) + dependency reinstall. No code changes needed.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, conda

---

### Task 1: Modify pyproject.toml — lower Python version requirement

**Files:**
- Modify: `pyproject.toml:6`

- [ ] **Step 1: Change `requires-python` from `>=3.11` to `>=3.10`**

Line 6 of `pyproject.toml`:
```
requires-python = ">=3.11"
```
Change to:
```
requires-python = ">=3.10"
```

- [ ] **Step 2: Verify the change**

Run: `Select-String -Path pyproject.toml -Pattern 'requires-python'`
Expected output shows: `requires-python = ">=3.10"`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "fix: lower requires-python to >=3.10 for conda pytorch251 compatibility

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Install project dependencies into pytorch251 conda environment

**Files:**
- Modify: `E:\aaa_SpecializedSoftware\MiniConda\envs\pytorch251\` (site-packages)

- [ ] **Step 1: Activate conda env and install project in editable mode**

```powershell
conda activate pytorch251
pip install -e .
```

Expected: All dependencies install without errors. Key packages installed:
- fastapi, uvicorn, typer, rich
- langchain-core, langchain-community, langchain-text-splitters, langchain-openai, langchain-chroma
- chromadb, sentence-transformers
- watchdog, pyyaml, python-dotenv, httpx, sse-starlette

- [ ] **Step 2: Verify key imports work**

```powershell
conda activate pytorch251
python -c "import fastapi; import chromadb; import langchain_core; import sentence_transformers; print('All imports OK')"
```

Expected: `All imports OK`

---

### Task 3: Verify GPU inference works

- [ ] **Step 1: Confirm CUDA is available**

```powershell
conda activate pytorch251
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA: True
GPU: <GPU name>
```

- [ ] **Step 2: Run a full RAG inference**

```powershell
conda activate pytorch251
rag ask "什么是RAG?"
```

Expected: Returns a RAG-generated answer within a few seconds (GPU accelerated).

- [ ] **Step 3: Commit verification result**

```bash
git add -A
git commit -m "verify: GPU inference working with pytorch251 conda environment

Co-Authored-By: Claude <noreply@anthropic.com>"
```
