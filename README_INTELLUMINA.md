# Intellumia

This file describes the refactored layout and how to run the demo without modifying the original project README.

Files added by the refactor:

- `intellumina/__init__.py` — package initializer
- `intellumina/knowledge_graph.py` — KnowledgeGraphBuilder and KnowledgeGraphEvaluator modules extracted from the notebook
- `run_demo.py` — demo script that builds, visualizes and evaluates the knowledge graph
- `requirements.txt` — minimal dependency list for the project

## Recommended quick steps

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

4. Run the demo:

```bash
python run_demo.py
```
