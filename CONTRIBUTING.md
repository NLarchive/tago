# Contributing

```bash
pip install -e ".[dev]"
python -m pytest -q                      # unit, architecture, notebook and end-to-end tests
python tools/build_notebook.py --check   # the notebook must match the package and notebooks/narrative.md
```

- **The notebook is generated.** Edit the package under `tago/` (cells are marked `# %% cell: <id>`) and the explanations in `notebooks/narrative.md`, then run `python tools/build_notebook.py`. Every code cell needs an explanation in the narrative.
- **Architecture.** Inner rings never import outer rings (`tests/test_architecture.py`).
- **Results are data.** Raw run artifacts go under `results/<run>/` unchanged. Tables in the README are generated from them and a test compares them; do not type numbers by hand.
- **Every fix is pinned.** A failure seen in a real run gets a test that replays the recorded model output.
- **Task graph.** Update `node.tasks.json` when a task progresses, and run `python -m tago tasks validate`.
