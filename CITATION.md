# Citation and Reproducibility

AgentUnit is designed as a research-grade framework. If you use AgentUnit in your research, please cite it using the following metadata.

## Citation

To cite AgentUnit in publications:

```bibtex
@software{agentunit2024,
  author = {Aviral Garg},
  title = {AgentUnit: A Framework for Multi-Agent System Evaluation and Benchmarking},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aviralgarg05/agentunit}},
  version = {0.1.0}
}
```

## Reproducibility Standards

AgentUnit ensures research reproducibility through:

1.  **Deterministic Evaluation**: Configurable seeds for random number generators and LLM temperature control.
2.  **Versioned Benchmarks**: Fixed versions of GAIA and AgentArena datasets.
3.  **Traceability**: Comprehensive logging of all agent interactions, tool calls, and metric calculations.
4.  **Experiment Tracking**: Built-in tracking of configuration, code versions (git commit), and results (metrics/tracker.py).

### How to Reproduce Experiments

1.  **Environment Setup**:
    ```bash
    git clone https://github.com/aviralgarg05/agentunit.git
    cd agentunit
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    ```

2.  **Configuration**:
    Set the same environment variables (provider API keys) and `ExperimentConfig` parameters.
    Start runs with a fixed seed:
    ```python
    import random
    import numpy as np
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    # Configure LLM temperature to 0.0 for deterministic outputs where possible
    ```

3.  **Running Benchmarks**:
    Use the provided scripts in `examples/` or `experiments/` which log all parameters.

4.  **Verifying Results**:
    Compare your `experiments/experiment_*.json` output with published results.
    Use `src/agentunit/stats` module for statistical significance testing between runs.
