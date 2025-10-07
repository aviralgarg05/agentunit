# No-Code Scenario Builder - Quick Start Guide

Use this guide alongside the [documentation map](../README.md#documentation-map). If you prefer hand-written Python scenarios, start with [Writing Scenarios](writing-scenarios.md); this document focuses on YAML/JSON-driven workflows.

## Overview

The No-Code Scenario Builder allows you to create AgentUnit test scenarios using simple YAML or JSON configuration files instead of writing Python code.

## Basic Example

Create a file `my_scenario.yaml`:

```yaml
name: My First Test
adapter:
  type: openai
  config:
    model: gpt-3.5-turbo
    temperature: 0.7
dataset:
  cases:
    - input: "What is 2+2?"
      expected: "4"
    - input: "What is the capital of France?"
      expected: "Paris"
metrics:
  - correctness
  - latency
timeout: 30
```

Then use it:

```python
from agentunit.nocode import ScenarioBuilder

builder = ScenarioBuilder()

# Load scenario from YAML
scenario = builder.from_yaml("my_scenario.yaml")

# Or generate Python code
code = builder.to_python("my_scenario.yaml")
print(code)
```

## Configuration Schema

### Required Fields

- **name**: Scenario name (string)
- **adapter**: Agent adapter configuration
  - **type**: One of: `openai`, `langgraph`, `crewai`, `autogen`, `swarm`, `phidata`, `promptflow`, `custom`
  - **config**: Adapter-specific configuration (object)
- **dataset**: Test cases
  - **cases**: Array of test cases (for inline datasets)
    OR
  - **source**: `file`, `generator`, or `huggingface`
  - **path**: Path to dataset file or HuggingFace dataset ID

### Optional Fields

- **metrics**: Array of metric names or configurations
- **timeout**: Maximum execution time in seconds
- **retries**: Number of retry attempts
- **tags**: Array of tags for organization
- **description**: Human-readable description

## Dataset Options

### Inline Cases

```yaml
dataset:
  cases:
    - input: "Question 1"
      expected: "Answer 1"
      context: "Additional context"
      metadata:
        difficulty: "easy"
    - input: "Question 2"
      expected: "Answer 2"
```

### From File

```yaml
dataset:
  source: file
  path: "./data/test_cases.json"
```

### From HuggingFace

```yaml
dataset:
  source: huggingface
  path: "squad"
  split: "validation"
  limit: 100
```

### Generated Dataset

```yaml
dataset:
  source: generator
  generator:
    type: llm
    config:
      num_cases: 10
      prompt: "Generate Q&A pairs about Python"
```

## Built-in Metrics

Common metric names you can use:
- `correctness`
- `latency`
- `faithfulness`
- `answer_relevancy`
- `context_recall`
- `context_precision`
- `exact_match`
- `f1_score`
- `pii_detection`
- `data_leakage`

Custom metrics:
```yaml
metrics:
  - name: custom_metric
    threshold: 0.8
    weight: 1.0
```

## Adapter Types

### OpenAI

```yaml
adapter:
  type: openai
  config:
    model: gpt-4
    temperature: 0.7
    max_tokens: 500
```

### LangGraph

```yaml
adapter:
  type: langgraph
  path: "./my_graph.py"
  config:
    model: gpt-4
```

### CrewAI

```yaml
adapter:
  type: crewai
  config:
    model: gpt-4
    max_turns: 5
```

### Custom

```yaml
adapter:
  type: custom
  path: "./my_adapter.py"
  config:
    # Your custom configuration
```

## Using Templates

Pre-built templates for common scenarios:

```python
from agentunit.nocode import TemplateLibrary

library = TemplateLibrary()

# List available templates
templates = library.list_templates()
for template in templates:
    print(f"{template.name}: {template.description}")

# Get a specific template
template = library.get_template("basic_qa")

# Apply template with customizations
config = library.apply_template(
    "basic_qa",
    name="My Custom Q&A Test",
    adapter={
        "config": {
            "model": "gpt-4",
            "temperature": 0.0,
        }
    },
    timeout=60,
)

# Save customized config
import yaml
with open("my_test.yaml", "w") as f:
    yaml.dump(config, f)
```

### Available Templates

1. **basic_qa**: Basic question-answering scenario
2. **rag_evaluation**: Retrieval-Augmented Generation evaluation
3. **agent_interaction**: Multi-turn agent conversation testing
4. **benchmark_test**: Standardized benchmark evaluation
5. **cost_optimization**: Cost-optimized scenario with fallback models
6. **privacy_test**: Privacy and PII detection testing

## Format Conversion

Convert between YAML, JSON, and Python:

```python
from agentunit.nocode import ConfigConverter, ConversionFormat

converter = ConfigConverter()

# YAML to JSON
converter.convert("scenario.yaml", ConversionFormat.JSON, "scenario.json")

# JSON to Python code
code = converter.convert_to_python("scenario.json")
print(code)

# YAML to Python
python_file = converter.convert("scenario.yaml", ConversionFormat.PYTHON, "scenario.py")
```

## Validation

Validate configurations before use:

```python
from agentunit.nocode import SchemaValidator

validator = SchemaValidator()

# Validate file
result = validator.validate_file("my_scenario.yaml")

if not result.valid:
    print("Validation errors:")
    for error in result.errors:
        print(f"  {error.path}: {error.message}")
else:
    print("✓ Configuration is valid")
    
# Check warnings
if result.warnings:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  {warning}")
```

## Loading Multiple Scenarios

Load all scenarios from a directory:

```python
from agentunit.nocode import ScenarioBuilder

builder = ScenarioBuilder()

# Load all YAML files from directory
scenarios = builder.from_directory(
    "scenarios/",
    pattern="*.yaml"
)

print(f"Loaded {len(scenarios)} scenarios")
```

## Complete Example

### scenario.yaml

```yaml
name: Customer Support Agent Test
description: Testing a customer support chatbot
tags:
  - support
  - chatbot
  - production

adapter:
  type: openai
  config:
    model: gpt-4
    temperature: 0.7
    max_tokens: 500
    timeout: 30

dataset:
  cases:
    - input: "How do I reset my password?"
      expected: "Click on 'Forgot Password' link on the login page"
      metadata:
        category: "account"
        priority: "high"
    
    - input: "What are your business hours?"
      expected: "We're open Monday-Friday, 9 AM - 5 PM EST"
      metadata:
        category: "info"
        priority: "low"
    
    - input: "My account is locked"
      expected: "I'll help you unlock it. Please verify your email"
      metadata:
        category: "account"
        priority: "high"

metrics:
  - correctness
  - latency
  - helpfulness

retries: 2
timeout: 60
```

### Using the scenario

```python
from agentunit.nocode import ScenarioBuilder
from agentunit import run_suite

# Load scenario
builder = ScenarioBuilder()
scenario = builder.from_yaml("scenario.yaml")

# Run tests
results = run_suite([scenario])

# Print results
print(f"Success rate: {results.success_rate:.1%}")
```

## Best Practices

1. **Use Templates**: Start with a template and customize it
2. **Validate Early**: Always validate configs before running
3. **Organize by Directory**: Group related scenarios in folders
4. **Version Control**: Keep scenario configs in git
5. **Use Metadata**: Add metadata to track test categories/priorities
6. **Set Timeouts**: Always specify reasonable timeouts
7. **Test Incrementally**: Start with small datasets, then scale up

## Troubleshooting

### Validation Errors

If you get validation errors, check:
- Required fields are present (name, adapter, dataset)
- Adapter type is one of the supported types
- Dataset has either `cases` or `source` field
- Metric names are valid or properly structured

### Import Errors

If you get import errors:
- Ensure jsonschema is installed: `pip install jsonschema`
- Ensure PyYAML is installed: `pip install pyyaml`

### Adapter Not Found

If adapter creation fails:
- Provide adapter instance directly to builder
- Or use code generation and implement adapter separately
- Check adapter type spelling

## Next Steps

1. Browse available templates: `TemplateLibrary().list_templates()`
2. Create your first scenario from a template
3. Customize it for your use case
4. Validate the configuration
5. Run the scenario and iterate

For more information, see the full API documentation.
