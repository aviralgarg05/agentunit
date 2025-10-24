# AgentUnit JSON Schemas

This directory contains JSON Schema definitions for AgentUnit artifacts and configuration files.

## Schemas

### `scenario.json`
Defines the structure for YAML scenario configuration files.

**Usage:**
```yaml
# Reference in your scenario YAML
# $schema: https://agentunit.dev/schemas/scenario.json

name: my-scenario
adapter:
  type: langraph
  path: ./my_agent.py
dataset:
  cases:
    - input: "test input"
      expected: "expected output"
metrics:
  - accuracy
  - faithfulness
```

**Validation:**
```bash
# With yamale
pip install yamale
yamale -s src/agentunit/schemas/scenario.json scenarios/

# With check-jsonschema
pip install check-jsonschema
check-jsonschema --schemafile src/agentunit/schemas/scenario.json scenarios/*.yaml
```

### `result.json`
Defines the structure for evaluation result artifacts produced by AgentUnit.

**Usage:**
Result files are automatically generated when using `--json` flag:
```bash
agentunit eval suite.py --json results/output.json
```

**Validation:**
```bash
check-jsonschema --schemafile src/agentunit/schemas/result.json results/*.json
```

**Applications:**
- CI/CD result parsing and validation
- Dashboard and visualization tools
- Historical result comparison
- Regression detection automation

### `metric.json`
Defines the structure for custom metric definitions.

**Usage:**
Define custom metrics in JSON format for dynamic loading:
```json
{
  "name": "custom_accuracy",
  "type": "deterministic",
  "inputs": {
    "requires_actual": true,
    "requires_expected": true
  },
  "implementation": {
    "module": "my_metrics",
    "class": "CustomAccuracyMetric"
  }
}
```

**Validation:**
```bash
check-jsonschema --schemafile src/agentunit/schemas/metric.json metrics/*.json
```

### `dataset.json`
Defines the structure for standalone dataset files.

**Usage:**
Create reusable datasets in JSON format:
```json
{
  "name": "rag-benchmark-v1",
  "description": "RAG accuracy benchmark dataset",
  "cases": [
    {
      "input": {
        "query": "What is the capital of France?",
        "context": ["France is a country in Europe. Paris is its capital."]
      },
      "expected": "Paris"
    }
  ]
}
```

**Validation:**
```bash
check-jsonschema --schemafile src/agentunit/schemas/dataset.json datasets/*.json
```

## IDE Integration

### VS Code
Add to your workspace `.vscode/settings.json`:
```json
{
  "yaml.schemas": {
    "./src/agentunit/schemas/scenario.json": ["scenarios/*.yaml", "tests/**/*scenario*.yaml"]
  },
  "json.schemas": [
    {
      "fileMatch": ["results/*.json"],
      "url": "./src/agentunit/schemas/result.json"
    },
    {
      "fileMatch": ["datasets/*.json"],
      "url": "./src/agentunit/schemas/dataset.json"
    },
    {
      "fileMatch": ["metrics/*.json"],
      "url": "./src/agentunit/schemas/metric.json"
    }
  ]
}
```

### PyCharm / IntelliJ IDEA
1. Go to Settings > Languages & Frameworks > Schemas and DTDs > JSON Schema Mappings
2. Add mappings for each schema file to corresponding file patterns

## Schema Versions

All schemas follow the project version. Breaking changes to schemas will result in a major version bump.

Current version: `0.6.0`

## Contributing

When adding new fields to schemas:
1. Add the field to the appropriate schema file
2. Update this README with usage examples
3. Add validation tests in `tests/test_schemas.py`
4. Update CHANGELOG.md with the schema changes
5. Consider backward compatibility for existing files

## References

- [JSON Schema Documentation](https://json-schema.org/)
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/)
- [YAML Schema Validation](https://github.com/23andMe/Yamale)
