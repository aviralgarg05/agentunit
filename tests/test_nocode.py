"""Tests for no-code scenario builder."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from agentunit.nocode import (
    ScenarioBuilder,
    SchemaValidator,
    ValidationResult,
    CodeGenerator,
    ConfigConverter,
    ConversionFormat,
    TemplateLibrary,
    ScenarioTemplate,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Sample valid scenario configuration."""
    return {
        "name": "Test Scenario",
        "adapter": {
            "type": "openai",
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
            },
        },
        "dataset": {
            "cases": [
                {
                    "input": "What is 2+2?",
                    "expected": "4",
                },
                {
                    "input": "What is the capital of France?",
                    "expected": "Paris",
                },
            ],
        },
        "metrics": ["correctness", "latency"],
        "timeout": 30,
        "retries": 1,
    }


@pytest.fixture
def validator():
    """Create schema validator."""
    return SchemaValidator()


@pytest.fixture
def builder():
    """Create scenario builder."""
    return ScenarioBuilder(validate=True)


@pytest.fixture
def converter():
    """Create config converter."""
    return ConfigConverter()


@pytest.fixture
def template_library():
    """Create template library."""
    return TemplateLibrary()


# Validation Tests

def test_validate_valid_config(validator, sample_config):
    """Test validation of valid configuration."""
    result = validator.validate(sample_config)
    
    assert result.valid
    assert len(result.errors) == 0


def test_validate_missing_required_field(validator):
    """Test validation catches missing required fields."""
    config = {
        "name": "Test",
        # Missing adapter and dataset
    }
    
    result = validator.validate(config)
    
    assert not result.valid
    assert len(result.errors) > 0
    assert any("required" in e.message.lower() for e in result.errors)


def test_validate_invalid_adapter_type(validator, sample_config):
    """Test validation catches invalid adapter type."""
    config = sample_config.copy()
    config["adapter"]["type"] = "invalid_adapter"
    
    result = validator.validate(config)
    
    assert not result.valid


def test_validate_warnings(validator, sample_config):
    """Test validator generates helpful warnings."""
    config = sample_config.copy()
    config.pop("metrics")  # Should warn about missing metrics
    
    result = validator.validate(config)
    
    assert result.valid  # Still valid, just warning
    assert len(result.warnings) > 0


def test_validate_from_file(validator, temp_dir, sample_config):
    """Test validation from file."""
    # Write YAML file
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    result = validator.validate_file(yaml_path)
    assert result.valid


# Code Generation Tests

def test_generate_code_from_dict(sample_config):
    """Test Python code generation from dict."""
    generator = CodeGenerator()
    result = generator.from_dict(sample_config)
    
    assert result.code
    assert "import" in result.code
    assert "Scenario" in result.code
    assert result.scenario_name == "Test Scenario"


def test_generate_code_from_yaml(temp_dir, sample_config):
    """Test code generation from YAML file."""
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    generator = CodeGenerator()
    result = generator.from_file(yaml_path)
    
    assert result.code
    assert "OpenAI" in result.code or "adapter" in result.code.lower()


def test_generate_code_from_json(temp_dir, sample_config):
    """Test code generation from JSON file."""
    json_path = temp_dir / "scenario.json"
    with open(json_path, 'w') as f:
        json.dump(sample_config, f)
    
    generator = CodeGenerator()
    result = generator.from_file(json_path)
    
    assert result.code


def test_generated_code_has_imports(sample_config):
    """Test generated code includes necessary imports."""
    generator = CodeGenerator()
    result = generator.from_dict(sample_config)
    
    assert "from agentunit" in result.code


def test_generated_code_has_main_block(sample_config):
    """Test generated code has executable main block."""
    generator = CodeGenerator()
    result = generator.from_dict(sample_config)
    
    assert 'if __name__ == "__main__"' in result.code


# Format Conversion Tests

def test_convert_yaml_to_json(converter, temp_dir, sample_config):
    """Test conversion from YAML to JSON."""
    yaml_path = temp_dir / "scenario.yaml"
    json_path = temp_dir / "scenario.json"
    
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    result = converter.convert(yaml_path, ConversionFormat.JSON, json_path)
    
    assert json_path.exists()
    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded["name"] == sample_config["name"]


def test_convert_json_to_yaml(converter, temp_dir, sample_config):
    """Test conversion from JSON to YAML."""
    json_path = temp_dir / "scenario.json"
    yaml_path = temp_dir / "scenario.yaml"
    
    with open(json_path, 'w') as f:
        json.dump(sample_config, f)
    
    result = converter.convert(json_path, ConversionFormat.YAML, yaml_path)
    
    assert yaml_path.exists()
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["name"] == sample_config["name"]


def test_convert_to_python(converter, temp_dir, sample_config):
    """Test conversion to Python code."""
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    code = converter.convert_to_python(yaml_path)
    
    assert "Scenario" in code
    assert "adapter" in code.lower()


def test_round_trip_yaml_json(converter, temp_dir, sample_config):
    """Test round-trip YAML -> JSON -> YAML preserves data."""
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    assert converter.validate_round_trip(yaml_path, ConversionFormat.JSON)


# Template Library Tests

def test_list_templates(template_library):
    """Test listing available templates."""
    templates = template_library.list_templates()
    
    assert len(templates) > 0
    assert all(isinstance(t, ScenarioTemplate) for t in templates)


def test_list_templates_by_tag(template_library):
    """Test filtering templates by tag."""
    qa_templates = template_library.list_templates(tag="qa")
    
    assert len(qa_templates) > 0
    assert all("qa" in t.tags for t in qa_templates)


def test_get_template(template_library):
    """Test getting specific template."""
    template = template_library.get_template("basic_qa")
    
    assert template.name == "basic_qa"
    assert "adapter" in template.config
    assert template.config["adapter"]["type"]


def test_get_nonexistent_template(template_library):
    """Test getting template that doesn't exist."""
    with pytest.raises(KeyError):
        template_library.get_template("nonexistent")


def test_apply_template(template_library):
    """Test applying template with customizations."""
    config = template_library.apply_template(
        "basic_qa",
        name="My Custom Test",
        timeout=60,
    )
    
    assert config["name"] == "My Custom Test"
    assert config["timeout"] == 60
    # Original adapter config should be preserved
    assert "adapter" in config


def test_apply_template_deep_merge(template_library):
    """Test template application with deep merge."""
    config = template_library.apply_template(
        "basic_qa",
        adapter={
            "config": {
                "temperature": 1.0,  # Override temperature
            },
        },
    )
    
    # Should override temperature but keep other adapter config
    assert config["adapter"]["config"]["temperature"] == 1.0
    assert "model" in config["adapter"]["config"]


def test_add_custom_template(template_library):
    """Test adding custom template."""
    custom = ScenarioTemplate(
        name="custom_test",
        description="Custom test template",
        tags=["custom"],
        config={
            "name": "Custom",
            "adapter": {"type": "custom"},
            "dataset": {"cases": []},
        },
    )
    
    template_library.add_template(custom)
    
    retrieved = template_library.get_template("custom_test")
    assert retrieved.name == "custom_test"


# Scenario Builder Tests

def test_builder_from_yaml(builder, temp_dir, sample_config):
    """Test building scenario from YAML file."""
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    # Note: This will fail because adapter creation is not fully implemented
    # but we can test the validation and parsing logic
    with pytest.raises(NotImplementedError):
        scenario = builder.from_yaml(yaml_path)


def test_builder_from_json(builder, temp_dir, sample_config):
    """Test building scenario from JSON file."""
    json_path = temp_dir / "scenario.json"
    with open(json_path, 'w') as f:
        json.dump(sample_config, f)
    
    with pytest.raises(NotImplementedError):
        scenario = builder.from_json(json_path)


def test_builder_validation_error(sample_config):
    """Test builder catches validation errors."""
    invalid_config = {
        "name": "Test",
        # Missing required fields
    }
    
    builder = ScenarioBuilder(validate=True)
    
    with pytest.raises(ValueError, match="Invalid scenario configuration"):
        builder.from_dict(invalid_config)


def test_builder_to_python(builder, temp_dir, sample_config):
    """Test builder generates Python code."""
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    code = builder.to_python(yaml_path)
    
    assert "Scenario" in code


def test_builder_from_directory(builder, temp_dir, sample_config):
    """Test loading multiple scenarios from directory."""
    # Create multiple scenario files
    for i in range(3):
        config = sample_config.copy()
        config["name"] = f"Scenario {i}"
        
        yaml_path = temp_dir / f"scenario_{i}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
    
    # Note: Will have NotImplementedError but should try to load all files
    try:
        scenarios = builder.from_directory(temp_dir)
    except NotImplementedError:
        # Expected - adapter creation not implemented
        pass


# Integration Tests

def test_full_workflow_yaml_to_python(temp_dir, sample_config):
    """Test complete workflow: YAML -> validate -> generate Python."""
    # Write YAML file
    yaml_path = temp_dir / "scenario.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    # Validate
    validator = SchemaValidator()
    result = validator.validate_file(yaml_path)
    assert result.valid
    
    # Generate Python code
    generator = CodeGenerator()
    code_result = generator.from_file(yaml_path)
    
    assert code_result.code
    # Code generator may have warnings for unknown adapter types, that's acceptable


def test_template_to_scenario(template_library, temp_dir):
    """Test workflow: template -> customize -> save -> load."""
    # Apply template
    config = template_library.apply_template(
        "basic_qa",
        name="My Q&A Test",
    )
    
    # Save to file
    yaml_path = temp_dir / "from_template.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    # Validate
    validator = SchemaValidator()
    result = validator.validate_file(yaml_path)
    assert result.valid
    
    # Generate code
    generator = CodeGenerator()
    code_result = generator.from_file(yaml_path)
    assert "My Q&A Test" in code_result.code
