"""Tests for custom dataset generators."""

import pytest

from agentunit import generators
from agentunit.datasets.base import DatasetCase
from agentunit.generators.augmentation import (
    AdversarialAugmenter,
    DistributionShifter,
    EdgeCaseGenerator,
    NoiseAugmenter,
)
from agentunit.generators.llm_generator import GeneratorConfig
from agentunit.generators.templates import (
    DatasetTemplate,
    PromptTemplate,
    get_template,
    list_templates,
)


def test_generator_imports():
    """Test that generator module can be imported."""
    assert hasattr(generators, "__all__")
    assert "LlamaDatasetGenerator" in generators.__all__
    assert "OpenAIDatasetGenerator" in generators.__all__


def test_adversarial_augmenter():
    """Test AdversarialAugmenter."""
    augmenter = AdversarialAugmenter(techniques=["jailbreak", "prompt_injection"])

    case = DatasetCase(
        id="test_1",
        query="What is the weather today?",
        expected_output="I'll check the weather for you."
    )

    adversarial_cases = augmenter.augment(case)

    assert len(adversarial_cases) == 2
    assert any("jailbreak" in c.metadata.get("augmentation", "") for c in adversarial_cases)
    assert any("injection" in c.metadata.get("augmentation", "") for c in adversarial_cases)


def test_noise_augmenter():
    """Test NoiseAugmenter."""
    augmenter = NoiseAugmenter(typo_rate=0.1, char_swap_rate=0.05)

    case = DatasetCase(
        id="test_2",
        query="This is a test query with multiple words",
        expected_output="Response"
    )

    noisy_cases = augmenter.augment(case, num_variants=3)

    assert len(noisy_cases) == 3
    assert all(c.metadata.get("augmentation") == "noise" for c in noisy_cases)
    assert all(c.id.startswith("test_2_noise_") for c in noisy_cases)


def test_edge_case_generator():
    """Test EdgeCaseGenerator."""
    generator = EdgeCaseGenerator()

    case = DatasetCase(
        id="test_3",
        query="Normal query",
        expected_output="Normal response"
    )

    edge_cases = generator.generate_edge_cases(case)

    assert len(edge_cases) > 0
    assert any(c.metadata.get("edge_case") == "empty_input" for c in edge_cases)
    assert any(c.metadata.get("edge_case") == "long_input" for c in edge_cases)
    assert any(c.metadata.get("edge_case") == "special_characters" for c in edge_cases)


def test_distribution_shifter():
    """Test DistributionShifter."""
    shifter = DistributionShifter()

    cases = [
        DatasetCase(id="case_1", query="What is AI?", expected_output="AI is..."),
        DatasetCase(id="case_2", query="Explain ML", expected_output="ML is...")
    ]

    # Test temporal shift
    temporal_cases = shifter.apply_shift(cases, "temporal", {"time_marker": "in 2050"})
    assert len(temporal_cases) == 2
    assert all("in 2050" in c.query for c in temporal_cases)

    # Test style shift
    style_cases = shifter.apply_shift(cases, "style", {"target_style": "informal"})
    assert len(style_cases) == 2
    assert all(c.metadata.get("shift_type") == "style" for c in style_cases)


def test_prompt_template():
    """Test PromptTemplate."""
    template = PromptTemplate(
        name="test_template",
        template="Hello {name}, your task is {task}",
        variables=["name", "task"]
    )

    rendered = template.render(name="Alice", task="coding")
    assert rendered == "Hello Alice, your task is coding"


def test_dataset_template():
    """Test DatasetTemplate."""
    # Test predefined templates
    templates = list_templates()
    assert len(templates) > 0
    assert "customer_service" in templates

    # Get a template
    template = get_template("customer_service")
    assert template is not None
    assert template.domain == "Customer Support"
    assert len(template.seed_cases) > 0

    # Test serialization
    data = template.to_dict()
    assert "name" in data
    assert "domain" in data

    # Test deserialization
    restored = DatasetTemplate.from_dict(data)
    assert restored.name == template.name
    assert restored.domain == template.domain


def test_generator_config():
    """Test GeneratorConfig."""
    config = GeneratorConfig(
        num_cases=20,
        temperature=0.9,
        edge_case_ratio=0.4
    )

    assert config.num_cases == 20
    assert config.temperature == pytest.approx(0.9)
    assert config.edge_case_ratio == pytest.approx(0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
