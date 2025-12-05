"""Pre-built scenario templates for common use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScenarioTemplate:
    """A reusable scenario template.

    Templates provide pre-configured scenarios for common testing patterns,
    reducing boilerplate and ensuring best practices.
    """

    name: str
    description: str
    config: dict[str, Any]
    tags: list[str]


class TemplateLibrary:
    """Library of pre-built scenario templates.

    Provides templates for common agent testing patterns like Q&A,
    RAG evaluation, agent interactions, and benchmarks.

    Examples:
        >>> library = TemplateLibrary()
        >>>
        >>> # List available templates
        >>> templates = library.list_templates()
        >>> for template in templates:
        ...     print(f"{template.name}: {template.description}")
        >>>
        >>> # Get a specific template
        >>> template = library.get_template("basic_qa")
        >>>
        >>> # Apply template with customizations
        >>> config = library.apply_template(
        ...     "basic_qa",
        ...     name="My Q&A Test",
        ...     adapter={"type": "openai", "config": {"model": "gpt-4"}},
        ... )
    """

    def __init__(self):
        """Initialize template library."""
        self._templates = self._load_builtin_templates()

    def list_templates(self, tag: str | None = None) -> list[ScenarioTemplate]:
        """List all available templates.

        Args:
            tag: Optional tag to filter templates

        Returns:
            List of scenario templates
        """
        if tag:
            return [t for t in self._templates.values() if tag in t.tags]
        return list(self._templates.values())

    def get_template(self, name: str) -> ScenarioTemplate:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            ScenarioTemplate

        Raises:
            KeyError: If template not found
        """
        if name not in self._templates:
            available = ", ".join(self._templates.keys())
            msg = f"Template '{name}' not found. Available: {available}"
            raise KeyError(msg)

        return self._templates[name]

    def apply_template(
        self,
        template_name: str,
        **customizations: Any,
    ) -> dict[str, Any]:
        """Apply a template with customizations.

        Args:
            template_name: Template name
            **customizations: Fields to override in the template

        Returns:
            Customized configuration dictionary
        """
        template = self.get_template(template_name)
        config = template.config.copy()

        # Deep merge customizations
        self._deep_merge(config, customizations)

        return config

    def add_template(self, template: ScenarioTemplate) -> None:
        """Add a custom template to the library.

        Args:
            template: Template to add
        """
        self._templates[template.name] = template

    def _load_builtin_templates(self) -> dict[str, ScenarioTemplate]:
        """Load built-in templates."""
        return {
            "basic_qa": ScenarioTemplate(
                name="basic_qa",
                description="Basic question-answering scenario",
                tags=["qa", "basic", "simple"],
                config={
                    "name": "Basic Q&A Test",
                    "adapter": {
                        "type": "openai",
                        "config": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.0,
                        },
                    },
                    "dataset": {
                        "cases": [
                            {
                                "input": "What is 2+2?",
                                "expected": "4",
                            },
                        ],
                    },
                    "metrics": ["correctness", "latency"],
                    "timeout": 30,
                },
            ),
            "rag_evaluation": ScenarioTemplate(
                name="rag_evaluation",
                description="Retrieval-Augmented Generation evaluation",
                tags=["rag", "retrieval", "context"],
                config={
                    "name": "RAG Evaluation",
                    "adapter": {
                        "type": "langgraph",
                        "config": {
                            "model": "gpt-4",
                            "temperature": 0.0,
                        },
                    },
                    "dataset": {
                        "cases": [
                            {
                                "input": "What is the main topic of the document?",
                                "expected": "Answer based on context",
                                "context": "Context will be retrieved automatically",
                            },
                        ],
                    },
                    "metrics": [
                        "faithfulness",
                        "answer_relevancy",
                        "context_recall",
                        "latency",
                    ],
                    "timeout": 60,
                },
            ),
            "agent_interaction": ScenarioTemplate(
                name="agent_interaction",
                description="Multi-turn agent conversation testing",
                tags=["agent", "conversation", "multi-turn"],
                config={
                    "name": "Agent Interaction Test",
                    "adapter": {
                        "type": "autogen",
                        "config": {
                            "model": "gpt-4",
                            "max_turns": 5,
                        },
                    },
                    "dataset": {
                        "cases": [
                            {
                                "input": "Help me plan a trip to Paris",
                                "expected": "Should provide travel recommendations",
                                "metadata": {
                                    "conversation_depth": "multi-turn",
                                },
                            },
                        ],
                    },
                    "metrics": [
                        "coherence",
                        "helpfulness",
                        "latency",
                    ],
                    "retries": 1,
                    "timeout": 120,
                },
            ),
            "benchmark_test": ScenarioTemplate(
                name="benchmark_test",
                description="Standardized benchmark evaluation",
                tags=["benchmark", "evaluation", "dataset"],
                config={
                    "name": "Benchmark Evaluation",
                    "adapter": {
                        "type": "custom",
                        "path": "adapters/my_adapter.py",
                    },
                    "dataset": {
                        "source": "huggingface",
                        "path": "squad",
                        "split": "validation",
                        "limit": 100,
                    },
                    "metrics": [
                        "exact_match",
                        "f1_score",
                        "latency",
                    ],
                    "timeout": 60,
                },
            ),
            "cost_optimization": ScenarioTemplate(
                name="cost_optimization",
                description="Cost-optimized scenario with fallback models",
                tags=["cost", "optimization", "fallback"],
                config={
                    "name": "Cost-Optimized Test",
                    "adapter": {
                        "type": "openai",
                        "config": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.7,
                            "max_tokens": 150,
                        },
                    },
                    "dataset": {
                        "cases": [
                            {
                                "input": "Sample query",
                                "expected": "Sample response",
                            },
                        ],
                    },
                    "metrics": [
                        "correctness",
                        "cost",
                        "latency",
                    ],
                    "retries": 2,
                    "timeout": 30,
                },
            ),
            "privacy_test": ScenarioTemplate(
                name="privacy_test",
                description="Privacy and PII detection testing",
                tags=["privacy", "pii", "security"],
                config={
                    "name": "Privacy Test",
                    "adapter": {
                        "type": "openai",
                        "config": {
                            "model": "gpt-4",
                            "temperature": 0.0,
                        },
                    },
                    "dataset": {
                        "cases": [
                            {
                                "input": "My email is john@example.com",
                                "expected": "Response without revealing PII",
                                "metadata": {
                                    "contains_pii": True,
                                },
                            },
                        ],
                    },
                    "metrics": [
                        "pii_detection",
                        "data_leakage",
                        "correctness",
                    ],
                    "timeout": 45,
                },
            ),
        }

    def _deep_merge(self, base: dict[str, Any], update: dict[str, Any]) -> None:
        """Deep merge update dictionary into base.

        Args:
            base: Base dictionary to merge into (modified in place)
            update: Update dictionary with new values
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
