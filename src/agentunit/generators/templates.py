"""Templates for dataset and prompt generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentunit.datasets.base import DatasetCase


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    name: str
    template: str
    variables: list[str]
    description: str | None = None

    def render(self, **kwargs) -> str:
        """Render template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt
        """
        rendered = self.template
        for var in self.variables:
            if var in kwargs:
                rendered = rendered.replace(f"{{{var}}}", str(kwargs[var]))
        return rendered

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Create template from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            variables=data["variables"],
            description=data.get("description"),
        )


@dataclass
class DatasetTemplate:
    """Template for dataset generation."""

    name: str
    domain: str
    task_description: str
    constraints: list[str]
    seed_cases: list[DatasetCase]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "task_description": self.task_description,
            "constraints": self.constraints,
            "seed_cases": [
                {
                    "id": case.id,
                    "query": case.query,
                    "expected_output": case.expected_output,
                    "metadata": case.metadata,
                }
                for case in self.seed_cases
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetTemplate:
        """Create template from dictionary."""
        seed_cases = [
            DatasetCase(
                id=case_data["id"],
                query=case_data["query"],
                expected_output=case_data.get("expected_output"),
                metadata=case_data.get("metadata", {}),
            )
            for case_data in data.get("seed_cases", [])
        ]

        return cls(
            name=data["name"],
            domain=data["domain"],
            task_description=data["task_description"],
            constraints=data.get("constraints", []),
            seed_cases=seed_cases,
            metadata=data.get("metadata", {}),
        )


# Predefined templates
COMMON_TEMPLATES = {
    "customer_service": DatasetTemplate(
        name="customer_service",
        domain="Customer Support",
        task_description="Handle customer inquiries, complaints, and requests professionally",
        constraints=[
            "Be polite and empathetic",
            "Provide accurate information",
            "Escalate when necessary",
        ],
        seed_cases=[
            DatasetCase(
                id="cs_example_1",
                query="I want to return a product I bought last week",
                expected_output="I'd be happy to help you with your return. Can you provide your order number?",
                metadata={"category": "returns"},
            )
        ],
        metadata={"industry": "e-commerce"},
    ),
    "code_review": DatasetTemplate(
        name="code_review",
        domain="Software Development",
        task_description="Review code for bugs, style issues, and improvements",
        constraints=[
            "Identify security vulnerabilities",
            "Suggest performance optimizations",
            "Check code style consistency",
        ],
        seed_cases=[
            DatasetCase(
                id="cr_example_1",
                query="Review this Python function: def add(a, b): return a+b",
                expected_output="The function is correct but could use type hints and docstring",
                metadata={"language": "python"},
            )
        ],
        metadata={"skill_level": "intermediate"},
    ),
    "medical_qa": DatasetTemplate(
        name="medical_qa",
        domain="Healthcare",
        task_description="Answer medical questions with accurate, evidence-based information",
        constraints=[
            "Provide disclaimer about not replacing professional medical advice",
            "Cite medical sources when possible",
            "Be clear about uncertainty",
        ],
        seed_cases=[
            DatasetCase(
                id="mq_example_1",
                query="What are the symptoms of flu?",
                expected_output="Common flu symptoms include fever, cough, sore throat, body aches. Consult a doctor for diagnosis.",
                metadata={"topic": "infectious_disease"},
            )
        ],
        metadata={"requires_disclaimer": True},
    ),
    "data_analysis": DatasetTemplate(
        name="data_analysis",
        domain="Data Science",
        task_description="Analyze datasets and provide insights",
        constraints=[
            "Use statistical methods appropriately",
            "Visualize data when helpful",
            "Explain findings clearly",
        ],
        seed_cases=[
            DatasetCase(
                id="da_example_1",
                query="Analyze this sales data: [100, 150, 120, 200, 180]",
                expected_output="Average sales: 150, showing upward trend with 80% increase from min to max",
                metadata={"data_type": "time_series"},
            )
        ],
        metadata={"tools": ["pandas", "matplotlib"]},
    ),
}


def get_template(name: str) -> DatasetTemplate | None:
    """Get a predefined template by name.

    Args:
        name: Template name

    Returns:
        DatasetTemplate if found, None otherwise
    """
    return COMMON_TEMPLATES.get(name)


def list_templates() -> list[str]:
    """List all available template names.

    Returns:
        List of template names
    """
    return list(COMMON_TEMPLATES.keys())


__all__ = [
    "COMMON_TEMPLATES",
    "DatasetTemplate",
    "PromptTemplate",
    "get_template",
    "list_templates",
]
