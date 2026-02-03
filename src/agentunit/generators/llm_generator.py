"""LLM-powered dataset generators using Llama and OpenAI models."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from agentunit.datasets.base import DatasetCase, DatasetSource


logger = logging.getLogger(__name__)

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from huggingface_hub import InferenceClient

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


@dataclass
class GeneratorConfig:
    """Configuration for dataset generation."""

    num_cases: int = 10
    temperature: float = 0.8
    max_tokens: int = 2048
    diversity_penalty: float = 0.5
    include_edge_cases: bool = True
    edge_case_ratio: float = 0.3


class LlamaDatasetGenerator:
    """Generate synthetic datasets using Llama models via HuggingFace Inference API."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_token: str | None = None,
        config: GeneratorConfig | None = None,
    ):
        """Initialize Llama dataset generator.

        Args:
            model: Llama model name on HuggingFace
            api_token: HuggingFace API token
            config: Generation configuration
        """
        if not HAS_HF_HUB:
            msg = (
                "huggingface_hub required for LlamaDatasetGenerator. "
                "Install with: pip install huggingface_hub"
            )
            raise ImportError(msg)

        self.model = model
        self.client = InferenceClient(token=api_token)
        self.config = config or GeneratorConfig()

    def _create_generation_prompt(self, domain: str, task_description: str) -> str:
        """Create a prompt for synthetic dataset generation."""
        return f"""You are an expert test case generator for AI agent evaluation.

Domain: {domain}
Task: {task_description}

Generate {self.config.num_cases} diverse test cases for evaluating an AI agent on this task.
Each test case should include:
1. A unique query/question
2. Expected output or behavior
3. Difficulty level (easy/medium/hard)
4. Any relevant metadata

Format your response as a JSON array of test cases:
[
  {{
    "id": "case_1",
    "query": "...",
    "expected_output": "...",
    "difficulty": "easy|medium|hard",
    "metadata": {{}}
  }},
  ...
]

Make sure to include:
- {int(self.config.num_cases * (1 - self.config.edge_case_ratio))} typical cases
- {int(self.config.num_cases * self.config.edge_case_ratio)} edge cases (boundary conditions, ambiguous inputs, adversarial examples)

Ensure diversity in query formulation and complexity."""

    async def generate(
        self, domain: str, task_description: str, constraints: list[str] | None = None
    ) -> DatasetSource:
        """
        Generate a dataset of synthetic test cases for a given domain and task.
        
        Parameters:
            domain (str): Domain name for the generated cases (e.g., "customer service").
            task_description (str): Description of the task the cases should exercise.
            constraints (list[str] | None): Optional list of additional constraints to apply when generating cases.
        
        Returns:
            DatasetSource: A dataset containing generated DatasetCase objects. Each case includes an `id`, `query`, optional `expected_output`, and `metadata` containing at least `difficulty`, `generated` (True), and `domain`.
        
        Raises:
            json.JSONDecodeError: If the model response cannot be parsed as JSON.
        """
        prompt = self._create_generation_prompt(domain, task_description)

        if constraints:
            prompt += "\n\nAdditional constraints:\n" + "\n".join(f"- {c}" for c in constraints)

        logger.debug("Llama generated prompt:\n%s", prompt)

        # Generate with Llama
        response = await asyncio.to_thread(
            self.client.text_generation,
            prompt=prompt,
            model=self.model,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            return_full_text=False,
        )
        logger.debug("Llama raw response:\n%s", response)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            cases_data = json.loads(response_text)

            cases: list[DatasetCase] = []
            for case_data in cases_data:
                case = DatasetCase(
                    id=case_data.get("id", f"generated_{len(cases)}"),
                    query=case_data["query"],
                    expected_output=case_data.get("expected_output"),
                    metadata={
                        "difficulty": case_data.get("difficulty", "medium"),
                        "generated": True,
                        "domain": domain,
                        **case_data.get("metadata", {}),
                    },
                )
                cases.append(case)

            return DatasetSource.from_list(
                cases, name=f"llama_generated_{domain.replace(' ', '_')}"
            )

        except json.JSONDecodeError:
            # msg = f"Failed to parse generated dataset: {e}\nResponse: {response}"
            logger.error(
                "Failed to parse Llama response JSON. Raw response:\n%s",
                response,
                exc_info=True,
            )
            raise

    def generate_sync(
        self, domain: str, task_description: str, constraints: list[str] | None = None
    ) -> DatasetSource:
        """Synchronous version of generate."""
        return asyncio.run(self.generate(domain, task_description, constraints))


class OpenAIDatasetGenerator:
    """Generate synthetic datasets using OpenAI models (GPT-4, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        config: GeneratorConfig | None = None,
    ):
        """Initialize OpenAI dataset generator.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            config: Generation configuration
        """
        if not HAS_OPENAI:
            msg = "openai required for OpenAIDatasetGenerator. Install with: pip install openai"
            raise ImportError(msg)

        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.config = config or GeneratorConfig()

    def _create_generation_prompt(self, domain: str, task_description: str) -> str:
        """Create a prompt for synthetic dataset generation."""
        return f"""You are an expert test case generator for AI agent evaluation.

Domain: {domain}
Task: {task_description}

Generate {self.config.num_cases} diverse test cases for evaluating an AI agent on this task.
Each test case should include:
1. A unique query/question
2. Expected output or behavior
3. Difficulty level (easy/medium/hard)
4. Any relevant metadata

Format your response as a JSON array only, with no additional text:
[
  {{
    "id": "case_1",
    "query": "...",
    "expected_output": "...",
    "difficulty": "easy|medium|hard",
    "metadata": {{}}
  }}
]

Distribution:
- {int(self.config.num_cases * (1 - self.config.edge_case_ratio))} typical cases
- {int(self.config.num_cases * self.config.edge_case_ratio)} edge cases (boundary conditions, ambiguous inputs, adversarial examples)

Ensure diversity in query formulation and complexity."""

    async def generate(
        self,
        domain: str,
        task_description: str,
        constraints: list[str] | None = None,
        seed_examples: list[dict[str, Any]] | None = None,
    ) -> DatasetSource:
        """
        Generate a synthetic dataset of test cases for a given domain and task.
        
        Constructs a chat prompt from the domain, task description, optional constraints, and optional seed examples, sends it to the configured OpenAI chat model, parses the model's JSON array response into DatasetCase objects, and returns a DatasetSource containing those cases.
        
        Parameters:
            domain (str): Domain or subject area for which to generate test cases.
            task_description (str): Description of the task or problem the generated cases should exercise.
            constraints (list[str] | None): Optional list of additional constraints to include in the prompt.
            seed_examples (list[dict[str, Any]] | None): Optional list of example cases to guide generation; included verbatim in the prompt.
        
        Returns:
            DatasetSource: A dataset source containing generated DatasetCase entries with metadata (including difficulty, generated=True, and domain).
        
        Raises:
            json.JSONDecodeError: If the model's response cannot be parsed as a JSON array of cases.
        """
        messages = [
            {"role": "system", "content": "You are an expert test case generator."},
            {"role": "user", "content": self._create_generation_prompt(domain, task_description)},
        ]

        if constraints:
            messages[1]["content"] += "\n\nAdditional constraints:\n" + "\n".join(
                f"- {c}" for c in constraints
            )

        if seed_examples:
            messages[1]["content"] += f"\n\nSeed examples:\n{json.dumps(seed_examples, indent=2)}"

        logger.debug("OpenAI generated prompt (messages):\n%s", json.dumps(messages, indent=2))
        # Generate with GPT
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"} if "gpt-4" in self.model else None,
        )

        response_text = response.choices[0].message.content
        logger.debug("OpenAI raw response text:\n%s", response_text)

        # Parse JSON response
        try:
            # Handle potential wrapping
            if not response_text.strip().startswith("["):
                # Try to extract JSON array
                import re

                json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)

            cases_data = json.loads(response_text)

            cases: list[DatasetCase] = []
            for case_data in cases_data:
                case = DatasetCase(
                    id=case_data.get("id", f"generated_{len(cases)}"),
                    query=case_data["query"],
                    expected_output=case_data.get("expected_output"),
                    metadata={
                        "difficulty": case_data.get("difficulty", "medium"),
                        "generated": True,
                        "domain": domain,
                        **case_data.get("metadata", {}),
                    },
                )
                cases.append(case)

            return DatasetSource.from_list(
                cases, name=f"openai_generated_{domain.replace(' ', '_')}"
            )

        except json.JSONDecodeError:
            logger.error(
                "Failed to parse OpenAI response JSON. Raw response:\n%s",
                response_text,
                exc_info=True,
            )
        raise

    def generate_sync(
        self,
        domain: str,
        task_description: str,
        constraints: list[str] | None = None,
        seed_examples: list[dict[str, Any]] | None = None,
    ) -> DatasetSource:
        """Synchronous version of generate."""
        return asyncio.run(self.generate(domain, task_description, constraints, seed_examples))


__all__ = [
    "GeneratorConfig",
    "LlamaDatasetGenerator",
    "OpenAIDatasetGenerator",
]