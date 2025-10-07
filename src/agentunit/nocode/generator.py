"""Python code generation from declarative configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import json


@dataclass
class GeneratedCode:
    """Result of code generation.
    
    Attributes:
        code: Generated Python code
        imports: Required import statements
        scenario_name: Name of generated scenario
        warnings: Any warnings about generation
    """
    
    code: str
    imports: List[str]
    scenario_name: str
    warnings: List[str]


class CodeGenerator:
    """Generate Python code from declarative scenario configurations.
    
    Converts YAML/JSON scenario definitions into executable Python code
    that can be run with AgentUnit.
    
    Examples:
        >>> generator = CodeGenerator()
        >>> 
        >>> # Generate from YAML file
        >>> result = generator.from_file("scenario.yaml")
        >>> print(result.code)
        >>> 
        >>> # Save generated code
        >>> output_path = Path("generated_scenario.py")
        >>> output_path.write_text(result.code)
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize code generator.
        
        Args:
            template_dir: Directory containing code templates (uses defaults if None)
        """
        self.template_dir = template_dir or (Path(__file__).parent / "templates")
    
    def from_dict(self, config: Dict[str, Any]) -> GeneratedCode:
        """Generate Python code from configuration dictionary.
        
        Args:
            config: Scenario configuration
        
        Returns:
            GeneratedCode with Python implementation
        """
        imports = [
            "from agentunit import Scenario, run_suite",
            "from agentunit.datasets.base import DatasetSource, DatasetCase",
        ]
        warnings = []
        
        scenario_name = config.get("name", "scenario")
        
        # Generate adapter code
        adapter_code = self._generate_adapter(config["adapter"], imports, warnings)
        
        # Generate dataset code
        dataset_code = self._generate_dataset(config["dataset"], imports, warnings)
        
        # Generate metrics code
        metrics_code = self._generate_metrics(config.get("metrics", []), imports, warnings)
        
        # Generate scenario instantiation
        scenario_code = self._generate_scenario(
            scenario_name,
            adapter_code,
            dataset_code,
            metrics_code,
            config,
        )
        
        # Combine into complete code
        code_lines = [
            '"""Generated AgentUnit scenario."""',
            "",
        ]
        
        # Add imports
        code_lines.extend(sorted(set(imports)))
        code_lines.append("")
        code_lines.append("")
        
        # Add scenario code
        code_lines.append(scenario_code)
        code_lines.append("")
        code_lines.append("")
        
        # Add main block
        code_lines.extend([
            'if __name__ == "__main__":',
            "    # Run the scenario",
            f"    result = run_suite([{scenario_name}])",
            '    print(f"Passed: {result.success_rate:.1%}")',
        ])
        
        code = "\n".join(code_lines)
        
        return GeneratedCode(
            code=code,
            imports=list(set(imports)),
            scenario_name=scenario_name,
            warnings=warnings,
        )
    
    def from_file(self, filepath: str | Path) -> GeneratedCode:
        """Generate Python code from YAML/JSON file.
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            GeneratedCode with Python implementation
        """
        filepath = Path(filepath)
        
        with open(filepath) as f:
            if filepath.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        return self.from_dict(config)
    
    def _generate_adapter(
        self,
        adapter_config: Dict[str, Any],
        imports: List[str],
        warnings: List[str],
    ) -> str:
        """Generate adapter configuration code."""
        adapter_type = adapter_config["type"]
        
        if adapter_type == "langgraph":
            imports.append("from agentunit.adapters import LangGraphAdapter")
            path = adapter_config.get("path", "")
            return f'LangGraphAdapter("{path}")'
        
        elif adapter_type == "crewai":
            imports.append("from agentunit.adapters import CrewAIAdapter")
            return "CrewAIAdapter(crew)"  # Crew needs to be defined separately
        
        elif adapter_type == "autogen":
            imports.append("from agentunit.adapters import AutoGenAdapter")
            config = adapter_config.get("config", {})
            config_str = self._format_dict(config)
            return f"AutoGenAdapter({config_str})"
        
        elif adapter_type == "swarm":
            imports.append("from agentunit.adapters import SwarmAdapter")
            config = adapter_config.get("config", {})
            config_str = self._format_dict(config)
            return f"SwarmAdapter({config_str})"
        
        else:
            warnings.append(f"Unknown adapter type '{adapter_type}', using generic adapter")
            return "adapter  # Define your custom adapter"
    
    def _generate_dataset(
        self,
        dataset_config: Dict[str, Any],
        imports: List[str],
        warnings: List[str],
    ) -> str:
        """Generate dataset code."""
        if "cases" in dataset_config:
            # Inline dataset
            cases = dataset_config["cases"]
            cases_code = "[\n"
            for case in cases:
                input_val = self._format_value(case["input"])
                expected_val = self._format_value(case["expected"])
                cases_code += f"        DatasetCase(input={input_val}, expected={expected_val}),\n"
            cases_code += "    ]"
            
            return f"DatasetSource(cases={cases_code})"
        
        elif dataset_config.get("source") == "file":
            path = dataset_config.get("path", "")
            return f'DatasetSource.from_file("{path}")'
        
        elif dataset_config.get("source") == "huggingface":
            imports.append("from agentunit.datasets.huggingface import HuggingFaceDataset")
            path = dataset_config.get("path", "")
            return f'HuggingFaceDataset("{path}")'
        
        elif dataset_config.get("source") == "generator":
            gen_config = dataset_config.get("generator", {})
            gen_type = gen_config.get("type", "template")
            
            if gen_type == "template":
                imports.append("from agentunit.generators import TemplateGenerator")
                return "TemplateGenerator(templates=[...]).generate()"
            elif gen_type == "llm":
                imports.append("from agentunit.generators import LLMPoweredGenerator")
                return "LLMPoweredGenerator(prompt='...').generate()"
            else:
                warnings.append(f"Unknown generator type '{gen_type}'")
                return "dataset  # Define your dataset"
        
        return "dataset  # Define your dataset"
    
    def _generate_metrics(
        self,
        metrics_config: List[Any],
        imports: List[str],
        warnings: List[str],
    ) -> str:
        """Generate metrics configuration code."""
        if not metrics_config:
            return "None"
        
        metrics = []
        
        for metric in metrics_config:
            if isinstance(metric, str):
                # Built-in metric name
                metric_name = metric.lower().replace(" ", "_")
                
                # Map common metric names to imports
                if "correctness" in metric_name or "accuracy" in metric_name:
                    imports.append("from agentunit.metrics import Correctness")
                    metrics.append("Correctness()")
                elif "latency" in metric_name or "response_time" in metric_name:
                    imports.append("from agentunit.metrics import Latency")
                    metrics.append("Latency()")
                elif "faithfulness" in metric_name:
                    imports.append("from agentunit.metrics import Faithfulness")
                    metrics.append("Faithfulness()")
                else:
                    warnings.append(f"Unknown metric '{metric}', please add manually")
                    metrics.append(f"# {metric} - add metric import and instantiation")
            
            elif isinstance(metric, dict):
                # Custom metric configuration
                metric_type = metric.get("type", metric.get("name"))
                warnings.append(f"Custom metric '{metric_type}' needs manual implementation")
                metrics.append(f"# Custom metric: {metric_type}")
        
        if metrics:
            metrics_str = ",\n        ".join(metrics)
            return f"[\n        {metrics_str}\n    ]"
        
        return "None"
    
    def _generate_scenario(
        self,
        name: str,
        adapter_code: str,
        dataset_code: str,
        metrics_code: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate scenario instantiation code."""
        lines = [
            f"# Scenario: {config.get('description', name)}",
            f"{name} = Scenario(",
            f'    name="{name}",',
            f"    adapter={adapter_code},",
            f"    dataset={dataset_code},",
        ]
        
        if metrics_code != "None":
            lines.append(f"    metrics={metrics_code},")
        
        if "retries" in config:
            lines.append(f"    retries={config['retries']},")
        
        if "timeout" in config:
            lines.append(f"    timeout={config['timeout']},")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for Python code."""
        if isinstance(value, str):
            # Escape quotes and format as string
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, (list, dict)):
            return repr(value)
        else:
            return repr(value)
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format a dictionary for Python code."""
        if not d:
            return "{}"
        
        items = []
        for key, value in d.items():
            formatted_value = self._format_value(value)
            items.append(f'"{key}": {formatted_value}')
        
        return "{" + ", ".join(items) + "}"
