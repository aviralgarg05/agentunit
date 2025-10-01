# LLM-Powered Auto-Test Generation Implementation Plan

*Intelligent test case generation using Llama 3.1, Qwen 2.5, and other LLM models*

## Overview

This document outlines the implementation strategy for integrating LLM-powered test generation into AgentUnit v0.4.0, enabling automatic creation of test scenarios from agent descriptions, code analysis, and edge case generation.

## Architecture Design

### Core Components

```python
# src/agentunit/generation/__init__.py
from .generator import LLMTestGenerator
from .analyzers import CodeAnalyzer, DescriptionAnalyzer
from .templates import TemplateEngine
from .validators import GeneratedTestValidator

__all__ = [
    'LLMTestGenerator',
    'CodeAnalyzer', 
    'DescriptionAnalyzer',
    'TemplateEngine',
    'GeneratedTestValidator'
]
```

### Main Generator Interface

```python
# src/agentunit/generation/generator.py
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

class LLMProvider(Enum):
    LLAMA_3_1 = "llama-3.1-8b"
    LLAMA_3_1_70B = "llama-3.1-70b"
    QWEN_2_5 = "qwen-2.5-7b"
    QWEN_2_5_14B = "qwen-2.5-14b"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    MISTRAL_7B = "mistral-7b"

@dataclass
class GenerationConfig:
    """Configuration for test generation"""
    model: LLMProvider
    temperature: float = 0.3
    max_tokens: int = 2048
    num_scenarios: int = 5
    include_edge_cases: bool = True
    include_negative_tests: bool = True
    difficulty_levels: List[str] = None  # ['basic', 'intermediate', 'advanced']
    focus_areas: List[str] = None  # ['functionality', 'edge_cases', 'performance']
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ['basic', 'intermediate', 'advanced']
        if self.focus_areas is None:
            self.focus_areas = ['functionality', 'edge_cases']

@dataclass
class GenerationResult:
    """Result of test generation"""
    scenarios: List['Scenario']
    confidence_scores: List[float]
    generation_metadata: Dict[str, Any]
    suggestions: List[str]
    warnings: List[str]

class LLMTestGenerator:
    """Main class for LLM-powered test generation"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.llm_client = self._initialize_llm_client()
        self.code_analyzer = CodeAnalyzer()
        self.description_analyzer = DescriptionAnalyzer()
        self.template_engine = TemplateEngine()
        self.validator = GeneratedTestValidator()
    
    async def generate_from_description(
        self, 
        agent_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """Generate test scenarios from natural language description"""
        
        # Analyze description to extract key information
        analysis = await self.description_analyzer.analyze(agent_description)
        
        # Generate scenarios using LLM
        raw_scenarios = await self._generate_scenarios_from_analysis(analysis)
        
        # Validate and refine generated scenarios
        validated_scenarios = await self._validate_and_refine(raw_scenarios)
        
        # Calculate confidence scores
        confidence_scores = await self._calculate_confidence_scores(validated_scenarios)
        
        return GenerationResult(
            scenarios=validated_scenarios,
            confidence_scores=confidence_scores,
            generation_metadata={
                'source': 'description',
                'model': self.config.model.value,
                'analysis': analysis.__dict__
            },
            suggestions=await self._generate_suggestions(validated_scenarios),
            warnings=await self._check_for_warnings(validated_scenarios)
        )
    
    async def generate_from_code(
        self,
        agent_code: str,
        code_type: str = "python"
    ) -> GenerationResult:
        """Generate test scenarios from agent code"""
        
        # Analyze code structure and functionality
        code_analysis = await self.code_analyzer.analyze(agent_code, code_type)
        
        # Extract testable components
        components = code_analysis.extract_testable_components()
        
        # Generate scenarios for each component
        all_scenarios = []
        all_confidence_scores = []
        
        for component in components:
            scenarios = await self._generate_scenarios_for_component(component)
            scores = await self._calculate_confidence_scores(scenarios)
            
            all_scenarios.extend(scenarios)
            all_confidence_scores.extend(scores)
        
        return GenerationResult(
            scenarios=all_scenarios,
            confidence_scores=all_confidence_scores,
            generation_metadata={
                'source': 'code',
                'model': self.config.model.value,
                'analysis': code_analysis.__dict__
            },
            suggestions=await self._generate_code_suggestions(code_analysis),
            warnings=await self._check_code_warnings(code_analysis)
        )
    
    async def generate_edge_cases(
        self,
        base_scenarios: List['Scenario'],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """Generate edge cases from existing scenarios"""
        
        edge_case_scenarios = []
        confidence_scores = []
        
        for scenario in base_scenarios:
            # Analyze scenario for edge case opportunities
            edge_opportunities = await self._identify_edge_opportunities(scenario)
            
            # Generate edge cases for each opportunity
            for opportunity in edge_opportunities:
                edge_scenario = await self._generate_edge_case(scenario, opportunity)
                confidence = await self._calculate_edge_case_confidence(edge_scenario, scenario)
                
                edge_case_scenarios.append(edge_scenario)
                confidence_scores.append(confidence)
        
        return GenerationResult(
            scenarios=edge_case_scenarios,
            confidence_scores=confidence_scores,
            generation_metadata={
                'source': 'edge_cases',
                'model': self.config.model.value,
                'base_scenarios_count': len(base_scenarios)
            },
            suggestions=await self._generate_edge_case_suggestions(edge_case_scenarios),
            warnings=await self._check_edge_case_warnings(edge_case_scenarios)
        )
```

### LLM Client Abstraction

```python
# src/agentunit/generation/llm_client.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
import asyncio

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured output matching a schema"""
        pass

class LlamaClient(LLMClient):
    """Client for Llama models via Ollama or vLLM"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Generate text using Llama via Ollama"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]

class QwenClient(LLMClient):
    """Client for Qwen models"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Generate text using Qwen API"""
        # Implementation for Qwen API
        pass

class OpenAIClient(LLMClient):
    """Client for OpenAI models"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Generate text using OpenAI API"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

class ClaudeClient(LLMClient):
    """Client for Anthropic Claude models"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"x-api-key": api_key}
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Generate text using Claude API"""
        
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]
```

### Code Analysis Engine

```python
# src/agentunit/generation/analyzers.py
import ast
import inspect
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CodeComponent:
    """Represents a testable component in code"""
    name: str
    type: str  # 'function', 'class', 'method'
    signature: str
    docstring: Optional[str]
    complexity: int
    dependencies: List[str]
    potential_test_cases: List[str]

@dataclass
class CodeAnalysis:
    """Result of code analysis"""
    components: List[CodeComponent]
    imports: List[str]
    complexity_score: float
    patterns: List[str]
    potential_issues: List[str]
    
    def extract_testable_components(self) -> List[CodeComponent]:
        """Extract components that should have tests"""
        return [c for c in self.components if c.type in ['function', 'method']]

class CodeAnalyzer:
    """Analyze code to identify testable components"""
    
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.pattern_detector = PatternDetector()
    
    async def analyze(self, code: str, code_type: str = "python") -> CodeAnalysis:
        """Analyze code and extract information for test generation"""
        
        if code_type == "python":
            return await self._analyze_python_code(code)
        else:
            raise ValueError(f"Unsupported code type: {code_type}")
    
    async def _analyze_python_code(self, code: str) -> CodeAnalysis:
        """Analyze Python code"""
        
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")
        
        # Extract components
        components = self.ast_analyzer.extract_components(tree)
        
        # Analyze imports
        imports = self.ast_analyzer.extract_imports(tree)
        
        # Calculate complexity
        complexity_score = self.ast_analyzer.calculate_complexity(tree)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(tree)
        
        # Identify potential issues
        potential_issues = self.ast_analyzer.identify_issues(tree)
        
        return CodeAnalysis(
            components=components,
            imports=imports,
            complexity_score=complexity_score,
            patterns=patterns,
            potential_issues=potential_issues
        )

class ASTAnalyzer:
    """Low-level AST analysis"""
    
    def extract_components(self, tree: ast.AST) -> List[CodeComponent]:
        """Extract testable components from AST"""
        components = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                component = self._analyze_function(node)
                components.append(component)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        component = self._analyze_method(item, node.name)
                        components.append(component)
        
        return components
    
    def _analyze_function(self, node: ast.FunctionDef) -> CodeComponent:
        """Analyze a function node"""
        return CodeComponent(
            name=node.name,
            type='function',
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node),
            complexity=self._calculate_node_complexity(node),
            dependencies=self._extract_dependencies(node),
            potential_test_cases=self._suggest_test_cases(node)
        )
    
    def _analyze_method(self, node: ast.FunctionDef, class_name: str) -> CodeComponent:
        """Analyze a method node"""
        return CodeComponent(
            name=f"{class_name}.{node.name}",
            type='method',
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node),
            complexity=self._calculate_node_complexity(node),
            dependencies=self._extract_dependencies(node),
            potential_test_cases=self._suggest_test_cases(node)
        )
```

### Prompt Templates

```python
# src/agentunit/generation/templates.py
from typing import Dict, Any, List
from jinja2 import Template

class TemplateEngine:
    """Generate prompts for LLM test generation"""
    
    def __init__(self):
        self.templates = {
            'description_analysis': self._load_description_template(),
            'code_analysis': self._load_code_template(),
            'edge_case_generation': self._load_edge_case_template(),
            'scenario_refinement': self._load_refinement_template()
        }
    
    def generate_description_prompt(
        self,
        description: str,
        num_scenarios: int = 5,
        focus_areas: List[str] = None
    ) -> str:
        """Generate prompt for description-based test generation"""
        
        template = self.templates['description_analysis']
        return template.render(
            agent_description=description,
            num_scenarios=num_scenarios,
            focus_areas=focus_areas or ['functionality', 'edge_cases'],
            scenario_template=self._get_scenario_template()
        )
    
    def generate_code_prompt(
        self,
        code_analysis: 'CodeAnalysis',
        component: 'CodeComponent'
    ) -> str:
        """Generate prompt for code-based test generation"""
        
        template = self.templates['code_analysis']
        return template.render(
            component=component,
            analysis=code_analysis,
            scenario_template=self._get_scenario_template()
        )
    
    def _load_description_template(self) -> Template:
        """Load template for description analysis"""
        template_str = """
You are an expert at creating comprehensive test scenarios for AI agents. 

Given this agent description:
{{agent_description}}

Create {{num_scenarios}} diverse test scenarios that cover:
{% for area in focus_areas %}
- {{area}}
{% endfor %}

For each scenario, provide:
1. A clear, specific input/prompt
2. Expected behavior or outcome
3. Success criteria
4. Potential failure modes

Use this format for each scenario:
{{scenario_template}}

Focus on creating realistic, challenging scenarios that would thoroughly test the agent's capabilities.
Consider edge cases, error conditions, and boundary situations.
"""
        return Template(template_str)
    
    def _load_code_template(self) -> Template:
        """Load template for code analysis"""
        template_str = """
You are an expert at creating test scenarios for AI agents based on their implementation.

Analyzing this component:
Name: {{component.name}}
Type: {{component.type}}
Signature: {{component.signature}}
Complexity: {{component.complexity}}

{% if component.docstring %}
Documentation: {{component.docstring}}
{% endif %}

Dependencies: {{component.dependencies|join(', ')}}
Potential Issues: {{analysis.potential_issues|join(', ')}}

Create comprehensive test scenarios that:
1. Test normal functionality
2. Test edge cases and boundary conditions  
3. Test error handling
4. Test integration with dependencies

Use this format:
{{scenario_template}}
"""
        return Template(template_str)
    
    def _get_scenario_template(self) -> str:
        """Get the standard scenario template"""
        return """
Scenario: [Brief descriptive name]
Input: [Specific input/prompt for the agent]
Expected Output: [What the agent should produce]
Success Criteria: [How to measure success]
Tags: [relevant tags like 'edge_case', 'error_handling', etc.]
"""
```

### CLI Integration

```python
# Enhancement to src/agentunit/cli.py
import click
from .generation import LLMTestGenerator, GenerationConfig, LLMProvider

@click.group()
def generate():
    """Generate test scenarios using LLM"""
    pass

@generate.command()
@click.option('--description', required=True, help='Agent description')
@click.option('--model', type=click.Choice([p.value for p in LLMProvider]), 
              default='llama-3.1-8b', help='LLM model to use')
@click.option('--num-scenarios', default=5, help='Number of scenarios to generate')
@click.option('--output', help='Output file for generated scenarios')
@click.option('--include-edge-cases/--no-edge-cases', default=True)
def from_description(description: str, model: str, num_scenarios: int, 
                    output: str, include_edge_cases: bool):
    """Generate scenarios from agent description"""
    
    config = GenerationConfig(
        model=LLMProvider(model),
        num_scenarios=num_scenarios,
        include_edge_cases=include_edge_cases
    )
    
    generator = LLMTestGenerator(config)
    
    # Run async generation
    import asyncio
    result = asyncio.run(generator.generate_from_description(description))
    
    # Output results
    if output:
        with open(output, 'w') as f:
            for scenario in result.scenarios:
                f.write(scenario.to_python_code())
    else:
        for i, scenario in enumerate(result.scenarios):
            click.echo(f"\n--- Scenario {i+1} (Confidence: {result.confidence_scores[i]:.2f}) ---")
            click.echo(scenario.to_python_code())

@generate.command()
@click.option('--code-file', required=True, type=click.Path(exists=True))
@click.option('--model', type=click.Choice([p.value for p in LLMProvider]), 
              default='llama-3.1-8b')
@click.option('--output', help='Output file for generated scenarios')
def from_code(code_file: str, model: str, output: str):
    """Generate scenarios from agent code"""
    
    with open(code_file, 'r') as f:
        code_content = f.read()
    
    config = GenerationConfig(model=LLMProvider(model))
    generator = LLMTestGenerator(config)
    
    import asyncio
    result = asyncio.run(generator.generate_from_code(code_content))
    
    # Output results
    click.echo(f"Generated {len(result.scenarios)} scenarios")
    if result.warnings:
        click.echo("Warnings:")
        for warning in result.warnings:
            click.echo(f"  - {warning}")
    
    if output:
        with open(output, 'w') as f:
            for scenario in result.scenarios:
                f.write(scenario.to_python_code())

@generate.command()
@click.option('--suite-file', required=True, type=click.Path(exists=True))
@click.option('--model', type=click.Choice([p.value for p in LLMProvider]), 
              default='llama-3.1-8b')
@click.option('--output', help='Output file for edge cases')
def edge_cases(suite_file: str, model: str, output: str):
    """Generate edge cases from existing scenarios"""
    
    # Load existing scenarios
    scenarios = load_scenarios_from_file(suite_file)
    
    config = GenerationConfig(model=LLMProvider(model))
    generator = LLMTestGenerator(config)
    
    import asyncio
    result = asyncio.run(generator.generate_edge_cases(scenarios))
    
    click.echo(f"Generated {len(result.scenarios)} edge case scenarios")
    
    if output:
        with open(output, 'w') as f:
            for scenario in result.scenarios:
                f.write(scenario.to_python_code())
```

## Implementation Timeline

### Phase 1: Core Infrastructure (2-3 weeks)
1. **LLM Client Abstraction**
   - Implement base client interface
   - Add Llama and Qwen clients
   - Basic prompt templating

2. **Code Analysis Engine**
   - Python AST analyzer
   - Component extraction
   - Basic complexity metrics

3. **CLI Integration**
   - Basic `agentunit generate` commands
   - File I/O for scenarios

### Phase 2: Generation Engine (3-4 weeks)
1. **Description Analysis**
   - Natural language processing
   - Intent extraction
   - Scenario generation

2. **Code-Based Generation**
   - Function/method analysis
   - Test case suggestion
   - Integration testing scenarios

3. **Quality Validation**
   - Confidence scoring
   - Generated test validation
   - Warning system

### Phase 3: Advanced Features (2-3 weeks)
1. **Edge Case Generation**
   - Adversarial scenario creation
   - Boundary condition testing
   - Error condition simulation

2. **Multi-Model Support**
   - OpenAI GPT integration
   - Claude integration
   - Model comparison

3. **Performance Optimization**
   - Async processing
   - Caching
   - Batch generation

## Configuration & Usage

### Installation
```bash
# Install with LLM generation support
pip install agentunit[generation]

# Or install specific LLM providers
pip install agentunit[llama]
pip install agentunit[openai]
pip install agentunit[anthropic]
```

### Configuration
```yaml
# ~/.agentunit/config.yaml
generation:
  default_model: llama-3.1-8b
  models:
    llama-3.1-8b:
      provider: ollama
      base_url: http://localhost:11434
    qwen-2.5-7b:
      provider: ollama
      base_url: http://localhost:11434  
    gpt-4-turbo:
      provider: openai
      api_key_env: OPENAI_API_KEY
    claude-3-sonnet:
      provider: anthropic
      api_key_env: ANTHROPIC_API_KEY
```

### Usage Examples
```python
from agentunit.generation import LLMTestGenerator, GenerationConfig, LLMProvider

# Generate from description
config = GenerationConfig(
    model=LLMProvider.LLAMA_3_1,
    num_scenarios=10,
    include_edge_cases=True
)

generator = LLMTestGenerator(config)
result = await generator.generate_from_description(
    "A customer service chatbot that handles billing inquiries and can escalate to human agents"
)

# Use generated scenarios
for scenario in result.scenarios:
    print(f"Scenario: {scenario.name}")
    print(f"Confidence: {result.confidence_scores[result.scenarios.index(scenario)]}")
    print(scenario.to_python_code())
```

This implementation plan provides a comprehensive foundation for LLM-powered test generation in AgentUnit v0.4.0, enabling developers to dramatically reduce manual test creation time while improving test coverage and quality.