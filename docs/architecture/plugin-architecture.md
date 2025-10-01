# AgentUnit Plugin Architecture Design

*Extensible plugin system for community-driven expansion*

## Overview

The AgentUnit plugin ecosystem enables developers to extend the framework with custom adapters, metrics, datasets, and other components. This design document outlines the architecture for a robust, secure, and user-friendly plugin system.

## Core Architecture

### Plugin Types

```python
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class PluginType(Enum):
    ADAPTER = "adapter"
    METRIC = "metric" 
    DATASET = "dataset"
    GENERATOR = "generator"
    REPORTER = "reporter"
    SAFETY_CHECK = "safety_check"
    BENCHMARK = "benchmark"

class BasePlugin(ABC):
    """Base class for all AgentUnit plugins"""
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Type of plugin"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """Required dependencies"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate plugin configuration"""
        pass
```

### Plugin Registry

```python
class PluginRegistry:
    """Central registry for managing plugins"""
    
    def __init__(self):
        self._plugins: Dict[PluginType, Dict[str, BasePlugin]] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
    
    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin"""
        if not plugin.validate():
            raise PluginValidationError(f"Plugin {plugin.name} failed validation")
        
        plugin_type = plugin.plugin_type
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = {}
        
        self._plugins[plugin_type][plugin.name] = plugin
        self._metadata[plugin.name] = PluginMetadata.from_plugin(plugin)
    
    def get(self, plugin_type: PluginType, name: str) -> Optional[BasePlugin]:
        """Get a registered plugin"""
        return self._plugins.get(plugin_type, {}).get(name)
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[PluginMetadata]:
        """List registered plugins"""
        if plugin_type:
            return [self._metadata[name] for name in self._plugins.get(plugin_type, {})]
        return list(self._metadata.values())
    
    def discover_from_entrypoints(self) -> None:
        """Discover plugins from setuptools entry points"""
        import pkg_resources
        
        for entry_point in pkg_resources.iter_entry_points('agentunit.plugins'):
            try:
                plugin_class = entry_point.load()
                plugin = plugin_class()
                self.register(plugin)
            except Exception as e:
                logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

# Global registry instance
plugin_registry = PluginRegistry()
```

### Plugin Metadata

```python
@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    homepage: Optional[str]
    dependencies: List[str]
    agentunit_version: str  # Minimum AgentUnit version required
    tags: List[str]
    license: str
    
    @classmethod
    def from_plugin(cls, plugin: BasePlugin) -> 'PluginMetadata':
        """Create metadata from plugin instance"""
        # Extract metadata from plugin attributes or docstrings
        return cls(
            name=plugin.name,
            version=plugin.version,
            plugin_type=plugin.plugin_type,
            description=getattr(plugin, '__doc__', '') or '',
            author=getattr(plugin, '__author__', 'Unknown'),
            homepage=getattr(plugin, '__homepage__', None),
            dependencies=plugin.dependencies,
            agentunit_version=getattr(plugin, '__agentunit_version__', '0.4.0'),
            tags=getattr(plugin, '__tags__', []),
            license=getattr(plugin, '__license__', 'MIT')
        )
```

## Plugin Development

### Adapter Plugin Template

```python
from agentunit.adapters.base import BaseAdapter
from agentunit.plugins import BasePlugin, PluginType

class CustomFrameworkAdapter(BaseAdapter, BasePlugin):
    """Example adapter plugin for a custom framework"""
    
    # Plugin metadata
    __author__ = "Your Name"
    __version__ = "1.0.0"
    __homepage__ = "https://github.com/yourname/agentunit-custom-adapter"
    __license__ = "MIT"
    __tags__ = ["custom", "framework", "llm"]
    __agentunit_version__ = "0.4.0"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.ADAPTER
    
    @property
    def name(self) -> str:
        return "custom-framework"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def dependencies(self) -> List[str]:
        return ["custom-framework>=2.0.0", "requests>=2.25.0"]
    
    def validate(self) -> bool:
        """Validate adapter configuration"""
        try:
            import custom_framework
            return hasattr(custom_framework, 'required_method')
        except ImportError:
            return False
    
    # Implement BaseAdapter methods
    def create_scenario(self, **kwargs) -> 'Scenario':
        # Implementation here
        pass
```

### Metric Plugin Template

```python
from agentunit.metrics.base import BaseMetric
from agentunit.plugins import BasePlugin, PluginType

class CustomMetric(BaseMetric, BasePlugin):
    """Example custom metric plugin"""
    
    __author__ = "Your Name"
    __version__ = "1.0.0"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.METRIC
    
    @property
    def name(self) -> str:
        return "custom-quality-score"
    
    def evaluate(self, expected: Any, actual: Any, context: Dict[str, Any] = None) -> float:
        # Custom metric implementation
        pass
```

## Package Distribution

### PyPI Package Structure

```
agentunit-custom-adapter/
├── setup.py
├── pyproject.toml
├── README.md
├── LICENSE
├── agentunit_custom_adapter/
│   ├── __init__.py
│   ├── adapter.py
│   └── tests/
│       └── test_adapter.py
└── entry_points.txt
```

### Setup Configuration

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="agentunit-custom-adapter",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "agentunit>=0.4.0",
        "custom-framework>=2.0.0"
    ],
    entry_points={
        'agentunit.plugins': [
            'custom-framework = agentunit_custom_adapter:CustomFrameworkAdapter',
        ],
    },
    # Standard package metadata
    author="Your Name",
    description="Custom framework adapter for AgentUnit",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourname/agentunit-custom-adapter",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```

### Plugin Template Generator

```python
class PluginTemplateGenerator:
    """Generate plugin templates"""
    
    def create_adapter_template(self, name: str, framework: str) -> str:
        """Generate adapter plugin template"""
        template = f"""
from agentunit.adapters.base import BaseAdapter
from agentunit.plugins import BasePlugin, PluginType

class {name}Adapter(BaseAdapter, BasePlugin):
    '''Adapter for {framework} framework'''
    
    __author__ = "{{author}}"
    __version__ = "1.0.0"
    __homepage__ = "{{homepage}}"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.ADAPTER
    
    @property
    def name(self) -> str:
        return "{name.lower()}"
    
    # Implement adapter methods here
"""
        return template
    
    def create_package_structure(self, plugin_name: str, plugin_type: PluginType) -> Dict[str, str]:
        """Generate complete package structure"""
        # Return dictionary of file paths and contents
        pass
```

## Plugin Management CLI

### CLI Commands

```bash
# Plugin discovery and installation
agentunit plugin search "llm metrics"
agentunit plugin info agentunit-anthropic-adapter
agentunit plugin install agentunit-anthropic-adapter
agentunit plugin install git+https://github.com/user/custom-plugin.git
agentunit plugin uninstall agentunit-anthropic-adapter

# Plugin development
agentunit plugin create --type adapter --name "my-custom-adapter" --framework "CustomLLM"
agentunit plugin validate ./my-plugin/
agentunit plugin publish ./my-plugin/ --pypi-token TOKEN

# Plugin management
agentunit plugin list --installed
agentunit plugin list --available
agentunit plugin update --all
agentunit plugin update agentunit-anthropic-adapter
```

### CLI Implementation

```python
import click
from typing import Optional

@click.group()
def plugin():
    """Plugin management commands"""
    pass

@plugin.command()
@click.argument('query')
@click.option('--type', 'plugin_type', help='Filter by plugin type')
def search(query: str, plugin_type: Optional[str]):
    """Search for plugins"""
    manager = PluginManager()
    results = manager.search_plugins(query, plugin_type)
    
    for plugin in results:
        click.echo(f"{plugin.name} ({plugin.version}) - {plugin.description}")

@plugin.command()
@click.argument('name')
@click.option('--upgrade', is_flag=True, help='Upgrade if already installed')
def install(name: str, upgrade: bool):
    """Install a plugin"""
    manager = PluginManager()
    try:
        result = manager.install_plugin(name, upgrade=upgrade)
        click.echo(f"Successfully installed {result.name} {result.version}")
    except PluginInstallError as e:
        click.echo(f"Installation failed: {e}", err=True)

@plugin.command()
@click.option('--type', 'plugin_type', help='Plugin type')
@click.option('--name', help='Plugin name')
@click.option('--framework', help='Target framework')
@click.option('--output', help='Output directory')
def create(plugin_type: str, name: str, framework: str, output: str):
    """Create a new plugin template"""
    generator = PluginTemplateGenerator()
    generator.create_plugin_template(
        plugin_type=PluginType(plugin_type),
        name=name,
        framework=framework,
        output_dir=output
    )
    click.echo(f"Created plugin template in {output}")
```

## Security & Validation

### Plugin Validation

```python
class PluginValidator:
    """Validate plugin security and compatibility"""
    
    def validate_plugin(self, plugin_path: str) -> ValidationResult:
        """Comprehensive plugin validation"""
        results = []
        
        # Code security analysis
        results.append(self._scan_for_security_issues(plugin_path))
        
        # Dependency validation
        results.append(self._validate_dependencies(plugin_path))
        
        # API compatibility
        results.append(self._check_api_compatibility(plugin_path))
        
        # Performance impact
        results.append(self._assess_performance_impact(plugin_path))
        
        return ValidationResult(results)
    
    def _scan_for_security_issues(self, plugin_path: str) -> SecurityScanResult:
        """Scan for potential security issues"""
        # Use bandit or similar tool
        pass
    
    def _validate_dependencies(self, plugin_path: str) -> DependencyValidationResult:
        """Validate plugin dependencies"""
        # Check for known vulnerabilities, license compatibility
        pass
```

### Sandboxing

```python
class PluginSandbox:
    """Sandbox environment for plugin execution"""
    
    def __init__(self, allowed_modules: List[str]):
        self.allowed_modules = allowed_modules
    
    def execute_in_sandbox(self, plugin: BasePlugin, method: str, *args, **kwargs):
        """Execute plugin method in sandboxed environment"""
        # Implement resource limits, module restrictions
        pass
```

## Plugin Marketplace

### Plugin Discovery Service

```python
class PluginMarketplace:
    """Central marketplace for AgentUnit plugins"""
    
    def search_plugins(self, query: str, filters: Dict[str, Any] = None) -> List[PluginInfo]:
        """Search plugins in marketplace"""
        pass
    
    def get_plugin_info(self, name: str) -> PluginInfo:
        """Get detailed plugin information"""
        pass
    
    def get_plugin_reviews(self, name: str) -> List[PluginReview]:
        """Get plugin reviews and ratings"""
        pass
    
    def submit_plugin(self, plugin_package: str, metadata: PluginMetadata) -> SubmissionResult:
        """Submit plugin to marketplace"""
        pass
```

### Web Interface

```python
# FastAPI web interface for plugin marketplace
from fastapi import FastAPI, Depends
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AgentUnit Plugin Marketplace")
templates = Jinja2Templates(directory="templates")

@app.get("/plugins")
async def list_plugins(category: Optional[str] = None):
    """List available plugins"""
    marketplace = PluginMarketplace()
    plugins = marketplace.search_plugins("", {"category": category})
    return {"plugins": plugins}

@app.get("/plugins/{plugin_name}")
async def get_plugin_details(plugin_name: str):
    """Get plugin details"""
    marketplace = PluginMarketplace()
    plugin = marketplace.get_plugin_info(plugin_name)
    reviews = marketplace.get_plugin_reviews(plugin_name)
    return {"plugin": plugin, "reviews": reviews}
```

## Integration with Core Framework

### Automatic Plugin Discovery

```python
# In src/agentunit/__init__.py
def discover_and_load_plugins():
    """Automatically discover and load plugins on import"""
    plugin_registry.discover_from_entrypoints()
    
    # Load plugins from configuration
    config = get_user_config()
    for plugin_name in config.get('enabled_plugins', []):
        try:
            plugin_registry.load_plugin(plugin_name)
        except Exception as e:
            logger.warning(f"Failed to load configured plugin {plugin_name}: {e}")

# Auto-discover on import
discover_and_load_plugins()
```

### Plugin Configuration

```python
# User configuration in ~/.agentunit/config.yaml
plugins:
  enabled:
    - anthropic-adapter
    - custom-metrics
  
  settings:
    anthropic-adapter:
      api_key_env: ANTHROPIC_API_KEY
      default_model: claude-3-sonnet
    
    custom-metrics:
      threshold: 0.8
      debug_mode: false
```

This plugin architecture provides a robust foundation for community-driven extension of AgentUnit while maintaining security, compatibility, and ease of use.