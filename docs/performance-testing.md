# AgentUnit Performance Testing

## Overview

This document outlines comprehensive performance testing strategies, benchmarks, and optimization guidelines for AgentUnit. Performance testing ensures that AgentUnit can handle production workloads efficiently and scale appropriately.

## Performance Test Suite

### 1. Load Testing

#### Basic Load Test

```python
# performance_tests/load_test.py
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter

async def basic_load_test():
    """Test basic load handling capabilities."""
    
    # Configure adapter for performance testing
    config = {
        "model": "gpt-3.5-turbo",  # Faster model for load testing
        "temperature": 0.1,        # Reduce randomness for consistency
        "max_turns": 3,           # Limit conversation length
        "timeout": 30             # Reasonable timeout
    }
    
    adapter = AG2Adapter(config)
    
    # Create lightweight test dataset
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": f"load_test_{i}",
                "input": f"Solve this simple math problem: {i} + {i*2}",
                "expected_output": str(i + i*2)
            }
            for i in range(1, 21)  # 20 test cases
        ]
    })
    
    scenario = Scenario("load_test", adapter, dataset)
    
    # Measure execution time
    start_time = time.time()
    result = await scenario.run()
    end_time = time.time()
    
    # Calculate performance metrics
    total_time = end_time - start_time
    throughput = len(result.runs) / total_time
    
    print(f"🚀 Load Test Results:")
    print(f"  Total Cases: {len(result.runs)}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} cases/second")
    print(f"  Success Rate: {result.success_rate:.2%}")
    print(f"  Avg Duration: {result.avg_duration:.2f}s")
    
    # Performance assertions
    assert throughput > 0.1, f"Throughput too low: {throughput}"
    assert result.success_rate > 0.8, f"Success rate too low: {result.success_rate}"
    assert result.avg_duration < 10.0, f"Average duration too high: {result.avg_duration}"

if __name__ == "__main__":
    asyncio.run(basic_load_test())
```

#### Concurrent Load Test

```python
# performance_tests/concurrent_load_test.py
import asyncio
import time
from agentunit import Scenario, DatasetSource, run_suite
from agentunit.adapters import AG2Adapter

async def concurrent_load_test():
    """Test concurrent scenario execution."""
    
    config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_turns": 2,
        "timeout": 20
    }
    
    # Create multiple identical scenarios
    scenarios = []
    for i in range(5):  # 5 concurrent scenarios
        dataset = DatasetSource.from_dict({
            "test_cases": [
                {
                    "id": f"concurrent_{i}_{j}",
                    "input": f"What is {j * 3} divided by 3?",
                    "expected_output": str(j)
                }
                for j in range(1, 6)  # 5 cases per scenario
            ]
        })
        
        adapter = AG2Adapter(config)
        scenario = Scenario(f"concurrent_test_{i}", adapter, dataset)
        scenarios.append(scenario)
    
    # Run scenarios concurrently
    start_time = time.time()
    results = await run_suite(scenarios, max_concurrent=5)
    end_time = time.time()
    
    # Analyze concurrent performance
    total_time = end_time - start_time
    total_cases = sum(len(result.runs) for result in results)
    concurrent_throughput = total_cases / total_time
    
    print(f"🔄 Concurrent Load Test Results:")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Total Cases: {total_cases}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Concurrent Throughput: {concurrent_throughput:.2f} cases/second")
    
    # Performance metrics per scenario
    for i, result in enumerate(results):
        print(f"  Scenario {i}: {result.success_rate:.2%} success, {result.avg_duration:.2f}s avg")
    
    # Assertions
    assert concurrent_throughput > 0.5, f"Concurrent throughput too low: {concurrent_throughput}"
    assert all(result.success_rate > 0.8 for result in results), "Some scenarios had low success rates"

if __name__ == "__main__":
    asyncio.run(concurrent_load_test())
```

### 2. Stress Testing

#### Memory Stress Test

```python
# performance_tests/memory_stress_test.py
import asyncio
import psutil
import gc
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter

async def memory_stress_test():
    """Test memory usage under stress conditions."""
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_turns": 5
    }
    
    adapter = AG2Adapter(config)
    
    # Large dataset to stress memory
    large_dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": f"memory_test_{i}",
                "input": f"Generate a {50 + i} word story about artificial intelligence.",
                "expected_output": "A detailed story about AI"
            }
            for i in range(100)  # 100 cases with longer responses
        ]
    })
    
    scenario = Scenario("memory_stress_test", adapter, large_dataset)
    
    # Track memory during execution
    memory_readings = []
    
    async def monitor_memory():
        while True:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            await asyncio.sleep(1)  # Check every second
    
    # Start memory monitoring
    monitor_task = asyncio.create_task(monitor_memory())
    
    try:
        result = await scenario.run()
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    final_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = max(memory_readings)
    memory_growth = final_memory - initial_memory
    
    # Force garbage collection
    gc.collect()
    post_gc_memory = process.memory_info().rss / 1024 / 1024
    memory_freed = final_memory - post_gc_memory
    
    print(f"🧠 Memory Stress Test Results:")
    print(f"  Initial Memory: {initial_memory:.1f} MB")
    print(f"  Peak Memory: {peak_memory:.1f} MB")
    print(f"  Final Memory: {final_memory:.1f} MB")
    print(f"  Memory Growth: {memory_growth:.1f} MB")
    print(f"  Memory Freed by GC: {memory_freed:.1f} MB")
    print(f"  Success Rate: {result.success_rate:.2%}")
    
    # Memory assertions
    assert memory_growth < 500, f"Memory growth too high: {memory_growth} MB"
    assert peak_memory < initial_memory + 1000, f"Peak memory too high: {peak_memory} MB"
    assert memory_freed > memory_growth * 0.5, f"Insufficient memory cleanup: {memory_freed} MB"

if __name__ == "__main__":
    asyncio.run(memory_stress_test())
```

#### Timeout Stress Test

```python
# performance_tests/timeout_stress_test.py
import asyncio
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter

async def timeout_stress_test():
    """Test behavior under timeout conditions."""
    
    # Configuration with aggressive timeouts
    config = {
        "model": "gpt-4",  # Slower model to trigger timeouts
        "temperature": 0.9,  # Higher randomness
        "max_turns": 10,     # Longer conversations
        "timeout": 5         # Aggressive timeout
    }
    
    adapter = AG2Adapter(config)
    
    # Complex dataset likely to cause timeouts
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": f"timeout_test_{i}",
                "input": f"Write a comprehensive {1000 + i*100} word analysis of quantum computing, including mathematical proofs and detailed explanations of quantum algorithms.",
                "expected_output": "Comprehensive quantum computing analysis"
            }
            for i in range(10)
        ]
    })
    
    scenario = Scenario("timeout_stress_test", adapter, dataset)
    
    result = await scenario.run()
    
    # Analyze timeout behavior
    timeout_count = sum(1 for run in result.runs if run.error and "timeout" in run.error.lower())
    error_count = sum(1 for run in result.runs if run.error)
    
    print(f"⏰ Timeout Stress Test Results:")
    print(f"  Total Cases: {len(result.runs)}")
    print(f"  Timeout Errors: {timeout_count}")
    print(f"  Other Errors: {error_count - timeout_count}")
    print(f"  Success Rate: {result.success_rate:.2%}")
    print(f"  Avg Duration: {result.avg_duration:.2f}s")
    
    # Timeout handling assertions
    assert timeout_count > 0, "Expected some timeouts to occur"
    assert result.success_rate < 1.0, "Expected some failures due to timeouts"
    assert not any(run.duration_ms > 7000 for run in result.runs), "Some runs exceeded timeout threshold"

if __name__ == "__main__":
    asyncio.run(timeout_stress_test())
```

### 3. Platform-Specific Performance Tests

#### AG2 Performance Test

```python
# performance_tests/ag2_performance_test.py
import asyncio
import time
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter

async def ag2_performance_test():
    """Performance test specific to AG2 adapter."""
    
    configs = [
        {"model": "gpt-3.5-turbo", "max_turns": 3},
        {"model": "gpt-4", "max_turns": 3},
        {"model": "gpt-3.5-turbo", "max_turns": 10},
    ]
    
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": f"ag2_perf_{i}",
                "input": f"Help me plan a project with {i+2} team members.",
                "expected_output": "Project planning assistance"
            }
            for i in range(5)
        ]
    })
    
    results = []
    
    for i, config in enumerate(configs):
        adapter = AG2Adapter(config)
        scenario = Scenario(f"ag2_config_{i}", adapter, dataset)
        
        start_time = time.time()
        result = await scenario.run()
        end_time = time.time()
        
        results.append({
            "config": config,
            "duration": end_time - start_time,
            "success_rate": result.success_rate,
            "avg_response_time": result.avg_duration
        })
    
    print(f"🤖 AG2 Performance Test Results:")
    for i, result in enumerate(results):
        config = result["config"]
        print(f"  Config {i+1} ({config['model']}, {config['max_turns']} turns):")
        print(f"    Total Time: {result['duration']:.2f}s")
        print(f"    Success Rate: {result['success_rate']:.2%}")
        print(f"    Avg Response: {result['avg_response_time']:.2f}s")
    
    # Find best performing configuration
    best_config = min(results, key=lambda x: x["duration"])
    print(f"  🏆 Best Config: {best_config['config']}")

if __name__ == "__main__":
    asyncio.run(ag2_performance_test())
```

## Benchmarking Framework

### Performance Benchmark Suite

```python
# performance_tests/benchmark_suite.py
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter, SwarmAdapter

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        
        benchmarks = [
            ("throughput", self.throughput_benchmark),
            ("latency", self.latency_benchmark),
            ("scalability", self.scalability_benchmark),
            ("resource_usage", self.resource_usage_benchmark),
            ("adapter_comparison", self.adapter_comparison_benchmark)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"🔄 Running {name} benchmark...")
            try:
                result = await benchmark_func()
                self.results["benchmarks"][name] = result
                print(f"✅ {name} benchmark completed")
            except Exception as e:
                print(f"❌ {name} benchmark failed: {e}")
                self.results["benchmarks"][name] = {"error": str(e)}
        
        return self.results
    
    async def throughput_benchmark(self) -> Dict[str, Any]:
        """Benchmark throughput performance."""
        
        config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_turns": 2
        }
        
        adapter = AG2Adapter(config)
        
        # Varying dataset sizes
        sizes = [10, 25, 50, 100]
        throughput_results = {}
        
        for size in sizes:
            dataset = DatasetSource.from_dict({
                "test_cases": [
                    {
                        "id": f"throughput_{i}",
                        "input": f"Calculate {i} * 2",
                        "expected_output": str(i * 2)
                    }
                    for i in range(1, size + 1)
                ]
            })
            
            scenario = Scenario(f"throughput_{size}", adapter, dataset)
            
            start_time = time.time()
            result = await scenario.run()
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = len(result.runs) / duration
            
            throughput_results[f"size_{size}"] = {
                "cases": len(result.runs),
                "duration": duration,
                "throughput": throughput,
                "success_rate": result.success_rate
            }
        
        return throughput_results
    
    async def latency_benchmark(self) -> Dict[str, Any]:
        """Benchmark response latency."""
        
        config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_turns": 1
        }
        
        adapter = AG2Adapter(config)
        
        # Single case repeated multiple times for latency measurement
        dataset = DatasetSource.from_dict({
            "test_cases": [
                {
                    "id": f"latency_{i}",
                    "input": "What is 2 + 2?",
                    "expected_output": "4"
                }
                for i in range(20)  # 20 identical cases
            ]
        })
        
        scenario = Scenario("latency_test", adapter, dataset)
        result = await scenario.run()
        
        # Calculate latency statistics
        latencies = [run.duration_ms for run in result.runs if run.success]
        
        if latencies:
            import statistics
            latency_stats = {
                "count": len(latencies),
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "min": min(latencies),
                "max": max(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "p99": sorted(latencies)[int(len(latencies) * 0.99)]
            }
        else:
            latency_stats = {"error": "No successful runs"}
        
        return latency_stats
    
    async def scalability_benchmark(self) -> Dict[str, Any]:
        """Benchmark scalability with concurrent scenarios."""
        
        config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_turns": 2
        }
        
        scalability_results = {}
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 5, 10]
        
        for concurrency in concurrency_levels:
            scenarios = []
            
            for i in range(concurrency):
                dataset = DatasetSource.from_dict({
                    "test_cases": [
                        {
                            "id": f"scale_{concurrency}_{i}_{j}",
                            "input": f"Solve {j} + {j}",
                            "expected_output": str(j + j)
                        }
                        for j in range(1, 6)  # 5 cases per scenario
                    ]
                })
                
                adapter = AG2Adapter(config)
                scenario = Scenario(f"scale_{concurrency}_{i}", adapter, dataset)
                scenarios.append(scenario)
            
            start_time = time.time()
            results = await run_suite(scenarios, max_concurrent=concurrency)
            end_time = time.time()
            
            total_duration = end_time - start_time
            total_cases = sum(len(result.runs) for result in results)
            throughput = total_cases / total_duration
            avg_success_rate = sum(result.success_rate for result in results) / len(results)
            
            scalability_results[f"concurrency_{concurrency}"] = {
                "scenarios": len(scenarios),
                "total_cases": total_cases,
                "duration": total_duration,
                "throughput": throughput,
                "avg_success_rate": avg_success_rate
            }
        
        return scalability_results
    
    async def resource_usage_benchmark(self) -> Dict[str, Any]:
        """Benchmark resource usage."""
        
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_turns": 5
        }
        
        adapter = AG2Adapter(config)
        
        dataset = DatasetSource.from_dict({
            "test_cases": [
                {
                    "id": f"resource_{i}",
                    "input": f"Write a {100 + i*10} word essay about technology.",
                    "expected_output": "Technology essay"
                }
                for i in range(20)
            ]
        })
        
        scenario = Scenario("resource_test", adapter, dataset)
        
        # Monitor resources during execution
        memory_readings = []
        cpu_readings = []
        
        async def monitor_resources():
            while True:
                memory_readings.append(process.memory_info().rss / 1024 / 1024)
                cpu_readings.append(process.cpu_percent())
                await asyncio.sleep(0.5)
        
        monitor_task = asyncio.create_task(monitor_resources())
        
        try:
            result = await scenario.run()
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        return {
            "memory": {
                "initial_mb": initial_memory,
                "final_mb": final_memory,
                "peak_mb": max(memory_readings) if memory_readings else initial_memory,
                "growth_mb": final_memory - initial_memory
            },
            "cpu": {
                "initial_percent": initial_cpu,
                "average_percent": sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0.0,
                "peak_percent": max(cpu_readings) if cpu_readings else 0.0
            },
            "performance": {
                "success_rate": result.success_rate,
                "avg_duration": result.avg_duration
            }
        }
    
    async def adapter_comparison_benchmark(self) -> Dict[str, Any]:
        """Compare performance across different adapters."""
        
        dataset = DatasetSource.from_dict({
            "test_cases": [
                {
                    "id": f"comparison_{i}",
                    "input": f"Explain the concept of {['AI', 'ML', 'NLP', 'computer vision', 'robotics'][i % 5]}",
                    "expected_output": "Technical explanation"
                }
                for i in range(10)
            ]
        })
        
        adapters = {
            "ag2_gpt35": AG2Adapter({"model": "gpt-3.5-turbo", "max_turns": 3}),
            "ag2_gpt4": AG2Adapter({"model": "gpt-4", "max_turns": 3}),
            "swarm": SwarmAdapter({"model": "gpt-3.5-turbo", "max_turns": 3})
        }
        
        comparison_results = {}
        
        for adapter_name, adapter in adapters.items():
            scenario = Scenario(f"comparison_{adapter_name}", adapter, dataset)
            
            start_time = time.time()
            result = await scenario.run()
            end_time = time.time()
            
            comparison_results[adapter_name] = {
                "duration": end_time - start_time,
                "success_rate": result.success_rate,
                "avg_response_time": result.avg_duration,
                "total_cases": len(result.runs)
            }
        
        return comparison_results
    
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Benchmark results saved to {filepath}")

async def run_performance_benchmarks():
    """Main function to run all performance benchmarks."""
    
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark.save_results(f"benchmark_results_{timestamp}.json")
    
    # Print summary
    print(f"\n📈 Performance Benchmark Summary:")
    for name, result in results["benchmarks"].items():
        if "error" in result:
            print(f"  ❌ {name}: {result['error']}")
        else:
            print(f"  ✅ {name}: Completed successfully")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_performance_benchmarks())
```

## Performance Optimization Guidelines

### 1. Configuration Optimization

```python
# Optimized configurations for different scenarios

# High Throughput Configuration
high_throughput_config = {
    "model": "gpt-3.5-turbo",  # Faster model
    "temperature": 0.1,        # Low randomness for consistency
    "max_turns": 3,           # Limit conversation length
    "timeout": 20,            # Reasonable timeout
    "batch_size": 10,         # Process in batches
    "max_concurrent": 5       # Concurrent processing
}

# Low Latency Configuration
low_latency_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,        # No randomness
    "max_turns": 1,           # Single turn only
    "timeout": 10,            # Aggressive timeout
    "cache_enabled": True,    # Enable caching
    "streaming": True         # Stream responses
}

# Resource Efficient Configuration
resource_efficient_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "max_turns": 2,
    "timeout": 30,
    "memory_limit": "1GB",    # Limit memory usage
    "cleanup_interval": 10,   # Regular cleanup
    "batch_size": 5          # Smaller batches
}
```

### 2. Dataset Optimization

```python
# Optimize datasets for performance

def optimize_dataset_for_performance(dataset: DatasetSource) -> DatasetSource:
    """Optimize dataset for better performance."""
    
    # Filter out overly complex cases
    optimized_cases = []
    
    for case in dataset.get_cases():
        input_text = case.get_input_text()
        
        # Skip cases that are too long
        if len(input_text) > 1000:
            continue
            
        # Skip cases with complex requirements
        if any(keyword in input_text.lower() for keyword in 
               ['comprehensive analysis', 'detailed report', 'extensive research']):
            continue
            
        optimized_cases.append(case)
    
    return DatasetSource(optimized_cases)

# Batch processing for large datasets
def create_batched_datasets(dataset: DatasetSource, batch_size: int = 20) -> List[DatasetSource]:
    """Split large dataset into smaller batches."""
    
    cases = dataset.get_cases()
    batches = []
    
    for i in range(0, len(cases), batch_size):
        batch_cases = cases[i:i + batch_size]
        batches.append(DatasetSource(batch_cases))
    
    return batches
```

### 3. Monitoring and Profiling

```python
# Performance monitoring utilities

class PerformanceMonitor:
    """Monitor performance during execution."""
    
    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "memory_usage": [],
            "cpu_usage": [],
            "response_times": [],
            "error_count": 0
        }
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        import psutil
        
        self.metrics["start_time"] = time.time()
        self.process = psutil.Process()
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_resources())
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.metrics["end_time"] = time.time()
        
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_resources(self):
        """Background resource monitoring."""
        while True:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self.metrics["memory_usage"].append(memory_mb)
                self.metrics["cpu_usage"].append(cpu_percent)
                
                await asyncio.sleep(1)  # Monitor every second
            except asyncio.CancelledError:
                break
    
    def record_response_time(self, response_time_ms: float):
        """Record a response time."""
        self.metrics["response_times"].append(response_time_ms)
    
    def record_error(self):
        """Record an error occurrence."""
        self.metrics["error_count"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        duration = self.metrics["end_time"] - self.metrics["start_time"]
        
        summary = {
            "duration_seconds": duration,
            "memory": {
                "peak_mb": max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
                "average_mb": sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
            },
            "cpu": {
                "peak_percent": max(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
                "average_percent": sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0
            },
            "response_times": {
                "count": len(self.metrics["response_times"]),
                "average_ms": sum(self.metrics["response_times"]) / len(self.metrics["response_times"]) if self.metrics["response_times"] else 0,
                "p95_ms": sorted(self.metrics["response_times"])[int(len(self.metrics["response_times"]) * 0.95)] if self.metrics["response_times"] else 0
            },
            "errors": self.metrics["error_count"]
        }
        
        return summary
```

## Performance Best Practices

### 1. Configuration Best Practices

- **Model Selection**: Use `gpt-3.5-turbo` for high throughput, `gpt-4` for quality
- **Temperature**: Lower values (0.0-0.3) for consistent performance
- **Max Turns**: Limit conversation length to reduce latency
- **Timeouts**: Set appropriate timeouts based on expected response times
- **Concurrency**: Use concurrent execution for independent scenarios

### 2. Resource Management

- **Memory**: Monitor memory usage and implement cleanup routines
- **CPU**: Balance CPU usage with response quality requirements
- **Network**: Implement retry logic and connection pooling
- **Caching**: Cache responses for repeated queries

### 3. Scaling Strategies

- **Horizontal Scaling**: Use multiple workers for large workloads
- **Batch Processing**: Process test cases in optimized batches
- **Load Balancing**: Distribute load across available resources
- **Auto-scaling**: Implement dynamic resource scaling

### 4. Monitoring and Alerting

- **Real-time Monitoring**: Track performance metrics in real-time
- **Performance Alerts**: Set up alerts for performance degradation
- **Trend Analysis**: Analyze performance trends over time
- **Capacity Planning**: Plan for future capacity requirements

This comprehensive performance testing framework ensures AgentUnit can handle production workloads efficiently while maintaining high quality and reliability.