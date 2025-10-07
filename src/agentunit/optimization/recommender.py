"""Recommendation engine for optimization suggestions."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from .analyzer import AnalysisResult


class RecommendationType(Enum):
    """Types of optimization recommendations."""
    PROMPT = "prompt"
    MODEL = "model"
    TOOL = "tool"
    PARAMETER = "parameter"
    ARCHITECTURE = "architecture"
    COST = "cost"
    PERFORMANCE = "performance"


@dataclass
class Recommendation:
    """A single optimization recommendation.
    
    Attributes:
        type: Type of recommendation
        title: Brief title
        description: Detailed description
        priority: Priority level (1-10, 10 being highest)
        impact: Expected impact description
        action: Specific action to take
        metadata: Additional metadata
    """
    type: RecommendationType
    title: str
    description: str
    priority: int
    impact: str
    action: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Recommender:
    """Generates optimization recommendations based on analysis results.
    
    This recommender can suggest:
    - Prompt engineering improvements
    - Model selection changes
    - Tool configuration adjustments
    - Parameter tuning
    - Architecture modifications
    - Cost optimization strategies
    - Performance improvements
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_model: Optional[str] = None
    ):
        """Initialize recommender.
        
        Args:
            use_llm: Whether to use LLM for generating recommendations
            llm_model: LLM model to use (e.g., "gpt-4o-mini" for meta-evaluation)
        """
        self.use_llm = use_llm
        self.llm_model = llm_model or "gpt-4o-mini"
    
    def generate_recommendations(
        self,
        analysis: AnalysisResult,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Recommendation]:
        """Generate recommendations based on analysis.
        
        Args:
            analysis: Analysis result from RunAnalyzer
            context: Additional context (current config, constraints, etc.)
        
        Returns:
            List of recommendations sorted by priority
        """
        recommendations = []
        context = context or {}
        
        # Success rate recommendations
        if analysis.success_rate < 0.7:
            recommendations.extend(self._recommend_for_low_success(analysis))
        
        # Performance recommendations
        if analysis.performance_bottlenecks:
            recommendations.extend(self._recommend_for_performance(analysis))
        
        # Cost recommendations
        if analysis.avg_cost_per_case > 0.1:
            recommendations.extend(self._recommend_for_cost(analysis))
        
        # Failure pattern recommendations
        if analysis.failure_patterns:
            recommendations.extend(self._recommend_for_failures(analysis))
        
        # LLM-powered recommendations (if enabled)
        if self.use_llm:
            recommendations.extend(self._llm_recommendations(analysis, context))
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations
    
    def _recommend_for_low_success(self, analysis: AnalysisResult) -> List[Recommendation]:
        """Recommendations for low success rate."""
        recommendations = []
        
        if analysis.success_rate < 0.5:
            recommendations.append(Recommendation(
                type=RecommendationType.PROMPT,
                title="Critical: Improve Prompt Clarity",
                description=f"Success rate is critically low at {analysis.success_rate:.1%}. "
                           "Consider adding more specific instructions, examples, or constraints to your prompts.",
                priority=10,
                impact=f"Could improve success rate from {analysis.success_rate:.1%} to >70%",
                action="Review and enhance prompts with clearer instructions and examples",
                metadata={"current_success_rate": analysis.success_rate}
            ))
            
            recommendations.append(Recommendation(
                type=RecommendationType.MODEL,
                title="Consider Using a More Capable Model",
                description="The current model may not be capable enough for these tasks. "
                           "Try upgrading to a more advanced model (e.g., GPT-4, Claude 3 Opus).",
                priority=9,
                impact="Could significantly improve success rate for complex reasoning tasks",
                action="Test with a more capable model like GPT-4 or Claude 3 Opus",
                metadata={"current_success_rate": analysis.success_rate}
            ))
        
        elif analysis.success_rate < 0.7:
            recommendations.append(Recommendation(
                type=RecommendationType.PROMPT,
                title="Refine Prompts for Better Performance",
                description=f"Success rate is {analysis.success_rate:.1%}. "
                           "Consider adding few-shot examples or step-by-step instructions.",
                priority=7,
                impact=f"Could improve success rate from {analysis.success_rate:.1%} to >80%",
                action="Add 2-3 examples to prompts and use chain-of-thought prompting",
                metadata={"current_success_rate": analysis.success_rate}
            ))
        
        return recommendations
    
    def _recommend_for_performance(self, analysis: AnalysisResult) -> List[Recommendation]:
        """Recommendations for performance issues."""
        recommendations = []
        
        for bottleneck in analysis.performance_bottlenecks:
            if bottleneck["type"] == "high_latency":
                recommendations.append(Recommendation(
                    type=RecommendationType.PERFORMANCE,
                    title="Reduce Response Latency",
                    description=f"Average latency is {bottleneck['avg_latency']:.2f}s, "
                               f"exceeding threshold of {bottleneck['threshold']:.2f}s.",
                    priority=8 if bottleneck["severity"] == "high" else 6,
                    impact="Could reduce latency by 30-50% through caching or parallel processing",
                    action="Implement caching for common queries and enable parallel tool execution",
                    metadata=bottleneck
                ))
            
            elif bottleneck["type"] == "variable_latency":
                recommendations.append(Recommendation(
                    type=RecommendationType.ARCHITECTURE,
                    title="Stabilize Response Times",
                    description=f"Latency varies significantly (std dev: {bottleneck['std_dev']:.2f}s). "
                               "This suggests inconsistent processing.",
                    priority=5,
                    impact="More predictable user experience",
                    action="Implement request batching or resource pooling",
                    metadata=bottleneck
                ))
        
        return recommendations
    
    def _recommend_for_cost(self, analysis: AnalysisResult) -> List[Recommendation]:
        """Recommendations for cost optimization."""
        recommendations = []
        
        if analysis.avg_cost_per_case > 0.1:
            recommendations.append(Recommendation(
                type=RecommendationType.COST,
                title="Optimize Token Usage",
                description=f"Average cost per case is ${analysis.avg_cost_per_case:.3f}. "
                           "Consider using a smaller model or reducing context length.",
                priority=7,
                impact=f"Could reduce costs by 50-70% (from ${analysis.total_cost:.2f})",
                action="Use GPT-4o-mini or Claude 3 Haiku for simpler tasks",
                metadata={
                    "avg_cost": analysis.avg_cost_per_case,
                    "total_cost": analysis.total_cost,
                    "total_tokens": analysis.total_tokens
                }
            ))
            
            recommendations.append(Recommendation(
                type=RecommendationType.PROMPT,
                title="Reduce Prompt Length",
                description="Long prompts increase costs. Compress prompts while maintaining clarity.",
                priority=6,
                impact="Could reduce token usage by 20-30%",
                action="Remove redundant instructions and use more concise language",
                metadata={"total_tokens": analysis.total_tokens}
            ))
        
        return recommendations
    
    def _recommend_for_failures(self, analysis: AnalysisResult) -> List[Recommendation]:
        """Recommendations based on failure patterns."""
        recommendations = []
        
        for pattern in analysis.failure_patterns:
            if pattern["type"] == "recurring_error":
                recommendations.append(Recommendation(
                    type=RecommendationType.TOOL,
                    title=f"Fix Recurring Error: {pattern['error'][:50]}",
                    description=f"Error '{pattern['error']}' occurs in {pattern['percentage']:.1f}% "
                               f"of failures ({pattern['count']} cases).",
                    priority=9,
                    impact=f"Could fix {pattern['percentage']:.1f}% of current failures",
                    action="Investigate and fix the root cause of this error",
                    metadata=pattern
                ))
            
            elif pattern["type"] == "timeout":
                recommendations.append(Recommendation(
                    type=RecommendationType.PARAMETER,
                    title="Address Timeout Issues",
                    description=f"{pattern['count']} cases ({pattern['percentage']:.1f}%) "
                               "are timing out.",
                    priority=8,
                    impact="Could recover a significant portion of failed cases",
                    action="Increase timeout limits or optimize long-running operations",
                    metadata=pattern
                ))
        
        return recommendations
    
    def _llm_recommendations(
        self,
        analysis: AnalysisResult,
        context: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate LLM-powered recommendations.
        
        Args:
            analysis: Analysis result
            context: Additional context
        
        Returns:
            List of LLM-generated recommendations
        """
        # Placeholder for LLM-based recommendations
        # In a real implementation, this would call an LLM API
        # with the analysis results and context to generate suggestions
        
        recommendations = []
        
        # Example: If we had LLM integration
        # prompt = f"""
        # Analyze this agent performance data and suggest optimizations:
        # 
        # Success Rate: {analysis.success_rate:.1%}
        # Avg Latency: {analysis.avg_latency:.2f}s
        # Avg Cost: ${analysis.avg_cost_per_case:.3f}
        # 
        # Failure Patterns: {analysis.failure_patterns}
        # Performance Issues: {analysis.performance_bottlenecks}
        # 
        # Suggest 3 specific, actionable improvements.
        # """
        # 
        # response = call_llm(self.llm_model, prompt)
        # recommendations = parse_llm_recommendations(response)
        
        return recommendations
