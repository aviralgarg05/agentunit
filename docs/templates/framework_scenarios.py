"""Reference suite showcasing Scenario convenience helpers for popular frameworks.

Copy sections relevant to your stack. Each helper wraps an adapter implementation and
applies ergonomic defaults so you can focus on datasets and evaluation policy.
"""

from __future__ import annotations

from agentunit.core.scenario import Scenario
from agentunit.datasets.base import DatasetSource, DatasetCase


# ---------------------------------------------------------------------------
# Shared dataset used by all scenarios in this template
# ---------------------------------------------------------------------------

def build_dataset() -> DatasetSource:
    def _loader():
        yield DatasetCase(
            id="faq-001",
            query="What is the capital of France?",
            expected_output="Paris is the capital of France.",
            context=["Paris is the capital of France."],
            metadata={"domain": "geography"},
        )

    return DatasetSource(name="faq-demo", loader=_loader)


dataset = build_dataset()


# ---------------------------------------------------------------------------
# Framework helpers
# ---------------------------------------------------------------------------

# LangGraph graph from file or in-memory object
langgraph_scenario = Scenario.load_langgraph("graphs/customer_support.py", dataset=dataset)


# OpenAI Agents flow
from my_flows import support_flow  # replace with your flow module
openai_agents_scenario = Scenario.from_openai_agents(
    flow=support_flow,
    dataset=dataset,
    name="support-flow",
    retries=2,
)


# CrewAI crew instance
from my_crewai_setup import crew  # replace with your crew definition
crewai_scenario = Scenario.from_crewai(crew, dataset=dataset, name="crewai-support")


# Phidata agent callable
from my_phi_project import marketing_agent  # replace with your agent factory
phidata_scenario = Scenario.from_phidata(
    agent=marketing_agent,
    dataset=dataset,
    name="marketing-phi",
    extra={"tenant": "enterprise"},
)


# Microsoft PromptFlow orchestration
from promptflow import load_flow  # type: ignore import for optional dependency
promptflow_scenario = Scenario.from_promptflow(
    flow=load_flow("flows/support.yaml"),
    dataset=dataset,
    name="promptflow-support",
    output_key="final_answer",
)


# OpenAI Swarm orchestrator
from my_swarm import escalation_swarm  # replace with swarm entry point
openai_swarm_scenario = Scenario.from_openai_swarm(
    swarm=escalation_swarm,
    dataset=dataset,
    name="escalation-swarm",
)


# Anthropic Claude deployed on Amazon Bedrock
from my_bedrock_runtime import bedrock_client  # replace with boto3 client factory
anthropic_bedrock_scenario = Scenario.from_anthropic_bedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet",
    dataset=dataset,
    name="claude-bedrock",
    invoke_kwargs={"temperature": 0.3},
)


# Self-hosted Mistral server
mistral_scenario = Scenario.from_mistral_server(
    base_url="https://mistral.internal",
    dataset=dataset,
    name="mistral-eu",
    model="mistral-large-latest",
    temperature=0.1,
)


# Rasa REST endpoint
rasa_scenario = Scenario.from_rasa_endpoint(
    target="https://rasa.company.com/webhooks/rest/webhook",
    dataset=dataset,
    name="rasa-helpdesk",
    sender_id="agentunit",
)


# ---------------------------------------------------------------------------
# Aggregate suite
# ---------------------------------------------------------------------------

def create_suite():
    """Return all configured scenarios.

    Drop the scenarios that are not relevant to your project or split them into
    multiple suites for targeted CI pipelines.
    """

    return [
        langgraph_scenario,
        openai_agents_scenario,
        crewai_scenario,
        phidata_scenario,
        promptflow_scenario,
        openai_swarm_scenario,
        anthropic_bedrock_scenario,
        mistral_scenario,
        rasa_scenario,
    ]


suite = create_suite()
