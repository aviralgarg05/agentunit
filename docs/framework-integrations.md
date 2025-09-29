# Framework Integrations

AgentUnit includes ready-made adapters for popular orchestration stacks. This guide
explains the prerequisites, helper APIs, and end-to-end wiring steps for each
integration added in the 2025 roadmap.

> **Tip**: All integrations are optional dependencies. Install only what you need.
> The helpers raise `AgentUnitError` with actionable guidance when the underlying
> SDK is missing.

## Phidata (`Scenario.from_phidata`)

- **Install**: `pip install phidata`
- **Inputs**: A Phidata agent object or callable. Works with Agents, Workflows, or Spaces.
- **Helper signature**:
  ```python
  Scenario.from_phidata(
      agent,
      dataset="faq",
      name="marketing-phi",
      input_builder=None,
      extra=None,
  )
  ```
- **Customisation**:
  - Provide `input_builder(case)` to translate `DatasetCase` fields into the dict your
    project expects.
  - Use `extra` to inject static control flags (for example feature toggles or user IDs).
- **Example**:
  ```python
  from my_phi_project import marketing_agent
  scenario = Scenario.from_phidata(
      agent=marketing_agent,
      dataset="faq",
      name="marketing-phi",
      extra={"tenant": "enterprise"},
  )
  ```

## Microsoft PromptFlow (`Scenario.from_promptflow`)

- **Install**: `pip install promptflow`
- **Inputs**: PromptFlow flow definition (object or callable) returned by `promptflow.load_flow`.
- **Helper signature**:
  ```python
  Scenario.from_promptflow(
      flow,
      dataset="faq",
      name="promptflow-support",
      context_builder=None,
      output_key="output",
      run_kwargs=None,
  )
  ```
- **Customisation**:
  - `context_builder(case)` lets you align dataset fields with the flow's expected
    input graph.
  - Override `output_key` or target nested outputs via the tuple notation (e.g.
    `output_key="final_answer"`).
  - Pass `run_kwargs` for PromptFlow runtime options (batch size, streaming, etc.).
- **Example**:
  ```python
  from promptflow import load_flow
  scenario = Scenario.from_promptflow(
      load_flow("flows/support.yaml"),
      dataset="faq",
      name="promptflow-support",
      output_key="final_answer",
  )
  ```

## OpenAI Swarm (`Scenario.from_openai_swarm`)

- **Install**: `pip install openai` (ensure Swarm preview is enabled for your account).
- **Inputs**: Swarm orchestrator or function that accepts `messages=...` plus optional metadata.
- **Helper signature**:
  ```python
  Scenario.from_openai_swarm(
      swarm,
      dataset="faq",
      name="escalation-swarm",
      message_builder=None,
      metadata_builder=None,
      run_kwargs=None,
  )
  ```
- **Customisation**:
  - Override `message_builder(case)` to produce advanced role sequences, system
    prompts, or tool call stubs.
  - Attach experimental flags via `metadata_builder(case)` (for example routing
    hints for orchestrator policies).
  - Use `run_kwargs` for parameters accepted by your swarm implementation
    (concurrency, resilience settings, etc.).
- **Example**:
  ```python
  from my_swarm import escalation_swarm
  scenario = Scenario.from_openai_swarm(
      escalation_swarm,
      dataset="faq",
      name="escalation-swarm",
  )
  ```

## Anthropic Claude on Amazon Bedrock (`Scenario.from_anthropic_bedrock`)

- **Install**: `pip install boto3` and ensure Bedrock access is granted (AWS credentials).
- **Inputs**: Bedrock runtime client (`boto3.client("bedrock-runtime")`) and target `model_id`.
- **Helper signature**:
  ```python
  Scenario.from_anthropic_bedrock(
      client,
      model_id="anthropic.claude-3-sonnet",
      dataset="faq",
      name=None,
      prompt_builder=None,
      invoke_kwargs=None,
      response_key="content",
  )
  ```
- **Customisation**:
  - Override `prompt_builder(case)` to embed tool definitions, conversation history,
    or safety parameters.
  - Populate `invoke_kwargs` with runtime configuration (`temperature`, `max_tokens`).
  - Change `response_key` when your deployment returns custom payload structures.
- **Example**:
  ```python
  import boto3
  bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
  scenario = Scenario.from_anthropic_bedrock(
      client=bedrock,
      model_id="anthropic.claude-3-sonnet",
      dataset="faq",
      name="claude-bedrock",
      invoke_kwargs={"temperature": 0.3},
  )
  ```

## Self-hosted Mistral server (`Scenario.from_mistral_server`)

- **Install**: No additional packages required (AgentUnit bundles `httpx`).
- **Inputs**: Base URL for a Mistral-compatible `/v1/chat/completions` API.
- **Helper signature**:
  ```python
  Scenario.from_mistral_server(
      base_url,
      dataset="faq",
      name=None,
      model="mistral-large-latest",
      max_tokens=512,
      temperature=0.2,
      extra_headers=None,
      message_builder=None,
      http_client=None,
  )
  ```
- **Customisation**:
  - Supply `message_builder(case)` to inject system prompts or tool context.
  - Provide `http_client` (preconfigured `httpx.Client`) for custom TLS, proxies, or
    authentication.
  - Add `extra_headers` when your deployment requires additional auth tokens.
- **Example**:
  ```python
  scenario = Scenario.from_mistral_server(
      base_url="https://mistral.company.internal",
      dataset="faq",
      name="mistral-production",
      temperature=0.1,
  )
  ```

## Rasa conversational bots (`Scenario.from_rasa_endpoint`)

- **Install**: No additional packages required (AgentUnit bundles `httpx`). For local
  bots you can pass a callable handler instead of an HTTP endpoint.
- **Inputs**: REST endpoint URL or Python callable compatible with Rasa's `/webhooks/rest/webhook` payload.
- **Helper signature**:
  ```python
  Scenario.from_rasa_endpoint(
      target,
      dataset="faq",
      name=None,
      sender_id="agentunit",
      session_params=None,
      timeout=10.0,
      headers=None,
      response_key="text",
  )
  ```
- **Customisation**:
  - Provide `session_params` to populate the `metadata` field with conversation
    traits (channel, locale, etc.).
  - Override `response_key` when your bot returns custom JSON structures.
  - Pass `target` as a callable for offline unit testing.
- **Example**:
  ```python
  scenario = Scenario.from_rasa_endpoint(
      target="https://rasa.company.com/webhooks/rest/webhook",
      dataset="faq",
      name="rasa-helpdesk",
      sender_id="agentunit",
  )
  ```

## Next steps

- Combine these helpers with the [framework scenario template](templates/framework_scenarios.py)
  to bootstrap multi-framework suites quickly.
- Continue with [Writing Scenarios](writing-scenarios.md) for advanced policies and
  dataset composition patterns.
