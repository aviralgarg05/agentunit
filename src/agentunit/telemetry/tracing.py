"""OpenTelemetry helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

try:  # pragma: no cover - preferred import path in recent releases
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:  # pragma: no cover - fallback for older distributions
    from opentelemetry.sdk.trace.export import OTLPSpanExporter  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def configure_tracer(exporter: str | None = None) -> None:
    current_provider = trace.get_tracer_provider()
    provider: TracerProvider
    if isinstance(current_provider, TracerProvider):
        provider = current_provider
    else:
        provider = TracerProvider(resource=Resource.create({"service.name": "agentunit"}))
        try:
            trace.set_tracer_provider(provider)
        except Exception as exc:  # pragma: no cover - only triggered when another SDK initialized first
            logger.debug("Tracer provider already configured: %s", exc)
            updated_provider = trace.get_tracer_provider()
            if isinstance(updated_provider, TracerProvider):
                provider = updated_provider
            else:
                logger.warning("Tracer provider configuration skipped; existing provider is %s", type(updated_provider))
                return

    processor_key = f"_agentunit_processor_{exporter or 'console'}"
    if getattr(provider, processor_key, False):
        return

    if exporter in {None, "console"}:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
    elif exporter == "otlp":
        processor = BatchSpanProcessor(OTLPSpanExporter())
    else:
        logger.warning("Unknown exporter '%s', falling back to console", exporter)
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    setattr(provider, processor_key, True)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[trace.Span]:
    tracer = trace.get_tracer("agentunit")
    with tracer.start_as_current_span(name) as current_span:
        for key, value in attributes.items():
            current_span.set_attribute(f"agentunit.{key}", value)
        yield current_span
