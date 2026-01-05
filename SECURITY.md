# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| 0.5.x   | :white_check_mark: |
| < 0.5.0 | :x:                |

## Reporting a Vulnerability

We take the security of AgentUnit seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed

### How to Report

**For security vulnerabilities, please email:** gargaviral99@gmail.com

In your report, please include:

1. **Description** - Clear description of the vulnerability
2. **Impact** - What an attacker could do with this vulnerability
3. **Steps to Reproduce** - Detailed steps to reproduce the issue
4. **Affected Versions** - Which versions are affected
5. **Suggested Fix** - If you have ideas on how to fix it
6. **Your Contact Info** - How we can reach you for follow-up

### What to Expect

1. **Acknowledgment** - We will acknowledge receipt of your report within 48 hours
2. **Initial Assessment** - We will provide an initial assessment within 5 business days
3. **Regular Updates** - We will keep you informed of our progress
4. **Fix Timeline** - Critical vulnerabilities will be addressed within 7-14 days
5. **Public Disclosure** - We will coordinate disclosure timing with you
6. **Credit** - You will be credited in the security advisory (unless you prefer to remain anonymous)

## Security Update Process

When a security vulnerability is confirmed:

1. We will develop and test a fix
2. We will prepare a security advisory
3. We will release a patch version
4. We will publish the security advisory
5. We will notify users through:
   - GitHub Security Advisories
   - Release notes
   - PyPI release announcement

## Security Best Practices for Users

### Installation

- Always install from official sources (PyPI)
- Verify package integrity
- Keep AgentUnit updated to the latest version

```bash
# Check current version
pip show agentunit

# Update to latest version
pip install --upgrade agentunit
```

### Configuration

- Never commit API keys or secrets to version control
- Use environment variables for sensitive configuration
- Rotate API keys regularly
- Use least-privilege principles for API access

Example secure configuration:

```python
import os
from agentunit import Scenario

# Good: Use environment variables
api_key = os.environ.get("OPENAI_API_KEY")

# Bad: Never hardcode secrets
# api_key = "sk-..."
```

### Data Protection

- Sanitize sensitive data before evaluation
- Use AgentUnit's privacy features for PII masking
- Review trace logs before sharing
- Be aware of data retention in external services

### OpenTelemetry Security

- Secure your OTLP endpoint with TLS
- Use authentication for telemetry backends
- Filter sensitive attributes from traces
- Review exported data regularly

## Known Security Considerations

### Third-Party Adapters

AgentUnit integrates with various third-party services (OpenAI, Anthropic, etc.):

- Each adapter may have its own security considerations
- Review third-party service security policies
- Understand data processing and retention policies
- Use adapter-specific security features

### Evaluation Data

- Test datasets may contain sensitive information
- Be cautious when sharing evaluation results
- Use mock data for public demonstrations
- Review metric outputs for information leakage

### Code Execution

- AgentUnit may execute adapter code dynamically
- Only use trusted adapters from verified sources
- Review custom adapter code before deployment
- Use sandboxing for untrusted scenarios

## Dependency Security

We monitor dependencies for known vulnerabilities using:

- GitHub Dependabot
- Safety checks in CI
- Regular dependency audits

To check for vulnerable dependencies:

```bash
# Using safety
pip install safety
safety check

# Using pip-audit
pip install pip-audit
pip-audit
```

## Compliance

### Data Privacy

AgentUnit supports privacy-preserving evaluation:

- PII detection and masking
- Differential privacy mechanisms
- Federated evaluation
- Secure aggregation

See [Privacy Documentation](docs/privacy.md) for details.

### License Compliance

- AgentUnit is MIT licensed
- Review licenses of all dependencies
- Ensure compliance with third-party service terms
- Respect data usage restrictions

## Security Changelog

### Version 0.6.0
- Added comprehensive security documentation
- Improved PII detection capabilities
- Enhanced telemetry filtering options

### Version 0.5.0
- Initial security policy
- Basic PII masking features

## Contact

- Security Email: gargaviral99@gmail.com
- General Issues: [GitHub Issues](https://github.com/aviralgarg05/agentunit/issues)
- Discussions: [GitHub Discussions](https://github.com/aviralgarg05/agentunit/discussions)

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help us maintain a secure project.

Thank you for helping keep AgentUnit secure!
