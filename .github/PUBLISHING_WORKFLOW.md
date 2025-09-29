# Automated Publishing Workflow

This document explains how to use the GitHub Actions workflow to automatically publish your AgentUnit package to PyPI.

## Overview

The workflow (`.github/workflows/publish.yml`) automatically publishes to PyPI when:
1. You push a version tag (like `v1.0.0`)
2. You manually trigger it via GitHub's web interface

## Setup Instructions

### 1. Configure GitHub Secrets

Follow the instructions in `.github/SECRETS_SETUP.md` to add your PyPI token to GitHub repository secrets.

### 2. Create and Push a Version Tag

```bash
# Update version in pyproject.toml first
# Then commit your changes
git add .
git commit -m "Release v1.0.0"

# Create and push the tag
git tag v1.0.0
git push origin v1.0.0
```

### 3. Monitor the Workflow

1. Go to your repository on GitHub
2. Click the "Actions" tab
3. Watch the "Publish to PyPI" workflow run
4. Check for any errors in the workflow logs

## Manual Trigger

You can also trigger publishing manually:

1. Go to repository → Actions tab
2. Click "Publish to PyPI" workflow
3. Click "Run workflow" button
4. Choose the branch and click "Run workflow"

## Workflow Steps

The automated workflow:
1. ✅ Checks out your code
2. ✅ Sets up Python 3.11
3. ✅ Installs build tools (`build`, `twine`)
4. ✅ Builds the package (`python -m build`)
5. ✅ Validates the package (`twine check dist/*`)
6. ✅ Publishes to PyPI using your token

## Version Management

### Semantic Versioning

Use semantic versioning for your tags:
- `v1.0.0` - Major release
- `v1.1.0` - Minor release (new features)
- `v1.0.1` - Patch release (bug fixes)

### Update Process

1. Update version in `pyproject.toml`:
   ```toml
   [tool.poetry]
   version = "1.0.0"  # Remove 'v' prefix
   ```

2. Update changelog/documentation if needed

3. Commit changes:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 1.0.0"
   ```

4. Create and push tag:
   ```bash
   git tag v1.0.0
   git push origin main
   git push origin v1.0.0
   ```

## Security Features

- 🔒 Uses environment protection (`release` environment)
- 🔑 Stores token in encrypted GitHub secrets
- 🛡️ Minimal permissions (`contents: read`)
- 📊 Verbose logging for transparency

## Troubleshooting

### Common Issues

**Build fails**: Check that `pyproject.toml` is valid and dependencies are correct.

**Token error**: Verify your `PYPI_API_TOKEN` secret is set correctly.

**Version conflict**: Ensure the version in `pyproject.toml` isn't already published.

**Permission denied**: Check that your PyPI token has the right scope.

### Manual Fallback

If the automated workflow fails, you can always publish manually:

```bash
cd /Users/aviralgarg/Everything/agentunit
source .venv/bin/activate
poetry build
poetry publish
```

## First Time Setup Checklist

- [ ] Add `PYPI_API_TOKEN` to GitHub repository secrets
- [ ] Set up `release` environment (optional but recommended)
- [ ] Update version in `pyproject.toml`
- [ ] Test the workflow with a pre-release tag (e.g., `v0.1.0-alpha`)
- [ ] Verify package appears on PyPI
- [ ] Test installation: `pip install agentunit`

Your package will be automatically published to PyPI whenever you create a new version tag! 🚀