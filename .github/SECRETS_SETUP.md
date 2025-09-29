# GitHub Repository Secrets Configuration

This file explains how to configure the necessary secrets for automated PyPI publishing.

## Required Secrets

### PYPI_API_TOKEN

**What it is**: Your PyPI API token for publishing packages
**Where to get it**: https://pypi.org/manage/account/token/

**How to create a PyPI token**:
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Give it a name like "agentunit-github-actions"
4. Choose scope: "Entire account" (recommended) or "Project: agentunit" (if package already exists)
5. Copy the token that starts with `pypi-`

**How to add to GitHub**:
1. Go to your repository: https://github.com/aviralgarg05/agentunit
2. Click "Settings" tab
3. Click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `PYPI_API_TOKEN`
6. Value: Paste your PyPI token (starts with `pypi-`)
7. Click "Add secret"

## Environment Protection (Recommended)

For additional security, set up an environment:

1. Go to repository Settings → Environments
2. Click "New environment"
3. Name it `release`
4. Configure protection rules:
   - ☑️ Required reviewers (add yourself)
   - ☑️ Wait timer (optional: 0-5 minutes)
   - ☑️ Deployment branches: Only protected branches

## Your PyPI Token

**IMPORTANT**: Add your actual PyPI token in GitHub secrets:

```
Secret Name: PYPI_API_TOKEN
Secret Value: [Your PyPI token that starts with 'pypi-']
```

## Security Notes

- Never commit secrets to your repository
- Use environment protection for production releases
- Rotate tokens periodically
- Use project-scoped tokens when possible
- Keep your PyPI token secure and don't share it