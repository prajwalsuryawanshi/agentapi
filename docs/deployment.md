# Deployment

## Python Package Publishing

AgentAPI uses Trusted Publisher (OIDC) workflows for package publishing.

### TestPyPI

Workflow: `.github/workflows/publish-testpypi.yml`

- Trigger manually from Actions tab.
- Publishes to `https://test.pypi.org/legacy/`.

### PyPI

Workflow: `.github/workflows/publish.yml`

- Triggered on GitHub Release publish.
- Also supports manual `workflow_dispatch`.

### Release Checklist

1. Update version in `pyproject.toml`.
2. Commit and push.
3. Create GitHub Release.
4. Verify package on PyPI.

## GitHub Pages Docs

Workflow: `.github/workflows/docs.yml`

- Builds docs from `docs/` and `mkdocs.yml`.
- Deploys to GitHub Pages.
- Custom domain is managed by `docs/CNAME`.

### Domain Setup

For `agentapi.prajwalsuryawanshi.in`:

- Add DNS CNAME record:
  - host: `agentapi`
  - value: `prajwalsuryawanshi.github.io`
- In GitHub Pages settings, set custom domain.
- Enable HTTPS once certificate is issued.
