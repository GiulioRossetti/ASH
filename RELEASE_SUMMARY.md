# Release v1.0.0 (Bulbasaur) - Summary

## What Has Been Prepared

All preparation work for releasing version 1.0.0 (codename Bulbasaur) has been completed:

### ✅ Version Configuration
- Version 1.0.0 is set in `ash_model/__init__.py`
- Version 1.0.0 is set in `setup.py`

### ✅ Package Build
- Package builds successfully without errors
- Distribution files created:
  - `ash_model-1.0.0-py3-none-any.whl` (Python wheel)
  - `ash_model-1.0.0.tar.gz` (source tarball)

### ✅ PyPI Publication Workflow
- GitHub Action workflow exists at `.github/workflows/python-publish.yml`
- Configured to trigger on release publication
- Will automatically build and publish to PyPI when release is created

### ✅ Documentation
- **RELEASE_NOTES_v1.0.0.md**: Comprehensive release notes
- **RELEASE_INSTRUCTIONS.md**: Detailed step-by-step instructions for creating the release

## What Needs to Be Done Manually

Due to authentication constraints in this environment, the following action must be performed manually:

### Create the GitHub Release

The GitHub release must be created through one of these methods:

#### Option 1: GitHub Web Interface (Easiest)
1. Go to https://github.com/GiulioRossetti/ASH/releases
2. Click "Draft a new release"
3. Set tag to `v1.0.0` (create new tag)
4. Set title to `v1.0.0 (Bulbasaur)`
5. Copy content from `RELEASE_NOTES_v1.0.0.md` into the description
6. Attach files: `dist/ash_model-1.0.0-py3-none-any.whl` and `dist/ash_model-1.0.0.tar.gz`
7. Click "Publish release"

#### Option 2: GitHub CLI (if authenticated)
```bash
gh release create v1.0.0 \
  --title "v1.0.0 (Bulbasaur)" \
  --notes-file RELEASE_NOTES_v1.0.0.md \
  dist/ash_model-1.0.0-py3-none-any.whl \
  dist/ash_model-1.0.0.tar.gz
```

## What Happens Automatically After Release

Once the release is published on GitHub:

1. **GitHub Action Triggers**: The `python-publish.yml` workflow automatically starts
2. **Package Build**: The action builds the package from source
3. **PyPI Publication**: The package is published to PyPI using the `PYPI_API_TOKEN` secret
4. **Availability**: The package becomes available at https://pypi.org/project/ash-model/

## Verification Steps

After creating the release, verify:
- [ ] Release visible at https://github.com/GiulioRossetti/ASH/releases/tag/v1.0.0
- [ ] GitHub Action completed successfully (check Actions tab)
- [ ] Package available at https://pypi.org/project/ash-model/1.0.0/
- [ ] Package installs correctly: `pip install ash-model==1.0.0`

## Prerequisites for PyPI Publication

Ensure the repository has the `PYPI_API_TOKEN` secret configured:
- Go to repository Settings → Secrets and variables → Actions
- Verify `PYPI_API_TOKEN` secret exists
- If not, create it with a valid PyPI API token

## Timeline

This release follows the previous release:
- v0.1.0 (Squirtle) - April 5, 2023
- v1.0.0 (Bulbasaur) - November 2025 ← Current release

The naming convention follows Pokémon names in Pokédex order (Bulbasaur is #001, appropriate for v1.0.0).
