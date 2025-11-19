# Instructions for Creating Release v1.0.0 (Bulbasaur)

This document provides step-by-step instructions for creating the v1.0.0 release of ASH.

## Prerequisites Completed ✓
- [x] Version set to 1.0.0 in `setup.py`
- [x] Version set to 1.0.0 in `ash_model/__init__.py`
- [x] Package builds successfully
- [x] Distribution files created (tar.gz and wheel)
- [x] GitHub Actions workflow configured for PyPI publishing

## Steps to Create the Release

### Option 1: Using GitHub Web Interface (Recommended)

1. **Navigate to the Releases Page**
   - Go to https://github.com/GiulioRossetti/ASH/releases
   - Click "Draft a new release"

2. **Configure the Release**
   - **Tag version**: `v1.0.0` (create new tag on publish)
   - **Target**: Select the branch `copilot/release-version-1-0-0` (or merge to main first)
   - **Release title**: `v1.0.0 (Bulbasaur)`
   - **Description**: Copy the content from `RELEASE_NOTES_v1.0.0.md`

3. **Attach Source Files**
   - Click "Attach binaries by dropping them here or selecting them"
   - Upload the following files from the `dist/` directory:
     - `ash_model-1.0.0-py3-none-any.whl`
     - `ash_model-1.0.0.tar.gz`

4. **Publish the Release**
   - Click "Publish release"
   - The GitHub Action will automatically trigger and publish to PyPI

### Option 2: Using GitHub CLI

If you have `gh` CLI installed and authenticated:

```bash
cd /home/runner/work/ASH/ASH

# Create the release with tag
gh release create v1.0.0 \
  --title "v1.0.0 (Bulbasaur)" \
  --notes-file RELEASE_NOTES_v1.0.0.md \
  dist/ash_model-1.0.0-py3-none-any.whl \
  dist/ash_model-1.0.0.tar.gz
```

### Option 3: Using Git and GitHub API

```bash
# Create and push the tag
git tag -a v1.0.0 -m "Release v1.0.0 (Bulbasaur)"
git push origin v1.0.0

# Then create the release via GitHub web interface or API
```

## What Happens After Publishing

1. **Automatic PyPI Publication**
   - The `python-publish.yml` GitHub Action will automatically trigger
   - It will build the package and publish it to PyPI
   - Monitor the Actions tab to ensure successful publication

2. **Verify PyPI Publication**
   - Visit https://pypi.org/project/ash-model/
   - Confirm version 1.0.0 is listed

3. **Test Installation**
   ```bash
   pip install ash-model==1.0.0
   ```

## Notes

- The GitHub Action requires a `PYPI_API_TOKEN` secret to be configured in the repository settings
- Ensure this secret is set up before publishing the release
- The action will only run when the tag starts with 'v' (which v1.0.0 does)

## Verification Checklist

After creating the release, verify:
- [ ] Release appears on https://github.com/GiulioRossetti/ASH/releases
- [ ] Tag v1.0.0 exists in the repository
- [ ] GitHub Action ran successfully (check Actions tab)
- [ ] Package published to PyPI at https://pypi.org/project/ash-model/1.0.0/
- [ ] Package can be installed: `pip install ash-model==1.0.0`
