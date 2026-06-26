# Release checklist for v1.0.0

This is the practical sequence from "code in a folder" to "citable software
with a DOI and a published paper." Steps are ordered so each unlocks the next.

## Phase 1 — version control (1 hour)

- [ ] Create a GitHub account if you don't have one.
- [ ] Create a new public repo named `neuroflower`.
- [ ] On your local machine:
  ```
  cd /path/to/neuroflower
  git init
  git add .
  git commit -m "Initial commit: v1.0.0"
  git branch -M main
  git remote add origin https://github.com/<your-username>/neuroflower.git
  git push -u origin main
  ```
- [ ] Edit `pyproject.toml` and `CITATION.cff`: replace `<your-username>` with
      your actual GitHub username.

## Phase 2 — make `pip install` work (15 minutes)

- [ ] In a clean Python environment:
  ```
  pip install -e .
  python -c "import neuroflower; print(neuroflower.__version__)"
  ```
  Expected: `1.0.0`.
- [ ] Run the tests:
  ```
  pip install -e ".[test]"
  pytest tests/ -v
  ```
  Expected: 5/5 passing.

## Phase 3 — continuous integration (15 minutes)

- [ ] Push the `.github/workflows/test.yml` (already in this repo).
- [ ] Watch GitHub → Actions tab: tests run automatically.
- [ ] Add the test-status badge to README.md once it's green:
  ```
  ![tests](https://github.com/<user>/neuroflower/actions/workflows/test.yml/badge.svg)
  ```

## Phase 4 — Zenodo + DOI (10 minutes)

- [ ] Go to https://zenodo.org, log in with GitHub.
- [ ] In settings → GitHub, flip the switch ON for the `neuroflower` repo.
- [ ] On GitHub, draft a new release:
      Tag = `v1.0.0`, Title = "Initial v1.0 release"
      Body = paste the [1.0.0] section from CHANGELOG.md
- [ ] Publish the release. Zenodo will automatically mint a DOI within a few
      minutes. You'll get an email with the DOI (looks like
      `10.5281/zenodo.XXXXXXX`).
- [ ] Add the Zenodo DOI badge to README.md and the Zenodo DOI line to
      CITATION.cff and methods_paper.md.

## Phase 5 — JOSS submission (1 day of writing, 2–6 weeks of review)

The Journal of Open Source Software publishes short peer-reviewed papers
about scientific software. The review is open on GitHub and focuses on
documentation and code quality, not novelty claims.

- [ ] Write `paper.md` (JOSS template) and `paper.bib`. The methods paper
      we already have can serve as the *long* form; JOSS wants a short
      ~1000-word version focused on software and usage.
- [ ] Submit at https://joss.theoj.org/papers/new. The form takes ~10 minutes.
- [ ] Reviewers will open issues on your GitHub repo asking for clarifications
      or documentation tweaks. The review process is open and collegial.
- [ ] On acceptance, you get a JOSS DOI for the paper. Add it to CITATION.cff,
      README.md, and the methods paper.

## Phase 6 — (optional) PyPI

- [ ] Only after you have stable external users or a JOSS publication.
- [ ] `pip install build twine && python -m build && twine upload dist/*`
- [ ] After PyPI publication, `pip install neuroflower` works for everyone.

## Phase 7 — maintenance

- [ ] Pin issues for FAQ / dataset-not-supported limitations.
- [ ] Tag minor versions (1.0.1, 1.1.0, …) for fixes and additive changes.
- [ ] **Never silently change locked parameters.** If the diagnostic
      battery in the methods paper would no longer apply, bump to 2.0.0 and
      re-run validation.
