# Local data directory

This repository does not redistribute source PDFs or user-owned reference
annotations. Those materials may be copyright-, license-, or privacy-restricted.
Only process documents you are authorized to send to the configured model
provider.

Place input PDFs anywhere below `data/corpus/`; discovery is recursive and the
`.pdf` extension is matched case-insensitively. Subdirectory names are
unrestricted and have no effect on source identity.

Pipeline outputs are written below `data/parsed/`, `data/extracted/`, and
`data/database/`. They are intentionally ignored by Git. `data/manifest.json`
is generated locally and can contain absolute source paths, so it must not be
published. Each assembly also writes `data/database/run_provenance.json` with
content IDs, model names, dependency versions, and hashes of the code, prompts,
configuration, extraction inputs, and any active local assembly extensions used
for that run. Absolute paths to local override files are not recorded.

A fresh clone can run the synthetic unit tests and process a user's authorized
documents. Reproducing any external dataset requires the corresponding source
documents, annotations, configuration, and model access from its owner.
