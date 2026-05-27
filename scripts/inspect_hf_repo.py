"""
Lists all files in the JeffreyChou/MM-AU HuggingFace dataset repo
so we can see the real folder structure before downloading.

Usage:
    conda activate srp26
    python scripts/inspect_hf_repo.py
"""

from huggingface_hub import list_repo_files

REPO_ID = "JeffreyChou/MM-AU"

print(f"Listing files in {REPO_ID}...\n")

files = sorted(list_repo_files(REPO_ID, repo_type="dataset"))

# Print all files, grouped by top-level folder
folders = {}
for f in files:
    top = f.split("/")[0]
    folders.setdefault(top, []).append(f)

for folder, contents in folders.items():
    print(f"[{folder}]  ({len(contents)} files)")
    for c in contents[:5]:      # show first 5 per folder
        print(f"  {c}")
    if len(contents) > 5:
        print(f"  ... and {len(contents) - 5} more")
    print()

print(f"Total files: {len(files)}")
