# Upload notes (maintainer only, not part of the model card)

This directory is the Hugging Face repository layout for
`<hf-user>/Ternary-Bonsai-2-27B-dspark-dflash`. Nothing in it was uploaded.
The two GGUF files are hard links to the files in `models/bonsai2-gguf/27B/`
(same inode, no copy). Do not upload this file.

Before you upload, fill in the placeholders in `README.md`:

- `<PR-URL-fix>`: the PrismML-Eng/llama.cpp pull request of branch `fix/dflash-borrowed-hadamard`.
- `<PR-URL-recipe>`: the Bonsai-demo pull request that adds `tools/dspark-retrain/`.
- `<PR-URL-benchmark>`: the Bonsai-demo pull request that adds `scripts/spec_bench/` and the GB10 Bonsai 2 entry.

Then check that no placeholder is left:

```bash
grep -n '<PR-URL' README.md   # expect no output
```

## Files

| file | bytes | sha256 |
| --- | ---: | --- |
| `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf` | 1104605632 | `eb80f88fc94f267b6b612cba4deac31ecb0dfff0f587f2786fc3de855ceaf84a` |
| `Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf` | 631713664 | `4ebd761b4f510a94982c1ae10a167ef36360f4e8bedbdb343143ee65dbe6142a` |
| `README.md` | model card | |
| `SHA256SUMS` | the two hashes in `sha256sum -c` format | |

Verify the local files first:

```bash
cd /home/REDACTED/Bonsai-demo/models/hf-release/Ternary-Bonsai-2-27B-dspark-dflash
sha256sum -c SHA256SUMS   # expect two OK lines
```

## Upload (hf CLI 1.32.0)

Set `HF_USER` to the account or organization that owns the repository.

```bash
export HF_USER=<hf-user>
export REPO=$HF_USER/Ternary-Bonsai-2-27B-dspark-dflash
cd /home/REDACTED/Bonsai-demo/models/hf-release/Ternary-Bonsai-2-27B-dspark-dflash

# 1. log in once (or export HF_TOKEN with a write token)
hf auth login

# 2. create the repository as private; make it public after the checks in step 4
hf repos create $REPO --type model --private

# 3. one commit with the card, the two GGUF files and the hash list
hf upload $REPO . . --type model \
  --include "README.md" --include "SHA256SUMS" --include "*.gguf" \
  --exclude "UPLOAD.md" \
  --commit-message "Add the DSpark v1 and v2 drafters for Ternary-Bonsai-2-27B"
```

Per-file form, if you prefer one commit per file:

```bash
hf upload $REPO README.md README.md
hf upload $REPO SHA256SUMS SHA256SUMS
hf upload $REPO Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf
hf upload $REPO Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf
```

Legacy CLI (`huggingface-cli`), same arguments:

```bash
huggingface-cli repo create Ternary-Bonsai-2-27B-dspark-dflash --type model --private
huggingface-cli upload $REPO . . --include "README.md" --include "SHA256SUMS" --include "*.gguf" --exclude "UPLOAD.md"
```

## Checks after the upload

```bash
# 4a. the repository lists the four files with the sizes above
hf models info $REPO 2>/dev/null || curl -s https://huggingface.co/api/models/$REPO | python3 -m json.tool | grep -E '"rfilename"|"size"'

# 4b. download to a scratch directory and compare the hashes
mkdir -p /tmp/hf-check && cd /tmp/hf-check
hf download $REPO --local-dir . --include "*.gguf" --include SHA256SUMS
sha256sum -c SHA256SUMS   # expect two OK lines

# 4c. byte count
stat -c '%n %s' *.gguf
# expect 1104605632 for the v2 file and 631713664 for the v1 file

# 4d. the card renders: open https://huggingface.co/$REPO and check the front matter
#     (license apache-2.0, base_model prism-ml/Ternary-Bonsai-2-27B-gguf) and the tables

# 5. make the repository public
hf repos settings $REPO --public
```

Then replace `<hf-user>` in the Bonsai-demo benchmark document and in the download snippet with the real account, and clean up:

```bash
rm -rf /tmp/hf-check
```
