# Review Session Instructions

Two audiences: **you (session preparer)** and **collaborator (reviewer)**.

---

## Part 1 — Preparing and Sharing a Review Session

### Prerequisites

- Cut TIFFs and remapped annotation GeoJSONs must already exist under `data/cuts/`
  (i.e. the `extract_rois` and annotation remapping steps have completed).
- Run from JUWELS or any machine with access to the project environment and `data/cuts/`.

### Step 1: Prepare the session

On JUWELS, submit the SLURM job (recommended for large datasets):

```bash
sbatch slurm/prepare_review.sbatch
```

Or run interactively with custom settings:

```bash
export CUTS_DIR="data/cuts" OUTPUT_DIR="data/review_sessions" \
       N_PER_SIZE=10 TILE_SIZES="128 256 512 1024" SEED=42 ZIP=true
sbatch --export=ALL slurm/prepare_review.sbatch
```

Or run directly (e.g. for a quick smoke test):

```bash
uv run scripts/prepare_review_session.py \
  --cuts-dir data/cuts \
  --output-dir data/review_sessions \
  --tile-sizes 128 256 512 1024 \
  --n-per-size 10 \
  --seed 42 \
  --zip
```

`--n-per-size 10` produces 5 positive + 5 negative tiles per scale (40 tiles total across 4 scales).
`--zip` writes a `review_sessions/<session_id>.zip` alongside the session directory.

### Step 2: Locate the output

```
review_sessions/
└── <session_id>/          ← directory (e.g. 2026-06-25_001)
    ├── session.json
    └── tiles/
        ├── tile_0001.png
        ├── tile_0002.png
        └── ...
review_sessions/<session_id>.zip   ← if --zip was used
```

PNG filenames are opaque — they do not encode label, cut name, or position.

### Step 3: Bundle the app alongside the session

The collaborator needs the app files as well as the session data.
Create a single transfer bundle:

```bash
SESSION_ID="<session_id>"   # replace with the actual ID printed by the script

mkdir -p review_bundle/app
cp -r review_sessions/"$SESSION_ID" review_bundle/
cp app/review_tiles.py app/requirements.txt review_bundle/app/

zip -r review_bundle_"$SESSION_ID".zip review_bundle/
```

Send the collaborator `review_bundle_<session_id>.zip`.

Alternatively, if you used `--zip`, send them `review_sessions/<session_id>.zip`
**plus** `app/review_tiles.py` and `app/requirements.txt` separately.

> **Note on blinding**: `session.json` contains hidden ground-truth labels.
> The app never displays them, but the collaborator technically has access.
> If strict blinding is required, strip `ground_truth`, `n_oocytes_ground_truth`,
> and `annotation_ids` from each tile entry before sharing.

---

## Part 2 — Running the Review App (Collaborator Instructions)

### What you receive

A ZIP archive containing:

```
review_bundle/
├── app/
│   ├── review_tiles.py
│   └── requirements.txt
└── <session_id>/
    ├── session.json
    └── tiles/
        ├── tile_0001.png
        └── ...
```

### Requirements

- Python 3.10 or newer
- pip

### Step 1: Unzip and install dependencies

```bash
unzip review_bundle_<session_id>.zip
cd review_bundle
pip install -r app/requirements.txt
```

### Step 2: Run the app

```bash
streamlit run app/review_tiles.py -- --session <session_id>
```

Replace `<session_id>` with the name of the session folder inside the bundle
(e.g. `2026-06-25_001`).

Streamlit will print a local URL — open it in your browser:

```
  Local URL: http://localhost:8501
```

### Step 3: Label the tiles

- Each screen shows one tile and asks: **Does this image contain an oocyte?**
- Click **Yes — I see an oocyte** or **No — no oocyte here**.
- Your label is saved to `session.json` immediately after each click.
- If you close the app and reopen it, it resumes from where you left off.
- A completion screen appears when all tiles have been labelled.

**Do not move or rename** `session.json` or the `tiles/` folder while the app is running.

### Step 4: Return the labelled session

Send back the updated `<session_id>/session.json` (tile PNGs do not need to be returned).
The `collaborator_label` and `labelled_at` fields in that file contain your responses.
