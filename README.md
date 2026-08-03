# Parametric Mood Library — Task 1

Decomposes 16 absolute facial-expression profiles (2 personality groups × 8 moods) into a compact **base + delta** structure for driving a real-time MetaHuman AI agent. This is the data-generation stage; a later, separate stage consumes this output inside Unreal Engine.

## The idea, briefly

Instead of storing 16 independent, unrelated poses, each group gets:
- **One base** — a per-channel *minimum* across that group's 8 moods. Represents the personality's always-on resting expression: whatever a channel never drops below, regardless of mood, isn't really "part of" any one mood — it's just how that face sits normally.
- **Eight deltas** — `delta = absolute − base`, one per mood. What that specific mood adds *on top of* the resting face. Always ≥ 0 by construction.

At runtime: `final_face = base + delta[mood] × blend_weight`, where `blend_weight` ramps 0→1→0 as a mood is entered and exited. This is why the decomposition matters — it lets the resting personality run continuously while individual moods fade in and out additively, rather than snapping between 16 disconnected absolute poses.

## Requirements

- Python 3
- `pandas`, `numpy`, `matplotlib` (`pip install pandas numpy matplotlib`)

## Data you need before running this

Raw per-participant facial capture, already produced by a separate upstream analysis pipeline (not included in this repo). For **every** participant in both groups, two CSV files:

| File | Contents |
|---|---|
| `pr_n.csv` | Raw per-frame ARKit blendshape values for participant `pr_n` — one row per captured video frame, 52 blendshape columns plus timing metadata |
| `pr_n_event_evidence.csv` | Detected expression "events" for that participant — columns include `Event ID`, `Expression`, `Frame Index`, `Peak Intensity` |

These need to sit in two folders, one per group:

```
<DATA_ROOT>/
├── data_groupa/
│   ├── pr_1.csv
│   ├── pr_1_event_evidence.csv
│   ├── pr_4.csv
│   ├── pr_4_event_evidence.csv
│   └── ... (one pair per Group A participant)
└── data_groupb/
    ├── pr_3.csv
    ├── pr_3_event_evidence.csv
    └── ... (one pair per Group B participant)
```

An optional `outliers_log.csv` (columns: `Participant`, `Event_ID`, `Status`) can flag specific detected events to exclude — the script skips this gracefully if the file doesn't exist.

## Configuration — edit these before running

Everything you need to change lives at the top of `decompose.py`, under `CONFIG`:

| Variable | What it is |
|---|---|
| `DATA_ROOT` | Path to the folder containing `data_groupa/` and `data_groupb/` |
| `GROUP_TAGS` | The full participant ID list for each group |
| `GROUP_LABEL` | Human-readable description of each personality group — **Group B's is currently a placeholder, fill in the real one before publishing results** |
| `OUTPUT_DIR` | Where all output (JSON files, manifest, validation report, charts) gets written |
| `OUTLIER_LOG_FILE` | Path to the optional outlier-exclusion CSV |

## How to run it

```bash
python decompose.py
```

That's it — no arguments. It prints progress per participant as it goes, then a final summary of what got written and where.

## What it actually does, step by step

1. **Per participant, per mood**: averages that participant's real detected peak-event frames, then subtracts *that participant's own* resting-face baseline (the mean across their entire recording). A participant with zero detected events for a mood is treated as *missing data*, not a confirmed zero — this matters for step 2.
2. **Per group, per mood**: averages across all participants in that group who actually had data for that mood (missing participants are skipped, not counted as zero — otherwise moods only a few people displayed get diluted toward invisible).
3. This produces the **16 absolute mood profiles** (2 groups × 8 moods).
4. **Base** = per-channel minimum across each group's 8 absolute profiles.
5. **Delta** = `absolute − base` per mood, with a hard check that halts the script if anything comes out negative (shouldn't be mathematically possible given how base is defined, but it's verified rather than assumed).

No contributor-masking or artificial intensity scaling is applied anywhere — every value is a genuine, unmodified measurement from real data.

## What it outputs

All written to `OUTPUT_DIR`:

- **18 individual JSON files** — `base_groupA.json`, `base_groupB.json`, and 16 `delta_group{A,B}_{MoodName}.json` files (e.g. `delta_groupA_PositiveSocial.json`). Each is a flat dictionary of the 52 ARKit channel names (exact camelCase, e.g. `mouthSmileLeft`) to their values.
- **`mood_library_manifest.json`** — everything combined into one file, structured as:
  ```json
  {
    "schema_version": "1.0",
    "groups": {
      "A": { "label": "...", "participants": [...], "base": {...}, "moods": { "PositiveSocial": {...}, ... } },
      "B": { ... }
    }
  }
  ```
- **`validation_report.md`** + **`validation_charts/`** — see below.

## Validation

Four checks run automatically every time, each with an accompanying chart:

1. **Reconstruction check** — confirms `base + delta` exactly reproduces the original absolute profile for all 16 (group, mood) pairs. Shown as a scatter plot; every point should land exactly on the diagonal.
2. **Base comparison** — horizontal bar chart of each group's top channels, side by side, with % difference called out on the bars that diverge most between groups.
3. **Delta sparsity** — a 52-channel × 16-mood heatmap. Deltas are expected to be sparse (most channels near zero for any given mood); this makes that visible at a glance rather than trusting a raw count.
4. **Per-mood comparison** — one small chart per mood, Group A's top 5 active channels vs Group B's for that same mood.

If the reconstruction check ever fails, or the script halts on a negative delta, something is genuinely wrong upstream — both are treated as hard errors, not warnings.

## Known caveats

- `GROUP_LABEL["B"]` is a placeholder and needs a real value before this is considered final.
- Validated end-to-end against a partial dataset (3 of the intended participants) during development — logic is confirmed correct, but the actual output numbers will shift once run against the full participant rosters.
