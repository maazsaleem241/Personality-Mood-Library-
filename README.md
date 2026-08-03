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
- **No Unreal Engine, no `unreal` module** — this script is pure Python and runs completely independently of Task 2 (the Unreal-side baking stage).

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

## Configuration

Everything you need to change lives at the top of `decompose.py`, under `CONFIG`:

| Variable | What it is |
|---|---|
| `DATA_ROOT` | Path to the folder containing `data_groupa/` and `data_groupb/` |
| `GROUP_TAGS` | The full participant ID list for each group |
| `GROUP_LABEL` | Human-readable description of each personality group |
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
- **`validation/`** — the validation report plus all 4 charts, together in one folder. See below for what each chart means.

## Validation

Four checks run automatically every time, written to `validation/validation_report.md` with an accompanying chart for each. Real results from a full run are included in this repo at `validation/validation_report.md` — the explanations below tell you how to actually read each chart, not just what it checks.

### 1. Reconstruction check — `1_reconstruction_check.png`

A scatter plot: every (group, mood, channel) value's *original* absolute measurement on the X axis, its *reconstructed* value (`base + delta`) on the Y axis. Since `delta` is defined as `absolute − base`, adding them back together should always land you exactly back where you started — so **every single point should sit exactly on the diagonal red dashed line**. This chart isn't really about the data, it's a correctness check on the math itself: any point that drifts off the line means something in the pipeline is broken, not that the underlying data is unusual. In the real run, this passed cleanly for all 16 (group, mood) pairs.

### 2. Base comparison — `2_base_comparison.png`

A horizontal bar chart: the 15 channels with the highest resting-face activation in *either* group, coral bars for Group A, teal for Group B, both on the same scale so their heights are directly comparable. This is the chart that answers "what does each personality's face do by default, before any mood is layered on?"

Each bar gets a small text label showing how the two groups compare on that specific channel:
- **A number like `+220%`** means Group A's resting activation on that channel is 220% higher than Group B's on that same channel — calculated as `(A − B) / B × 100`, so B is always the reference point.
- **`A only`** or **`B only`** means the *other* group measured essentially zero on that channel — there's no meaningful percentage to compute against a zero denominator, so it's labeled explicitly instead of showing something like `+9000%`.
- **No label at all** means both groups measured essentially zero there — nothing to report either way.

In the real run: Group A's base is dominated by jaw/mouth-stretch channels (`jawForward`, `mouthStretchLeft/Right`), while Group B's is dominated by eye-squint channels (`eyeSquintLeft/Right`) — genuinely different resting facial signatures between the two groups, not just a scaled-up/down version of the same one.

### 3. Delta sparsity — `3_sparsity_heatmap.png`

A heatmap: 16 rows (one per group+mood combination), 52 columns (one per ARKit channel), brightness = how active that channel is for that mood. The brief's expectation is that each row should be **mostly dark with a handful of bright cells** — a mood should only meaningfully move a small subset of muscles beyond the resting base, not light up the whole face. If a row looks broadly bright across most columns instead of a few concentrated hotspots, that mood's signal likely isn't cleanly separated from noise.

The real run's counts (in `validation_report.md`, Section 3) mostly land in the expected range, though a few — Group A's Concentration (21 channels) and Aversion (17 channels) in particular — read as noticeably denser than the rest. Worth a second look at those two specifically before treating them as final.

### 4. Per-mood comparison — `4_per_mood_comparison.png`

Eight small charts, one per mood, each showing that mood's top 5 delta channels for Group A (coral) next to Group B (teal) side by side. This is the chart for asking "when both groups feel the same mood, do they express it through the same muscles, or different ones?"

Two patterns to look for: **overlap** (both groups' top channels are mostly the same, e.g. StartleResponse and AttentionalEngagement in the real run — both groups lead with `eyeWide*`/`browInnerUp` respectively) suggests that mood is expressed similarly across personalities, just at different intensities. **Divergence** (the top channels barely overlap, e.g. Concentration — Group A leads with `mouthSmile*`, Group B leads with `browDown*`) suggests the two groups are doing something genuinely different facially for that same labeled mood, which is worth a sanity check against the source video rather than assumed to be correct.

If the reconstruction check ever fails, or the script halts on a negative delta, something is genuinely wrong upstream — both are treated as hard errors, not warnings.
