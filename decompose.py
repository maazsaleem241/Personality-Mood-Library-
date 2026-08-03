"""
decompose.py

Task 1: turn raw per-participant ARKit capture into a parametric
base + delta mood library.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. For each group, for each participant, for each of the 8 moods:
     average that participant's real detected peak-event frames, then
     subtract that participant's OWN resting-face baseline (mean across
     their entire recording). A participant with zero detected events for
     a mood is treated as missing data (NaN), not a confirmed zero.
  2. Average across all participants in the group (NaN-aware - someone
     missing a mood doesn't drag the average toward zero).
  3. This gives the 16 "absolute mood profile vectors" (2 groups x 8
     moods) that the brief's Section 3.1 refers to as input.
  4. base[group] = per-channel MINIMUM across that group's 8 absolute
     mood vectors.
  5. delta[group][mood] = absolute[group][mood] - base[group]. Checked
     to be >= 0 everywhere; halts with a clear error if not (per spec).
  6. Writes 18 individual JSON files + one combined manifest.
  7. Runs all 4 validation checks and writes validation_report.md with
     4 accompanying charts.

NOTE: no contributor masking, no intensity boost - raw baseline-corrected
values only. See Task 1 working notes for why.

Run as a normal Python script (pandas/numpy/matplotlib only - no Unreal
dependency at all).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = r"D:\Intern work\JSON Manifest\Data"
GROUP_DIR = {"A": os.path.join(DATA_ROOT, "data_groupa"),
             "B": os.path.join(DATA_ROOT, "data_groupb")}

GROUP_TAGS = {
    "A": ["pr_1", "pr_4", "pr_8", "pr_15", "pr_17", "pr_22"],
    "B": ["pr_3", "pr_9", "pr_10", "pr_11", "pr_21", "pr_23"],
}

# Group A and B's label is given in the brief.
GROUP_LABEL = {
    "A": "Genuinely Kind / Expressive",
    "B": "TBD - Reserved / Stoic",
}

OUTPUT_DIR = r"D:\Intern work\JSON Manifest\facial_library\results"
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")  # holds both the report AND the charts

# Task 1 camelCase mood key -> the exact 'Expression' display string used
# in pr_n_event_evidence.csv (from the analysis script's own MAPPING dict)
MOOD_TO_EXPRESSION_DISPLAY = {
    "PositiveSocial": "Positive Social Expression",
    "TensionStress": "Tension/Stress",
    "Skepticism": "Skepticism",
    "StartleResponse": "Startle Response",
    "Concentration": "Concentration",
    "Dejection": "Dejection",
    "AttentionalEngagement": "Attentional Engagement",
    "Aversion": "Aversion",
}
MOODS = list(MOOD_TO_EXPRESSION_DISPLAY.keys())

# Canonical 52 ARKit channel names, exactly as they appear as columns in
# pr_n.csv (PascalCase). camelCase JSON keys are derived from these.
ARKIT_52 = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft", "MouthSmileLeft", "MouthSmileRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight",
    "TongueOut",
]

MAX_EVENTS_PER_MOOD = 100  # matches the convention used in earlier group-average work


def to_camel(name):
    return name[0].lower() + name[1:]


# ============================================================
# STEP 1: per-participant, per-mood absolute vectors
# ============================================================

def compute_participant_profiles(participant_id, group_dir):
    """Returns dict: mood_key -> pd.Series (52 ARKit channels) or None if
    this participant had zero valid events for that mood."""
    csv_path = os.path.join(group_dir, f"{participant_id}.csv")
    ev_path = os.path.join(group_dir, f"{participant_id}_event_evidence.csv")

    if not (os.path.exists(csv_path) and os.path.exists(ev_path)):
        print(f"    !! Missing files for {participant_id} in {group_dir} - skipping participant")
        return {mood: None for mood in MOODS}

    df_shapes = pd.read_csv(csv_path)
    df_ev = pd.read_csv(ev_path)

    baseline_vector = df_shapes[ARKIT_52].mean()

    profiles = {}
    for mood, display_name in MOOD_TO_EXPRESSION_DISPLAY.items():
        df_filtered = df_ev[df_ev["Expression"] == display_name].head(MAX_EVENTS_PER_MOOD)

        valid_frames = [int(row_ev["Frame Index"]) for _, row_ev in df_filtered.iterrows()]
        valid_frames = [f for f in valid_frames if f < len(df_shapes)]

        if valid_frames:
            mean_vector = (df_shapes.iloc[valid_frames][ARKIT_52].mean() - baseline_vector).clip(lower=0.0)
            profiles[mood] = mean_vector
        else:
            profiles[mood] = None  # missing data, not zero

    return profiles


# ============================================================
# STEP 2: group-average, NaN-aware
# ============================================================

def build_group_absolute_profiles(group, group_tags_override=None):
    group_dir = GROUP_DIR[group]
    tags = group_tags_override if group_tags_override is not None else GROUP_TAGS[group]

    per_participant = {}
    for pid in tags:
        print(f"  Processing {pid}...")
        per_participant[pid] = compute_participant_profiles(pid, group_dir)

    absolute_profiles = {}
    for mood in MOODS:
        vectors = [per_participant[pid][mood] for pid in tags if per_participant[pid][mood] is not None]
        if vectors:
            stacked = pd.concat(vectors, axis=1)
            absolute_profiles[mood] = stacked.mean(axis=1).fillna(0.0)
        else:
            print(f"    !! No participant in Group {group} had any valid '{mood}' events - defaulting to all-zero")
            absolute_profiles[mood] = pd.Series(0.0, index=ARKIT_52)

    return absolute_profiles


# ============================================================
# STEP 3: decompose into base + delta
# ============================================================

def decompose_group(absolute_profiles):
    stacked = pd.concat(absolute_profiles.values(), axis=1)
    stacked.columns = list(absolute_profiles.keys())

    base = stacked.min(axis=1)

    deltas = {}
    for mood in MOODS:
        delta = absolute_profiles[mood] - base
        negative = delta[delta < -1e-9]
        if len(negative) > 0:
            raise RuntimeError(
                f"NEGATIVE DELTA DETECTED for mood '{mood}' - this should be mathematically "
                f"impossible since base is the min across all moods. Channels affected: "
                f"{dict(negative)}. Stopping per spec - investigate before proceeding."
            )
        deltas[mood] = delta

    return base, deltas


if __name__ == "__main__":
    print("=== Task 1: Parametric Mood Library Decomposition ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    all_absolute, all_base, all_delta = {}, {}, {}

    for group in ["A", "B"]:
        print(f"--- Group {group} ---")
        absolute_profiles = build_group_absolute_profiles(group)
        base, deltas = decompose_group(absolute_profiles)
        all_absolute[group] = absolute_profiles
        all_base[group] = base
        all_delta[group] = deltas
        print(f"  Group {group} done.\n")

    # ============================================================
    # SAVE 18 INDIVIDUAL JSON FILES + MANIFEST
    # ============================================================
    def vector_to_camel_dict(vec):
        return {to_camel(k): round(float(v), 6) for k, v in vec.items()}

    manifest = {"schema_version": "1.0", "groups": {}}

    for group in ["A", "B"]:
        base_dict = vector_to_camel_dict(all_base[group])
        with open(os.path.join(OUTPUT_DIR, f"base_group{group}.json"), "w") as f:
            json.dump(base_dict, f, indent=2)

        mood_dicts = {}
        for mood in MOODS:
            delta_dict = vector_to_camel_dict(all_delta[group][mood])
            mood_dicts[mood] = delta_dict
            with open(os.path.join(OUTPUT_DIR, f"delta_group{group}_{mood}.json"), "w") as f:
                json.dump(delta_dict, f, indent=2)

        manifest["groups"][group] = {
            "label": GROUP_LABEL[group],
            "participants": GROUP_TAGS[group],
            "base": base_dict,
            "moods": mood_dicts,
        }

    with open(os.path.join(OUTPUT_DIR, "mood_library_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved 18 individual JSON files + manifest to {OUTPUT_DIR}")

    # ============================================================
    # VALIDATION
    # ============================================================
    report_lines = ["# Task 1 Validation Report\n"]

    # --- Check 1: reconstruction ---
    recon_errors = []
    recon_x, recon_y = [], []
    for group in ["A", "B"]:
        for mood in MOODS:
            reconstructed = all_base[group] + all_delta[group][mood]
            original = all_absolute[group][mood]
            diff = (reconstructed - original).abs()
            recon_x.extend(original.values)
            recon_y.extend(reconstructed.values)
            if diff.max() > 1e-6:
                recon_errors.append((group, mood, diff.max()))

    report_lines.append("## 1. Reconstruction Check")
    if recon_errors:
        report_lines.append(f"**FAILED** - {len(recon_errors)} (group, mood) pairs did not reconstruct exactly:")
        for g, m, e in recon_errors:
            report_lines.append(f"- Group {g}, {m}: max error {e:.8f}")
    else:
        report_lines.append("**PASSED** - base + delta reproduces the original absolute profile exactly (within float tolerance) for all 16 (group, mood) pairs.")
    report_lines.append("")

    plt.figure(figsize=(6, 6))
    plt.scatter(recon_x, recon_y, s=8, alpha=0.4, color="#444444")
    lims = [0, max(max(recon_x), max(recon_y)) * 1.05]
    plt.plot(lims, lims, "--", color="red", linewidth=1, label="y = x (perfect reconstruction)")
    plt.xlabel("Original absolute value")
    plt.ylabel("Reconstructed (base + delta)")
    plt.title("Reconstruction Check: base + delta vs. original")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, "1_reconstruction_check.png"), dpi=150)
    plt.close()

    # --- Check 2: top 5 base channels per group + comparison chart ---
    report_lines.append("## 2. Top 5 Base Channels Per Group")
    for group in ["A", "B"]:
        top5 = all_base[group].sort_values(ascending=False).head(5)
        report_lines.append(f"**Group {group}:**")
        for ch, v in top5.items():
            report_lines.append(f"- {to_camel(ch)}: {v:.4f}")
        report_lines.append("")

    COLOR_A, COLOR_B = "#FF7F50", "#2E8B8B"  # coral, teal - used consistently in all charts
    combined_top = pd.concat([all_base["A"], all_base["B"]], axis=1)
    combined_top.columns = ["A", "B"]
    combined_top["max_val"] = combined_top.max(axis=1)
    top_channels = combined_top.sort_values("max_val", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top_channels))
    ax.barh(y_pos - 0.2, top_channels["A"], height=0.4, color=COLOR_A, label="Group A")
    ax.barh(y_pos + 0.2, top_channels["B"], height=0.4, color=COLOR_B, label="Group B")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([to_camel(c) for c in top_channels.index])
    ax.invert_yaxis()
    ax.set_xlabel("Base activation")
    ax.set_title("Personality Base Comparison: Group A vs Group B (top 15 channels)")

    # Fixed room for annotation text: without this, labels on the longest
    # bars get clipped by the plot border instead of sitting past the bar.
    max_bar_val = top_channels[["A", "B"]].values.max()
    ax.set_xlim(0, max_bar_val * 1.35)
    ax.legend(loc="lower right")

    for i, ch in enumerate(top_channels.index):
        a_val, b_val = top_channels.loc[ch, "A"], top_channels.loc[ch, "B"]
        if a_val <= 0.001 and b_val <= 0.001:
            continue  # both genuinely zero - nothing to compute, not a judgment call

        x_pos = max(a_val, b_val) + max_bar_val * 0.03

        if b_val <= 0.001 and a_val > 0.001:
            label = "A only"
        elif a_val <= 0.001 and b_val > 0.001:
            label = "B only"
        else:
            pct = (a_val - b_val) / b_val * 100
            label = f"{pct:+.0f}%"

        ax.text(x_pos, i, label, va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, "2_base_comparison.png"), dpi=150)
    plt.close()

    # --- Check 3: delta sparsity ---
    report_lines.append("## 3. Delta Sparsity (channels active above 0.05)")
    sparsity_matrix = pd.DataFrame(index=ARKIT_52)
    for group in ["A", "B"]:
        for mood in MOODS:
            col_name = f"{group}_{mood}"
            sparsity_matrix[col_name] = all_delta[group][mood]
            active_count = (all_delta[group][mood] > 0.05).sum()
            report_lines.append(f"- Group {group}, {mood}: {active_count} channels above 0.05")
    report_lines.append("")

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(sparsity_matrix.T.values, aspect="auto", cmap="magma", vmin=0)
    ax.set_yticks(np.arange(len(sparsity_matrix.columns)))
    ax.set_yticklabels(sparsity_matrix.columns, fontsize=8)
    ax.set_xticks(np.arange(len(ARKIT_52)))
    ax.set_xticklabels([to_camel(c) for c in ARKIT_52], rotation=90, fontsize=6)
    ax.set_title("Delta Sparsity: 16 (group, mood) rows x 52 channels")
    plt.colorbar(im, ax=ax, label="Delta activation", shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, "3_sparsity_heatmap.png"), dpi=150)
    plt.close()

    # --- Check 4: per-mood top-5 comparison, small multiples ---
    report_lines.append("## 4. Per-Mood Top 5 Delta Channels: Group A vs Group B")
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    axes = axes.flatten()
    for idx, mood in enumerate(MOODS):
        ax = axes[idx]
        a_delta, b_delta = all_delta["A"][mood], all_delta["B"][mood]
        top_union = pd.concat([a_delta, b_delta], axis=1)
        top_union.columns = ["A", "B"]
        top_union["max_val"] = top_union.max(axis=1)
        top5_mood = top_union.sort_values("max_val", ascending=False).head(5)

        y_pos = np.arange(len(top5_mood))
        ax.barh(y_pos - 0.2, top5_mood["A"], height=0.4, color=COLOR_A, label="Group A")
        ax.barh(y_pos + 0.2, top5_mood["B"], height=0.4, color=COLOR_B, label="Group B")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([to_camel(c) for c in top5_mood.index], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(mood, fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8)

        report_lines.append(f"**{mood}:**")
        report_lines.append(f"- Group A top 5: {', '.join(to_camel(c) for c in a_delta.sort_values(ascending=False).head(5).index)}")
        report_lines.append(f"- Group B top 5: {', '.join(to_camel(c) for c in b_delta.sort_values(ascending=False).head(5).index)}")
        report_lines.append("")

    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, "4_per_mood_comparison.png"), dpi=150)
    plt.close()

    report_lines.append("## Charts")
    report_lines.append("![Reconstruction check](1_reconstruction_check.png)")
    report_lines.append("![Base comparison](2_base_comparison.png)")
    report_lines.append("![Sparsity heatmap](3_sparsity_heatmap.png)")
    report_lines.append("![Per-mood comparison](4_per_mood_comparison.png)")

    with open(os.path.join(VALIDATION_DIR, "validation_report.md"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"Validation report + 4 charts saved to {OUTPUT_DIR}")
    print("\n=== Done ===")