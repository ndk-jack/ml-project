from pathlib import Path
import json
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "experiments" / "reports"

INPUT_FILES = {
    "label_7d": REPORTS_DIR / "feature_importance_label_7d.json",
    "label_30d": REPORTS_DIR / "feature_importance_label_30d.json",
}

OUTPUT_DIR = REPORTS_DIR / "feature_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep geography for now, deprioritize background_rate_yr only if weak
FORCE_KEEP = {
    "latitude",
    "longitude",
    "magnitude",
    "count_1d",
    "count_7d",
    "count_30d",
    "count_90d",
    "energy_1d",
    "energy_7d",
    "energy_30d",
    "energy_90d",
    "b_value_7d",
    "b_value_90d",
    "mag_mean_90d",
}

FORCE_DROP = {
    "background_rate_yr",  # deprioritized after ablation
}

TOP_N_PER_TARGET = 20


def main():
    aggregate = defaultdict(lambda: {
        "total_gain": 0.0,
        "targets": [],
        "rank_sum": 0,
        "count": 0,
    })

    by_target = {}

    for target, path in INPUT_FILES.items():
        data = json.loads(path.read_text())
        top = data[:TOP_N_PER_TARGET]
        by_target[target] = top

        for rank, row in enumerate(top, start=1):
            feat = row["feature"]
            gain = float(row["importance_gain"])
            aggregate[feat]["total_gain"] += gain
            aggregate[feat]["targets"].append(target)
            aggregate[feat]["rank_sum"] += rank
            aggregate[feat]["count"] += 1

    ranked = []
    for feat, info in aggregate.items():
        ranked.append({
            "feature": feat,
            "total_gain": info["total_gain"],
            "targets": sorted(info["targets"]),
            "avg_rank": info["rank_sum"] / info["count"],
            "count": info["count"],
        })

    ranked = sorted(ranked, key=lambda x: (-x["count"], -x["total_gain"], x["avg_rank"]))

    candidate_features = []
    for row in ranked:
        feat = row["feature"]
        if feat in FORCE_DROP:
            continue
        candidate_features.append(feat)

    candidate_features = sorted(set(candidate_features).union(FORCE_KEEP))

    report = {
        "input_files": {k: str(v) for k, v in INPUT_FILES.items()},
        "top_n_per_target": TOP_N_PER_TARGET,
        "force_keep": sorted(FORCE_KEEP),
        "force_drop": sorted(FORCE_DROP),
        "ranked_features": ranked,
        "candidate_feature_set_v1": candidate_features,
        "candidate_feature_count": len(candidate_features),
    }

    out_json = OUTPUT_DIR / "feature_selection_report_v1.json"
    out_txt = OUTPUT_DIR / "feature_selection_report_v1.txt"

    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("FEATURE SELECTION REPORT V1")
    lines.append("")
    lines.append("Top aggregated candidate features:")
    for row in ranked[:25]:
        lines.append(
            f"{row['feature']} | total_gain={row['total_gain']:.2f} | "
            f"targets={','.join(row['targets'])} | avg_rank={row['avg_rank']:.2f}"
        )

    lines.append("")
    lines.append("Candidate feature set v1:")
    for feat in candidate_features:
        lines.append(feat)

    out_txt.write_text("\n".join(lines))

    print("report_json", out_json)
    print("report_txt", out_txt)
    print("candidate_feature_count", len(candidate_features))
    print("top_15_candidates")
    for feat in candidate_features[:15]:
        print(feat)


if __name__ == "__main__":
    main()
