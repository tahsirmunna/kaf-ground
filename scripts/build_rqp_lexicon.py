"""
Derive the RadGraph Query Pruning lexicon from the RadGraph corpus.

    python scripts/build_rqp_lexicon.py \
        --radgraph data/radgraph/MIMIC-CXR_graphs.json \
        --exclude data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.json \
        --out kafground/inference/rqp_modifiers.txt \
        --roles runs/rqp_roles.json            # optional, for the token-role ablations

Every entity token in the training reports is assigned a role:

    anat      most often labelled ANAT-DP (location, laterality)
    modifier  OBS-* token that is the source of a `modify` relation in >50% of uses
    finding   any other OBS-* token

The pruning list keeps the modifiers that are both almost always used that way
(modify share >= 0.90) and well attested (>= 100 entities). Studies that
appear in MS-CXR are excluded, so the evaluation set contributes nothing to
the lexicon. Only the query phrase is pruned at test time; report graphs of
test studies are never read.
"""
import argparse
import collections
import json
import re
from pathlib import Path

STUDY = re.compile(r"(p\d+/p\d+/s\d+)")
MODIFY_PURITY, MIN_ATTESTED = 0.90, 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radgraph", required=True)
    ap.add_argument("--exclude", required=True, help="MS-CXR annotation json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roles", default=None, help="also write {token: role} json")
    a = ap.parse_args()

    mscxr = json.load(open(a.exclude))
    excl = {m.group(1) for im in mscxr["images"] if (m := STUDY.search(im["path"]))}
    print(f"excluding {len(excl)} MS-CXR studies")

    labels = collections.defaultdict(collections.Counter)
    rels = collections.defaultdict(collections.Counter)
    used = 0
    for key, report in json.load(open(a.radgraph)).items():
        m = STUDY.search(key)
        if m and m.group(1) in excl:
            continue
        used += 1
        for e in report.get("entities", {}).values():
            t = e["tokens"].lower()
            labels[t][e["label"]] += 1
            rs = e.get("relations") or []
            if not rs:
                rels[t]["<none>"] += 1
            for r in rs:
                rels[t][r[0]] += 1
    print(f"{used} reports, {len(labels)} distinct entity tokens")

    roles, modifiers = {}, []
    for t, lc in labels.items():
        top = lc.most_common(1)[0][0]
        n_rel = sum(rels[t].values()) or 1
        share = rels[t].get("modify", 0) / n_rel
        n = sum(lc.values())
        role = "anat" if top == "ANAT-DP" else "modifier" if share > 0.5 else "finding"
        roles[t] = role
        if role == "modifier" and share >= MODIFY_PURITY and n >= MIN_ATTESTED and " " not in t:
            modifiers.append(t)

    header = ("# RadGraph Query Pruning lexicon: tokens whose RadGraph entities are almost always\n"
              "# the source of a `modify` relation (share >= 0.90, attested >= 100 times) in the\n"
              "# MIMIC-CXR training reports, MS-CXR studies excluded. Regenerate with\n"
              "# scripts/build_rqp_lexicon.py.\n")
    Path(a.out).write_text(header + "\n".join(sorted(modifiers)) + "\n")
    print(f"{len(modifiers)} pruning tokens -> {a.out}")
    if a.roles:
        Path(a.roles).parent.mkdir(parents=True, exist_ok=True)
        Path(a.roles).write_text(json.dumps(roles))
        print(f"roles for {len(roles)} tokens -> {a.roles}")


if __name__ == "__main__":
    main()
