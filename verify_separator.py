"""
verify_separator.py
-------------------
Validates separator.py classifier against existing data/processed/ files.

NOTE: Most "mismatches" are EXPECTED IMPROVEMENTS:
  The old parser dumped everything flat into destinations/.
  The new classifier correctly routes to destinations/treks, peaks, etc.
  True bugs are files that get misclassified to a WRONG category entirely.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path("src/utills").resolve()))
from separator import classify

PROCESSED = Path("data/processed")

# Map old flat categories to what the new system uses
# "destinations" == any destinations/* is a valid upgrade, not a mismatch
def is_correct(current: str, predicted: str) -> bool:
    # Exact match
    if current == predicted:
        return True
    # Old flat "destinations" + new granular destinations/* → upgrade, not error
    if current == "destinations" and predicted.startswith("destinations/"):
        return True
    # Old flat "insights" → testimonials or information is correct
    if current == "insights" and predicted in ("testimonials", "information"):
        return True
    return False

counts_by_pred  = defaultdict(int)
true_mismatches = []
upgrades        = []
total           = 0

for md_file in sorted(PROCESSED.rglob("*.md")):
    total += 1
    current_cat = str(md_file.parent.relative_to(PROCESSED)).replace("\\", "/")
    predicted   = classify(md_file)
    counts_by_pred[predicted] += 1

    if is_correct(current_cat, predicted):
        upgrades.append((md_file.name, current_cat, predicted))
    else:
        true_mismatches.append((md_file.name, current_cat, predicted))

# ── Print upgraded (flat → granular) ─────────────────────────────────────────
print(f"\n{'='*80}")
print("  GRANULAR UPGRADES (old flat category -> new sub-category)  [CORRECT]")
print(f"{'='*80}")
print(f"  {'FILE':<55} {'OLD':<20} {'NEW'}")
print(f"  {'-'*55} {'-'*20} {'-'*25}")
for name, old, new in upgrades:
    print(f"  {name:<55} {old:<20} {new}")

# ── Print true mismatches ─────────────────────────────────────────────────────
if true_mismatches:
    print(f"\n{'='*80}")
    print("  NEEDS REVIEW (possibly wrong category)")
    print(f"{'='*80}")
    print(f"  {'FILE':<55} {'OLD':<20} {'PREDICTED'}")
    print(f"  {'-'*55} {'-'*20} {'-'*25}")
    for name, old, new in true_mismatches:
        print(f"  {name:<55} {old:<20} {new}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  CLASSIFIER DISTRIBUTION")
print(f"{'='*80}")
for cat, n in sorted(counts_by_pred.items(), key=lambda x: -x[1]):
    bar = "#" * min(n, 50)
    print(f"  {cat:<35} {n:>4}  {bar}")

correct = len(upgrades)
print(f"\n  Total files           : {total}")
print(f"  Correctly classified  : {correct}  ({correct/total*100:.1f}%)")
print(f"  Needs review          : {len(true_mismatches)}")
