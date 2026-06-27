"""Generate the negation control set: minimal pairs where only the negation changes.
CheckList negation test, Ribeiro et al. 2020 — https://aclanthology.org/2020.acl-main.442/"""

import csv, json, os

NOUNS = ["movie", "book", "hotel", "restaurant", "meal",
         "phone", "laptop", "game", "show", "app"]
ADJS = ["great", "good", "nice", "wonderful", "excellent",
        "enjoyable", "pleasant", "impressive", "lovely", "satisfying"]

TEMPLATES = [
    ("simple", "The {n} was {a}.", "The {n} was not {a}."),
    ("distractor", "I expected the {n} to be {a}, and it was.",
                   "I expected the {n} to be {a}, but it was not."),
]
COLS = ["pair_id", "template", "cue", "negated", "gold_label", "text"]

rows, pid = [], 0
for name, pos, neg in TEMPLATES:
    for n in NOUNS:
        for a in ADJS:
            rows.append([pid, name, a, 0, "positive", pos.format(n=n, a=a)])
            rows.append([pid, name, a, 1, "negative", neg.format(n=n, a=a)])
            pid += 1

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "negation_control.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(COLS)
    w.writerows(rows)
with open(os.path.join(here, "negation_control.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(dict(zip(COLS, r))) + "\n")

print(f"{len(rows)} sentences, {pid} pairs")
