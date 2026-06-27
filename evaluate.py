"""Measure how often a sentiment model flips its label when a sentence is negated
(CheckList negation test). Usage: python3 evaluate.py [--hf]"""

import argparse, csv, os

POS = {"great", "good", "nice", "wonderful", "excellent", "enjoyable", "pleasant",
       "impressive", "lovely", "satisfying", "love", "like", "enjoy", "recommend",
       "appreciate", "adore", "fantastic", "awesome", "amazing", "best", "perfect",
       "brilliant", "terrific", "superb", "happy"}
NEG = {"bad", "terrible", "awful", "horrible", "worst", "hate", "dislike", "poor",
       "boring", "disappointing", "ugly", "sad", "broken", "slow", "useless"}


def lexicon(text):
    toks = [t.strip(".,!?;:'\"").lower() for t in text.split()]
    return "negative" if sum(t in NEG for t in toks) > sum(t in POS for t in toks) else "positive"


def hf(model):
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=model, truncation=True)
    return lambda text: "positive" if "pos" in clf(text)[0]["label"].lower() else "negative"


def report(name, predict, rows, pairs):
    acc = sum(predict(r["text"]) == r["gold_label"] for r in rows) / len(rows)
    per, eligible, flipped = {}, 0, 0
    for base, neg in pairs:
        per.setdefault(base["template"], [0, 0])
        if predict(base["text"]) == "positive":
            eligible += 1; per[base["template"]][1] += 1
            if predict(neg["text"]) == "negative":
                flipped += 1; per[base["template"]][0] += 1
    flip = flipped / eligible if eligible else 0.0
    print(f"\n{name}")
    print(f"  accuracy ........... {acc:6.1%}")
    print(f"  flip rate .......... {flip:6.1%}  ({flipped}/{eligible})")
    print(f"  failure rate ....... {1 - flip:6.1%}")
    for t, (f, e) in sorted(per.items()):
        print(f"    {t:10s} {f / e:6.1%}  ({f}/{e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "negation_control.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["pair_id"], {})[r["negated"]] = r
    pairs = [(p["0"], p["1"]) for p in by_id.values()]

    print(f"{len(rows)} sentences, {len(pairs)} pairs")
    report("lexicon (negation-blind)", lexicon, rows, pairs)
    if args.hf:
        try:
            predict = hf(args.model)
        except Exception as e:
            print(f"\n[--hf unavailable: {e}]")
            return
        report(args.model, predict, rows, pairs)


if __name__ == "__main__":
    main()
