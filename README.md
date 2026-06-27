# Does a sentiment model actually read the word "not" or does it learn it by proxies?

A control dataset of 400 sentences (200 minimal pairs) that is testing one property of a
sentiment classifier: does it respond to negation, or does it just react to a strong
word? Each pair contains 1 negation of the:

```
The movie was great.       -> positive
The movie was not great.   -> negative
```

- Dataset: `[negation_control.csv](negation_control.csv)` / `[negation_control.jsonl](negation_control.jsonl)` (400 sentences, 200 pairs)
- Generator: `[generate.py](generate.py)` (Python stdlib only, deterministic)
- Evaluator: `[evaluate.py](evaluate.py)`

## The hypothesis and the paper

Negation Sensitivity: a sentiment model should understand that "not" reverses sentiment, in contrast to labelling any sentence that contains "strong words" ("super", "great", "love") as positive.

This is taken from CheckList:

> Ribeiro, Wu, Guestrin, Singh (2020). *Beyond Accuracy: Behavioral Testing of NLP
> Models with CheckList.* ACL 2020. [https://arxiv.org/abs/2005.04118](https://arxiv.org/abs/2005.04118)

Checklist looks at one model capability at a time, like unit testing. One of those tests is a negation test built from mix-and-match sentences. It tests one model capability at a time, the way you unit-test software. One of
those capabilities is a negation test built from small templated sentences;

### Hypothesis:

If a model understands a sentence, then it should understand its opposite as well. If X was great then X was not great should flip the prediction from positive to negative. If the model only looks at strong words, then the prediction will not flip.

## Why it is a control dataset


|                       | sentence                                           | gold     |
| --------------------- | -------------------------------------------------- | -------- |
| base (`negated=0`)    | The hotel was wonderful.                           | positive |
| negated (`negated=1`) | The hotel was not wonderful.                       | negative |
| negated (`negated=1`) | The hotel was expected to be great, but it was not | negative |


In the pair, the noun, the sentimend word, the tense and everything else is identical. We only add a negation or in the second case, a string of distractors, such as showing the negator later in the phrase.

Within a pair the noun, the sentiment word, the tense and the structure are identical;
only the negation differs. So if a model calls the base positive but fails to flip the
negated version, the negation is the only thing that can be responsible.

## The data


| Column       | Meaning                                                      |
| ------------ | ------------------------------------------------------------ |
| `pair_id`    | Shared by the two sentences of a pair.                       |
| `template`   | `simple` or `distractor`.                                    |
| `cue`        | The strong positive word held fixed across the pair.         |
| `negated`    | `0` = base, `1` = negated.                                   |
| `gold_label` | `positive` for the base, `negative` for the negated version. |
| `text`       | The sentence.                                                |


Rows from the file:

```csv
pair_id,template,cue,negated,gold_label,text
0,simple,great,0,positive,The movie was great.
0,simple,great,1,negative,The movie was not great.
100,distractor,great,0,positive,"I expected the movie to be great, and it was."
100,distractor,great,1,negative,"I expected the movie to be great, but it was not."
```

## How it was generated

`generate.py`:
10 nouns × 10 positive adjectives × 2 templates = 200 pairs = 400 sentences.

For item each, we also get a negated version, in order to form a pair.

Can be run with:

```bash
python3 generate.py
```

## Does the control work?

A control dataset have the ground truth and serves as testing. If the property for which we are testing is missing, than it fails the model. The metric we follow is the **flipt rate**: how many sentences did it flip to negative once a type of negation was applied. 

A control dataset should fail models that lack the property and pass models that have
it. 

Two reference models:

- Negative control: a bag-of-words lexicon classifier 
- Positive control: `distilbert-base-uncased-finetuned-sst-2-english`,


| Model                    | accuracy | flip · `simple` | flip · `distractor` | failure rate |
| ------------------------ | -------- | --------------- | ------------------- | ------------ |
| Lexicon (negation-blind) | 50.0%    | 0% (0/100)      | 0% (0/100)          | 100%         |
| DistilBERT (SST-2)       | 100%     | 100% (100/100)  | 100% (100/100)      | 0%           |


The lexicon model labels "not great" the same as "great". It fails EVERYT time. DistilBERT behaves well and flips for every pair, so the dataset gives off no false alarm on a model that understands negation.  The contrast shows that the data-set isolates negation.



How to run:

```bash
python3 evaluate.py          # lexicon baseline


uv venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv transformers
.venv/bin/python evaluate.py --hf
```

## Limitations

- We test the easiest forms of negation, where "not" sits in front of the sentiment word. But what happens if there is double negation or it sits somewhere else? What happens if there are contractions ("can't") ? or jargon? The dataset would need to be extended.

## Links

- Repository: [https://github.com/AlexandruMititelu/negation-control](https://github.com/AlexandruMititelu/negation-control)
- Dataset: `[negation_control.csv](negation_control.csv)`
- Generator: `[generate.py](generate.py)`
- Evaluator: `[evaluate.py](evaluate.py)`

## Reference

Ribeiro, Wu, Guestrin, Singh. *Beyond Accuracy: Behavioral Testing of NLP Models with
CheckList.* ACL 2020. [https://aclanthology.org/2020.acl-main.442/](https://aclanthology.org/2020.acl-main.442/)
Validation model: `distilbert-base-uncased-finetuned-sst-2-english`,
[https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).