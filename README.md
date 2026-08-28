# writing-style

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ci](https://github.com/parsakh00/writing-style-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/parsakh00/writing-style-lab/actions/workflows/ci.yml)

A writing skill for scientific papers. It measures a draft against published papers and
reports where the draft differs, in numbers: which phrases no paper uses and what papers
write in their place, how the draft cites, which of its habits papers do not share, and
where its sentences sit against the range of the field.

Everything in it was measured on 6 million words of published papers, and every rule
carries its count.

## Install

```
pip install git+https://github.com/parsakh00/writing-style-lab
```

The skill directory, `.claude/skills/writing-style/`, is also the package. Copy it into
an agent's skills folder and it works there unchanged. No dependencies.

## Use

```
writing-style DRAFT.md                   # score a paper draft
writing-style DRAFT.md --suggest         # for each phrase no paper uses, what papers write
writing-style DRAFT.md --reference group # one research group's own targets
writing-style DRAFT.md --register letter # cover letters; also: docs
```

From Python:

```python
from writingstyle import report
print(report(text, suggest=True))
```

## What it reports

- **Sequences.** Which of the draft's word sequences papers have used, and with
  `--suggest`, the continuation papers write after the same two words where the draft
  departs.
- **General rules**, each verified across the corpus: how a quantity noun takes its
  preposition, what a comparison is made with, where "therefore" goes, which
  abstract-noun claims papers make and which they do not, and how a method sentence is
  built.
- **Citing.** Citation density against the published range, and every author named to
  introduce a finding, which papers almost never do.
- **Register.** The stock hedges and attenuators that papers do not use, and the
  contrastive constructions that drafts overuse.
- **Structure.** Words before the verb and sentences opening on old information, after
  Gopen and Swan; clauses per sentence against the field's range.
- **Targets.** Sentence, vocabulary and register measures against the middle range of
  published papers. A real paper lands inside most of them and outside some; a draft
  inside all of them was written to the metric.

The full policy, with the numbers behind each rule, is `SKILL.md`.

## A beginning

This is a first version, built from one corpus. It will get better with other corpora,
other groups' profiles, and corrections from people who write in fields it has not
seen. A rule that looks wrong for your field is checked against the papers, not
argued: the skill measures the construction in the corpus and reports the count. Any
idea for improving it is welcome; see `CONTRIBUTING.md`.
