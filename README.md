# writing-style

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ci](https://github.com/parsakh00/writing-style-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/parsakh00/writing-style-lab/actions/workflows/ci.yml)

A writing skill for scientific papers. It measures a draft against published papers and
reports where the draft differs, in numbers: which phrases no paper uses and what papers
write in their place, how the draft cites, which of its habits papers do not share, and
where its sentences sit against the range of the field.

Everything in it was measured on 6 million words of published papers, and every rule
carries its count.

Try it in the browser, without installing anything: [check a draft](https://parsakh00.github.io/writing-style-lab/check.html).

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

- **Sequences.** Which phrases papers use, and what they write where the draft departs.
- **Rules.** Constructions papers use and constructions they do not, each with its count.
- **Citing.** How often the draft cites, and how.
- **Register.** Hedges and qualifiers that papers leave out.
- **Structure.** Sentence shape, after Gopen and Swan.
- **Targets.** Where the draft sits against the middle range of published papers.

The full policy, with the numbers, is `SKILL.md`.

## A beginning

This is a first version, built from one corpus. It will get better with other corpora,
other groups' profiles, and corrections from people who write in fields it has not
seen. A rule that looks wrong for your field is checked against the papers, not
argued: the skill measures the construction in the corpus and reports the count. Any
idea for improving it is welcome; see `CONTRIBUTING.md`. Feedback on a checked draft goes
through the [feedback form](https://github.com/parsakh00/writing-style-lab/issues/new?template=feedback.yml),
and every report is measured against the corpus before a rule changes.
