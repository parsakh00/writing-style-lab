---
name: writing-style
description: Use when drafting or revising prose the user will publish or submit - papers, manuscripts, reports, proposals, documentation - or when they say a draft reads like AI, sounds generic, flat, or needs to sound more human. Also use when asked to score a draft against a style corpus or hit measured style targets.
---

# writing-style

Targets measured from 6 million words of published papers. Self-contained: copy this
directory anywhere and it works.

```
python check.py DRAFT.md
```

No dependencies beyond the Python standard library. The bundled `data/` holds the style
reference (615 excerpts from 615 adsorption and simulation papers), a second profile for one
research group (90 excerpts from 19 of their papers), a vocabulary of 31,550 word types,
the 300 connective formulas papers rely on, 77,988 connective word sequences, and
139,258 word triples with counts for `--suggest`.

**Calibration: a real published paper scores 6 to 8 of 13 targets.** The bands are p25
to p75, so half the corpus sits outside any given one. Thirteen out of thirteen means the
draft was written to the metric, not to the reader.

If the full repository is present, `scripts/score.py` adds parser-based syntax features
and a second reference profile for one research group.

## The register rule

Two kinds of contrast, and only one is a tell.

**Stock hedges are absent from papers.** Measured at 0.000 per 1000 words across 786,313
words: "at the expense of", "that said", "in exchange for", "is not to say", "to be
fair", "the flip side". Attenuators likewise sit near zero: "for now", "arguably", "to some extent", "we
would not". Never write these. ("In practice", "it is worth" and "it should be noted"
were once on this list; published authors use them, and they are not.)

**Contrastive specification is normal and is overused.** Papers use "rather than",
"while the", "on the other hand" and "whereas" at **0.90 per 1000 words combined**. An
unedited draft runs **5.11**, and applying the rest of this skill only brings it to 3.13.
That is the most resistant habit measured here.

The distinction is what the contrast does. Specifying a mechanism earns it:

> attributed to accessibility rather than to number
> physisorption rather than covalent attachment

Softening a claim does not:

> the gain is real, though X is lost
> useful, while the cost is significant

**Target: at most one contrastive construction per thousand words.** Where a second
wants to appear, state the two facts as two sentences. "The gain is real. X is lost."

Concessives are separately normal at 0.8 to 2.6 per 1000. It is the attachment to a
claim that reads wrong, not the qualification itself.

An earlier version of this section merged the two groups and reported drafts at 3 to 6
per 1000, mostly on the strength of contrast papers use freely. The measure was wrong
and the rule derived from it was too broad.

## Targets

Ranges are p25 to p75. Where the corpora differ, both are given; the group range is the
narrower target when writing for that group.

Both columns are measured by `check.py` itself on 1200-word excerpts, so a draft and
its band are the same quantity. `--reference group` switches the column.

| feature | corpus | that group |
|---|---|---|
| stock hedges /1000w | 0.00 - 0.00 | 0.00 - 0.00 |
| contrastive constructions /1000w | 0.00 - 0.85 | 0.00 - 1.42 |
| attenuators /1000w | 0.00 - 0.00 | 0.00 - 0.85 |
| concessives /1000w | 0.85 - 2.57 | 0.86 - 2.96 |
| self-qualifying sentences | 0.02 - 0.07 | 0.02 - 0.08 |
| passive per clause | 0.37 - 0.60 | 0.24 - 0.39 |
| first person /1000w | 0.9 - 5.9 | 4.3 - 12.0 |
| mean sentence length | 21.6 - 27.0 | 25.0 - 30.1 |
| sentence length IQR | 11 - 16 | 16 - 22 |
| commas /1000w | 50 - 73 | 51 - 68 |
| numeric tokens /1000w | 22 - 61 | 29 - 61 |
| long words /1000w | 255 - 296 | 222 - 268 |
| nominalisation /1000w | 51 - 70 | 48 - 62 |

Nine rules follow from these:

1. **Noun-heavy.** "The pressure tensor calculation", not "we calculate how the pressure
   tensor behaves".
2. **The passive is correct here.** General advice to prefer the active voice moves
   scientific prose away from published writing.
3. **First person is normal.** These authors write "we".
4. **Long sentences, heavily punctuated**, with lengths that differ from neighbour to
   neighbour.
5. **Plain words.** Machine prose runs 332 long words per 1000 against 226 to 278.
6. **Verbs stay verbs.** Machine prose nominalises at 90 per 1000.
7. **Carry the numbers.** Prose *about* results reads machine-written; prose *containing*
   them does not. A failed draft held 1.2 numeric tokens per 1000 against a floor of 24.
8. **Never "In conclusion".** Present in 44% of machine papers, absent from real ones.
9. **Apply while drafting.** Variation retrofitted into finished uniform prose reads
   retrofitted.

## Writing for this group

`data/group_reference.json` holds style targets from 90 excerpts of 19 papers by one
research group and its coauthors, 2015 to 2023. `data/group_profile.json` holds their
vocabulary and 105 word combinations, measured on 103,300 words and 3,197 sentences.
Rates below are per 1000 words.

Where their targets differ from the wider corpus, theirs are the narrower and correct
ones for their manuscripts: less passive (0.24 to 0.39 against 0.37 to 0.60), more first
person (4.3 to 12.0 against 0.9 to 5.9), longer and more varied sentences.

### The pattern in their combinations

Seven habits account for nearly all of it.

**1. Cause is stated as a verb link, and it is the spine of the prose.** "due to" 1.22,
"because" 0.80, "leads to" 0.69, "thus" 0.69, "therefore" 0.29, "results in" 0.28,
"so that" 0.24, "because of" 0.13, "as a result" 0.08. Never "so" as a clause opener.

**2. The method verb is "can be" plus a participle**, at 2.02 per 1000, their single most
used frame. "can be written as" 0.25, "can be calculated from", "can be obtained by",
"can be expressed as"; also "is given by" 0.14, "is calculated from" 0.16, "is defined
as", "were performed" 0.24, "using the" 0.59, "is set to".

**3. Sentences open by pointing back.** "This" 1.35 and "These" 0.50 as subjects,
"However," 0.79 (82 sentences in 19 papers), "For a/the" 0.45, "When" 0.29, "Here,"
0.24, "Thus," 0.24, "Although" 0.18, "For example," 0.15, "Such" 0.15, "In addition,"
0.13, "Since" 0.13, "As shown in" 0.12, "In this work, we" 0.08.

**4. Stance is a pointer, not a softener.** "note that" 0.32, "we note that" 0.20, "it
should be noted that" 0.11: together 0.6 per 1000, all directing attention. Alongside:
"in general" 0.18, "in practice" 0.17, "it is possible to" 0.13, "in principle" 0.12,
"it is worth" 0.10, "indicates that" 0.18, "suggests that" 0.14, "appears to" 0.11,
"is likely" 0.09, "is expected to" 0.09, "as expected" 0.10. An earlier version of this
skill listed "in practice", "it is worth" and "it should be noted" as attenuators at
zero. This group uses all three, and the rule now excludes them.

**5. "We" takes model-building verbs.** "we have" 0.41, "we note" 0.22, "we also" 0.19,
"we consider" 0.16, "we present" 0.15, "we assume" 0.14, "we found" 0.09, "we show"
0.09, "we first" 0.07, "we use" 0.06.

**6. Comparison has a fixed set of frames.** "compared with" 0.44, "in terms of" 0.31,
"similar to" 0.20, "close to" 0.17, "consistent with" 0.14, "as a function of" 0.11,
"in good agreement with" 0.09, "with respect to" 0.09, "in comparison with" 0.07,
"differs from" 0.07, "agrees well with" 0.05.

**7. Space and system are named the same way every time.** "parallel to the surface"
0.27, "normal to the surface" 0.23, "the simulation box" 0.22, "the pore wall" 0.19,
"in the bulk phase" 0.18, "in equilibrium with the bulk" 0.15, "the definition of the"
0.13, "the first adsorbed layer" 0.11, "the thickness of the", "the surface area of",
"the center of mass", "term on the right".

Their content pairs, for when the subject is theirs: pressure tensor 2.29, tangential
pressure 1.41, surface area 0.96, pore width 0.48, pore size 0.41, adsorbed layer 0.40,
bulk phase 0.37, molecular simulation 0.36, Monte Carlo 0.32, simulation results 0.32,
adsorbate molecules 0.28, bulk pressure 0.24, density profile 0.24, intermolecular
forces 0.23, free energy 0.22, ensemble average 0.20, number density 0.18, uniquely
defined 0.15, Irving-Kirkwood 0.15, inhomogeneous systems 0.12, slit pores 0.12,
statistical mechanical 0.11.

Their subject vocabulary at 40 to 59 times the wider corpus rate: tensor, tangential,
irving, kirkwood, harasima, lj-eos, polycation, polyanion, microgel, nonsolvent.

## Registers

The reference is built from journal articles and several of its numbers describe that
genre rather than good writing. `check.py --register letter` and `--register docs` apply
only the universal measures.

| | paper | letter | documentation |
|---|---|---|---|
| register measures, numeric density, long words | yes | yes | yes |
| sentence length, passive, first person, nominalisation, commas | yes | no | no |

A cover letter is written in "we" and scores 22 first person per 1000 words against a
paper range of 0.9 to 5.9, and 0.29 passive against 0.37 to 0.60. Documentation is
imperative and scores lower still. Neither is a fault.

## Vocabulary

The research pipeline holds word frequencies from **6 million words** of
published papers: 800 PMC chemistry and materials articles, 615 adsorption and simulation
papers, and 14 from the target group. 41,017 word types; the copy in `data/vocab.json`
keeps the 31,550 seen at least three times.

`score.py` reports every word in a draft that falls below 1 occurrence per million in
that corpus. Technical terms will appear there and are expected. **General-purpose words
should not.**

The words this caught, none of them technical, none appearing anywhere in six million
words of papers:

> silently · honoured · defensible · admits · expenditure · repairs · inverts · subtlety

and these, used at 100 to 1000 times the published rate:

> badly · visibly · wrong · young · ordinary · settled · nothing · impractical

Every one is a judgement standing where a measurement belongs. Papers write "over-binds
oxygen by 40%"; the draft wrote "over-binds oxygen badly". They write "the models
disagree by 3 kJ/mol"; the draft wrote "which model to trust is not settled". A paper
does not call a method "young" or an objection "ordinary".

**Do not use a general-purpose word that the corpus does not use.** Where the impulse is
to characterise, give the number instead.

## Word sequences

The measure closest to "the word order reads machine-written". Take every word trigram
that carries at least two function words: these are phrasing, not content. Papers build
almost entirely from sequences other papers have used. On 129 excerpts from 19 held-out
papers, **52 to 63% of connective trigrams are attested** in 6M words of papers, never
under 35%. Drafts in the default register run **24 to 39%**, below every excerpt. With
the rest of this skill applied fully they reach 45 to 65%.

`check.py` reports the share, flags it under 46% (the p05), and lists the unattested sequences. The ones a draft
produces fall into five habits, each with what papers do instead:

| habit | draft | papers |
|---|---|---|
| clause-final adverb | "underestimates the binding energy there." | "at the metal site" |
| "so" as a clause connective | "so the problem is", "so an error" | "therefore", "as a result," new sentence |
| negated verb as the claim | "does not use", "cannot represent", "does not follow" | "is independent of", "is not captured by" |
| copular ranking | "is second at 18%", "is a second source of" | "AllScAIP gives 18%." |
| chains on "and" | "recovers X and diverges for Y", "cannot represent, and every" | two sentences |

The test is mechanical: a sequence that appears nowhere in 6M words of papers is one
the reader has not met. With the repository present, `scripts/suggest_sequences.py DRAFT`
lists each unattested triple with what papers write after the same two words: "the
uptake of" (126) where a draft had "the uptake from", "isosteric heat of" (49) for
"isosteric heat at", "the experimental data" (517) for "the experimental mean", "by a
factor of" (59) for "by a comparable factor". Rebuilding from those took a draft from
67% unattested triples to the paper range. Rebuild that sentence around a sequence papers use, and the
connective formulas below are the shortest route.

## Four habits, measured

A single sentence flagged by a reader was taken apart against 245 papers. Every fault
in it is a habit of drafts, not of papers:

| habit | papers | group | drafts |
|---|---|---|---|
| colon inside the sentence introducing evidence | 1.1% of sentences | 1.6% | 7.2% |
| passive with the agent attached by "by" | 1.4% | 1.3% | 4.3% |
| trailing "depending on" | 0.10 /1000w | 0.10 | 1.24 |
| "is therefore", "is thus" inside the verb | 0.03 /1000w | 0.01 | 1.85 |

The sentence:

> The binding energy at the metal site is therefore underestimated by every generic
> force field: the DREIDING and UFF parameterisations give isosteric heats at zero
> coverage of 28 and 31 kJ mol-1 [3], compared with calorimetric values of 42 to 47
> kJ mol-1 [4-6] depending on the sample.

carries all four in 43 words. Papers build the same content the other way round: the
quantity is the subject, the verb is "is" or "gives", the comparison rides on "while"
or "whereas", and the conclusion stands in its own sentence, usually first or last:

> Generic force fields underestimate the binding energy at this site. The isosteric
> heat at zero coverage is 28 kJ mol-1 with DREIDING and 31 kJ mol-1 with UFF [3],
> whereas calorimetry gives 42 to 47 kJ mol-1 [4-6].

`check.py` reports all four when they exceed three times the paper rate.

A second flagged paragraph had a different fault: its claims were carried by abstract
nouns papers do not use. Per 1000 words in 245 papers: "is a second source of error"
0.000, "is a property of" 0.000, "without bound" 0.001, a sentence opening "With the
multiplicity set from" 0.001, "more weakly" 0.005. What papers write in those places:
"was set to" 0.129, "we set" 0.065 (this group 0.155), "decreases from X to Y" 0.050,
"The second ..." as the subject 0.041, "diverges" 0.010. The rebuilt paragraph puts the
action in the verb and the number in the sentence:

> We therefore set the multiplicity from the metal oxidation states and repeated the
> calculations. For MACE-POLAR, the CO2 uptake in Ni-MOF-74 at 100 kPa decreases from
> 3.42 to 0.44 mol/kg and the agreement with experiment becomes worse. For AllScAIP,
> the Henry coefficient diverges.


## Citing

Measured on 245 papers with every citation kept as a marker: 1,136,629 words, 15,312
citations (`scripts/measure_citations.py`).

**Density.** 9.9 to 17.2 citations per 1000 words (p25 to p75), and 13 to 24% of
sentences carry one. A 400-word introduction or discussion paragraph without a citation
is outside the published range; four to seven is inside it.

**Placement.** 57% of markers sit directly after the noun phrase the claim refers to,
36% before a comma or semicolon that closes that phrase, and 7% at the end of the
sentence. The marker belongs to the claim, not to the sentence:

> The adsorption capacity [4], phase equilibrium [6], and transport [9] of confined
> phases are linked to the pore structure.
>
> Several semi-empirical equations have been proposed, such as the Sips equation [112]
> and the Toth equation [113].

**The finding is stated in the author's words and the reference is attached.** Naming
an author with a reporting verb ("X et al. showed that") is 1.6% of citations, 0.22 per
1000 words. Direct quotation of a source runs about 1.5 per paper and is not used for a
finding. In the 19 group papers the count is 4 named-author findings against 1,749
markers. Write:

> The binding energy at the open metal site includes a charge-transfer term of about
> 10 kJ mol-1 [4].

Not:

> Poloni and coworkers showed that charge transfer contributes about 10 kJ mol-1.

A name appears when the name is the topic: a method called after its authors (the
Irving-Kirkwood definition, the Harasima contour), or a result being compared with by
name ("Srivastava et al. [28] reported a systematic study of...", used in reviews).

**The introducers papers use**, per 1000 words: previous / previously 0.35, various
0.15, according to 0.13, recent / recently 0.24, for example 0.11, see 0.10, widely
0.06, often 0.06, in the literature 0.04, reported in 0.04, as shown in 0.04, has been
shown 0.03, in ref. 0.03, have been reported 0.03, reported by 0.03, has been used 0.02,
for instance 0.02, is known to 0.01. What follows a mid-sentence marker is most often
", and", "the", "in", "where", "which", "however".

`check.py` reports citation density against the band and lists every named-author
finding.

Sentence complexity is not a fault to correct. The group's sentences carry 1.6 to 2.2
clauses on average and 13 to 33% have three or more. A draft inside that range reads
as theirs; a detector's remark that a sentence "uses a complex structure with multiple
clauses" describes an ordinary sentence in this field. The same detector's explanations
of sentences it reads as human name "precise word choice" (test error, median isotherm
error, framework-adsorbate pairs), "factual clarity" (data stated without figurative
language), and "a technical detail connected to a broader quantity" (an error in the
binding energy expressed as a factor in the Boltzmann weight). Those are the
specificity rule above, stated by a reader.

## Rebuilding a sentence

The method that moved a draft furthest, and the one to use on any sentence a reader
flags. Take every word triple in the sentence. For each one that appears in no paper,
look up what papers write after the same two words. Rebuild from those.

```
python check.py DRAFT.md --suggest
```

The output for one draft, and what was done with it:

| draft had | papers write after the same two words | rebuilt as |
|---|---|---|
| the uptake from | the uptake **of** (126) | the uptake of CO2 from eSEN |
| isosteric heat at | isosteric heat **of** (49) | isosteric heat of adsorption at zero coverage |
| the experimental mean | the experimental **data** (517), **value** (62) | within 7% of the experimental value |
| by a comparable factor | by a **factor** (59) | by a factor of 7.5 |
| the Boltzmann weight | the Boltzmann **constant** (48) | the Boltzmann factor (the standard term; not itself in this corpus) |
| the training distribution | the training **set** (96) | close to those in the training set |
| at low coverage | at low **pressures** (61) | adsorption at low pressures |
| we therefore set the multiplicity from | was set **according to** | the spin multiplicity was set according to |
| was therefore retained | | Therefore, the closed-shell configuration was used for |

Three rules for using it. Technical names are unattested by nature and are left alone:
real papers themselves run **56 to 63% unattested** on all triples, and a draft that
reaches that range is done. A suggestion is a continuation, not a synonym; "the
Boltzmann constant" was not the meaning, and the replacement was the standard term
"Boltzmann factor", which the index does not hold either. And every rebuilt sentence gets read once more
for the science, because the index knows phrasing and nothing else: the same pass that
fixed the phrasing above also caught "the d orbitals of the metal ion" in a sentence
about Mg2+, which has no d electrons and had survived six drafts.

Measured on the same passage across eight drafts: unattested triples 64% to 53%,
connective sequences attested 39% to 60%, draft habits flagged 4 to 0, citations from
none to 16.8 per 1000 words.

## General rules from the suggestions

Each replacement above was checked across the whole corpus before becoming a rule.
These hold for any scientific draft, not for one sentence. `check.py` reports every
one it finds.

**1. A quantity noun takes "of".** Across 19 quantity nouns (uptake, heat, amount,
density, capacity, loading, enthalpy, energy, rate, value, distribution, coefficient,
fraction, ratio, number, error, spread, deviation) "of" follows 63 to 98% of the time.
"Deviation from" is the one second form (32%). Never "uptake at", "uptake from",
"spread among", "deviation for", "heat at".

**2. Compare with the experimental value, data or results, never with bare
"experiment".** "than the experimental" 38 against "than experiment" 0; "with the
experimental" 277 against 40; "to the experimental" 238 against 19.

**3. A factor has a number.** "a factor of" 81; "by a similar factor" 7, "by a
comparable factor" 0. Write "by a factor of 6", or give the two values.

**4. "This is because the ...".** After "is because" papers write "the" 127 times,
"of" 23, "a" never in the top five. The cause is a definite thing.

**5. State a condition as the variable that was set.** After "at low" and "at high"
papers write temperature, pH, pressure, concentration. Not "at low coverage" or "in the
zero-coverage limit" when the variable was pressure.

**6. Four abstract-noun claims exist; the rest do not.** Per million words: "is a
function of" 118, "is a result of" 54, "is a measure of" 12, "is a consequence of" 12.
Then: "is a source of" 3.6, "is a feature of" 1.0, "is a property of" 0.0, "is a
limitation of" 0.0, "is an indication of" 0.0. Outside the four, put the claim in a verb:
"the framework does not cause the divergence" rather than "is not a property of".

**7. "Therefore" opens the sentence.** "Therefore, the" 1516 against "is therefore"
104 and "was therefore" 33. Inside the verb it reads as hedged; at the front it reads
as a step.

**8. Method verbs come with their preposition.** "was set to / at / as", "was used to /
as / for", "was chosen as", "was taken as / from", "were used to / as / for". A method
sentence is a passive without an agent plus one of these; "was retained for" and "was
adopted" are not in the set.

**9. Use the canonical noun.** training set 161 (training data 44); test set 122 (test
data 8); Henry coefficient 57 (Henry constant 6); binding energy 858, interaction
energy 338, adsorption energy 301; isosteric heat 66.

**10. Cause is a verb link.** "due to", "leads to", "results in", "arises from",
"because"; never "so" as a clause opener and never the claim as a colon-introduced
list. Measured in the group section above.

## Phrasing

Vocabulary is not enough. Papers are built out of connective formulas, and a draft can
use only corpus-attested words while joining none of them the way papers do.

the n-gram index holds 141,296 bigrams and 194,362 trigrams from the same
6 million words. Rebuild both derived files with `python scripts/build_ngrams.py`.

**Measured: real papers at 800 words use 7 to 13% of the 60 most common connective
formulas. Five drafts used 2 to 3%.**

The formulas most heavily used in papers, with their rate per million:

| | | | |
|---|---|---|---|
| due to the (729) | as shown in (689) | the presence of (601) | the surface of (496) |
| shown in the (478) | the formation of (423) | on the surface (393) | are shown in (387) |
| in order to (380) | the effect of (358) | based on the (344) | as well as (323) |
| according to the (319) | than that of (301) | attributed to the (292) | was used to (279) |

This inverts the received advice. Every humanization guide says to strip stock phrases,
and academic prose is *made* of stock phrases. It uses a different set from the one those
guides target: papers write "attributed to the", not "delve into". Writing around all
formula is itself a departure from the register.

`score.py` reports coverage against the 7 to 13% band and names what a low draft is
missing. Use the formulas. Do not reach for a fresh construction where the corpus has a
settled one.

## Structure

Two references hold up under measurement and are built in.

**Gopen and Swan, The Science of Scientific Writing (American Scientist, 1990).** Seven
reader-expectation principles, quoted: (1) follow a grammatical subject as soon as
possible with its verb; (2) place in the stress position, the end of the sentence, the
new information you want the reader to emphasize; (3) place the person or thing whose
story the sentence tells at the beginning, in the topic position; (4) place old
information in the topic position for linkage backward; (5) articulate the action of
every clause in its verb; (6) provide context before asking the reader to consider
anything new; (7) make the emphasis of the substance coincide with the emphasis the
structure raises. They add that none is a rule and any can be broken to effect.

Two are checkable without a parser, and `check.py` reports both. Sentences with more
than 12 words before the first verb run 15 to 29% in 19 published papers; a draft
written to sound academic reached 60%. Sentences opening on This, These, Such, Here or
It, which put the old information first, run 5 to 19% in papers; drafts in the default
register open none that way, and rebuilding a paragraph so each sentence starts from the
previous one is the single change that most improves how it reads.

**Mensh and Kording, Ten simple rules for structuring papers (PLOS Comput Biol, 2017).**
One contribution per paper, stated in the title. Context, content, conclusion at every
scale: the paper, each paragraph, and each sentence. One point per paragraph, covered in
one place only. Results as a sequence of declarative statements, each carried by a
figure whose title states the conclusion. The abstract tells the whole story: context,
gap, method, result, interpretation. The introduction's last paragraph states the result
more specifically than the abstract does. Time goes where readers go: title, abstract,
figures, outline.

## Academic phrasebank, measured

The Manchester Academic Phrasebank lists frames by section. The field-neutral ones were
measured per 1000 words on 5.9M words of papers, 103K words of this group, and 141K
words of machine output.

Frames papers use, with rates: "however, the" 0.50; "were carried out / performed" 0.56;
"in addition," 0.47; "moreover," 0.46; "in order to" 0.38; "furthermore," 0.34; "may be
attributed to" 0.29; "was used to" 0.28; "on the other hand" 0.16; "hence," 0.16; "in
contrast to" 0.13; "as a result" 0.13; "is consistent with" 0.12; "in this work, we"
0.11; "note that" 0.10; "the present work" 0.09; "it can be seen that" 0.09; "prior to"
0.09; "in particular," 0.08; "interestingly," 0.08; "in good agreement with" 0.06; "it
should be noted" 0.05; "importantly," 0.05.

Frames machine output uses at twice the paper rate or more, which the phrasebank lists
without warning: "in conclusion," 0.60 against 0.03; "in summary," 0.45 against 0.07;
"these results suggest that" 0.46 against 0.10; "plays a key role" 0.23 against 0.09;
"provides new insights into" 0.20 against 0.00; "specifically," 0.22 against 0.04;
"further work is needed" 0.12 against 0.01; "as shown in figure" 0.14 against 0.02.

Frames the phrasebank offers that papers in this field do not use (under 0.02): "sheds
new light on", "fills a gap", "makes an important contribution", "a possible explanation
for", "cannot be ruled out", "should be interpreted with caution", "beyond the scope of",
"little is known about", "remains poorly understood", "is a major challenge".

## Academic Word List, measured

Coxhead's Academic Word List (570 word families) was matched against the same three
corpora (`data/awl_measured.json`). 423 families appear at 10 per million or more in
papers, 42 do not appear at all. The list is a vocabulary of academic English in
general; what matters is which of it this field uses.

Families papers use and machine output avoids (papers over 100 per million, machine
under half): obtain, simulate, data, phase, constant, layer, previous, predict, region,
occur, whereas, estimate, hence, illustrate, assign, bulk, define, compute, assume,
section, deviate, normal, denote, fluctuate, error, formula, random, prior, conclude,
confine, statistic, sufficient.

Families machine output uses at three times the paper rate or more: enhance (4332
against 563 per million), integrate, monitor, insight, crucial, ensure, highlight,
challenge, assess, facilitate, comprehensive, impact, focus, environment, significant,
demonstrate, evaluate, conduct, detect, select, exhibit, reveal, potential, unique.
These are correct English and correct academic English; they are also the words a
draft reaches for when it has nothing specific to say. Use them where the meaning is
that word and nowhere else.

## The published humanization levers, tested

The installable "humanizer" skills (blader/humanizer, 5.1K installs; jpeggdev/humanize-
writing; momo2young/humanize-academic-writing, 1.4K) all descend from Wikipedia's "Signs
of AI writing" list. Every rule they share was measured here, per 1000 words, on 5.9M
words of papers against 141K words of instruction-tuned output.

| rule | papers | machine | verdict |
|---|---|---|---|
| moreover / furthermore / additionally | 0.95 | 2.35 | real tell, but papers still use them |
| crucial / pivotal / vital / testament | 0.13 | 1.02 | real tell |
| robust / seamless / innovative / transformative | 0.04 | 0.48 | real tell |
| copula avoidance: serves as, represents a | 0.12 | 0.65 | real tell |
| -ing tails: highlighting, underscoring | 0.01 | 0.23 | real tell |
| not only X but Y | 0.10 | 0.28 | real tell |
| it is worth noting / could potentially | 0.04 | 0.11 | real tell |
| delve / landscape / tapestry / leverage | 0.03 | 0.12 | real, and rare either way |
| "in order to" as filler | 0.38 | 0.01 | **backwards**: papers use it |
| "from X to Y" as a false range | 0.37 | 0.15 | **backwards**: papers state ranges |
| semicolons | 1.38 | 0.08 | **backwards**: papers use them |
| em dashes | 0.10 | 0.05 | backwards, and rare either way |
| rule of three | 2.31 | 1.74 | not a tell in papers |
| force short sentences, spread lengths, strip "In conclusion" | | | confirmed earlier |

Two things follow. The word-level tells are real for raw model output and are worth
one pass. But a draft written to this skill already sits at 0.00 on every one of them
(measured across 4,855 words), and still reads machine-written on word order. Those
lists fix a problem such a draft does not have, and none of them measures sequences,
which is where the difference actually lives.

Each of the three was then applied in full to the same 400-word passage, alongside a
draft written to this skill. GPTZero returned 1.000 with every
sentence flagged on all four. On this skill's own measures the humanizer drafts scored
2 to 5 of 13 bands and 32 to 38% on sequences; the draft written to this skill scored 6
of 13 and 51%.

## Does following this work

Measured across four drafts against the excerpt reference: features within one
standard deviation rose from 73% to 82%, median absolute z fell from 0.38 to 0.12, and
outliers beyond two standard deviations fell from 11% to 3%. The register fix accounted
for most of the final step.

That is an improvement in matching published scientific register. It is the only claim
this skill makes.

## How to use it

1. Write the draft yourself, in whatever English comes out. The measures below work on
   a text that already carries your argument; they do not supply one.
2. `python check.py DRAFT.md`, with `--reference group` for a manuscript from this
   group and `--register letter` or `--register docs` outside a paper.
3. Fix in this order: stock hedges and attenuators to zero; unattested sequences
   rebuilt around a formula papers use; numbers where the draft characterises;
   contrastive constructions down to one per thousand words. Stop at 6 to 8 of 13 bands
   and 46% or more on sequences. A real paper scores there, and higher means the draft
   was written to the metric.
4. Keep every claim and every number. The skill changes register, not content.

## Detectors

Do not tune text against a detector's score, and do not present a detector verdict as a
measure of writing quality. Every target in this skill is a property of published
papers, measured on published papers. That is the standard, and the only one.

## Improving this skill

This is a first version built from one corpus and one research group's papers, and it
improves through use. When a user of this skill reports a phrase, a rule, or a target
that seems wrong or missing:

1. Check it against the bundled data first: `data/vocab.json` for a word,
   `data/sequences.json` and `data/trigrams.json` for a phrase, `data/reference.json`
   for a target. Report the rate you find, with the count, so the claim is settled by
   the corpus and not by opinion.
2. If the data confirm the report, make the change locally in SKILL.md, check.py or
   data/ so the user has it now.
3. Then ask once whether they would like to contribute the change back, and if yes,
   open an issue or a pull request on the repository with the measurement attached.
   Record the answer in `data/local_preferences.json` and do not ask again in that
   project.

Anyone reading this is welcome to contribute directly: a corpus from another field, a
group profile of their own, a rule with its numbers, or a correction.
