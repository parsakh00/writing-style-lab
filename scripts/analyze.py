"""Measure the separation between the human and AI corpora, and write the policy.

Produces four things:

  results/separation.csv       every feature, both distributions, effect size, corrected p
  results/human_reference.json the human distribution, which score.py compares drafts to
  results/summary.md           the readable report
  STYLE.md                     the policy, with target ranges taken from the human corpus

Three deliberate choices in the statistics.

Effect size leads, not the p-value. With a few hundred documents per side almost
everything reaches significance, so a ranking by p-value would be a ranking by nothing.
Cohen's d says how far apart the distributions actually are, which is the question.

Multiple comparisons are corrected. Around 250 features are tested at once; at the usual
threshold a dozen would clear it by chance alone. Benjamini-Hochberg keeps the false
discovery rate controlled without the severity of Bonferroni.

The discriminative model is cross-validated. An in-sample fit on 250 features and a few
hundred documents would report a near-perfect separation whether or not one exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.features import feature_group  # noqa: E402

META_COLS = {"doc_id", "label", "source", "year", "mirrors"}

# Readable glosses for the features most likely to reach the policy. Anything absent
# falls back to its raw name, which is ugly but never wrong.
GLOSS = {
    "sent_len_cv": "sentence length variation (sd / mean)",
    "sent_len_sd": "sentence length spread, in words",
    "sent_len_mean": "mean sentence length, in words",
    "sent_len_neighbour_delta": "average change in length between consecutive sentences",
    "short_sent_frac": "share of sentences of 8 words or fewer",
    "long_sent_frac": "share of sentences of 35 words or more",
    "para_sents_mean": "sentences per paragraph",
    "mtld": "lexical diversity (MTLD)",
    "ttr_500": "distinct words in the first 500",
    "hapax_rate": "share of vocabulary used exactly once",
    "nominalisation_rate": "nouns made from verbs (-tion, -ment, -ity) per 1000 words",
    "long_word_rate": "words of 8+ letters per 1000",
    "transition_rate": "connectives (however, moreover, furthermore) per 1000 words",
    "sent_initial_transition_frac": "share of sentences opening with a connective",
    "sent_initial_this_frac": "share of sentences opening with 'This'",
    "hedge_rate": "hedges (may, suggest, appear) per 1000 words",
    "booster_rate": "intensifiers (crucial, significantly, remarkable) per 1000 words",
    "first_person_rate": "we / our / us per 1000 words",
    "tricolon_rate": "three-item lists per 1000 words",
    "opener_entropy": "variety of sentence opening words",
    "opener_top1_frac": "share of sentences sharing the single most common opener",
    "tell_total_rate": "stock phrases per 1000 words",
    "punct_semicolon_rate": "semicolons per 1000 words",
    "punct_colon_rate": "colons per 1000 words",
    "punct_comma_rate": "commas per 1000 words",
    "punct_paren_rate": "parentheses per 1000 words",
    "punct_long_dash_rate": "em and en dashes per 1000 words",
    "passive_per_clause": "passive constructions per clause",
    "dep_depth_mean": "mean syntactic depth per sentence",
    "subordination_rate": "subordinate clauses per 1000 tokens",
    "noun_verb_ratio": "nouns per verb",
    "lm_ppl": "GPT-2 perplexity",
    "lm_logprob_sd": "variation in how surprising each word is",
    "lm_logprob_neighbour_delta": "word-to-word swing in surprise",
    "lm_surprise_frac": "share of words the model finds genuinely unlikely",
    "lm_rank_top10_frac": "share of words among the model's top 10 guesses",
    "w_sent_len_cv_worst": "sentence variation in the most uniform passage",
    "frac_uniform_windows": "share of passages that are uniform in rhythm",
    "longest_uniform_run": "longest consecutive stretch of uniform passages",
}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-sd standardised mean difference. Positive means higher in AI."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downward.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def restrict_to_mirrored(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the human documents that the AI side was generated to mirror.

    The generated corpus is built one-to-one against specific human papers, matched on
    title and target length. Comparing it against the *whole* human corpus throws that
    away and reintroduces a length difference whenever the generation run is incomplete.

    That is not hypothetical here. Generation batches are sorted by target length so a
    batch is not held up by its longest member, which means a run cut short by walltime
    yields only the shortest papers. Pairing against the mirrored subset makes the length
    match exact by construction, however far the run got.
    """
    ai = df[df.label == "ai"]
    mirrored = set(ai["mirrors"].dropna()) if "mirrors" in df.columns else set()
    if not mirrored:
        return df
    keep_human = df.label.ne("human") | df.doc_id.isin(mirrored)
    out = df[keep_human]
    n_drop = len(df) - len(out)
    if n_drop:
        print(f"  paired mode: dropped {n_drop} human documents with no generated "
              f"counterpart, leaving {int((out.label == 'human').sum())} matched pairs")
    return out


def length_balance(df: pd.DataFrame) -> dict:
    """Check that the two labels are comparable in length before anything else.

    This project has come close to a confident wrong answer three separate times through
    length: HC3 human answers ran 436 words against 322 for the model, RAID's floor kept
    38% of machine abstracts and 0.1% of human ones, and the first generation run
    produced 800-word papers to mirror 3822-word ones. Each would have separated the
    labels beautifully, and each would have been measuring word count.

    A check that depends on somebody remembering to look is not a check, so it runs on
    every analysis and reports loudly. It does not block: length differences are
    sometimes real and intended, and the caller is better placed to judge. But it will
    not be silent about it.
    """
    if "n_words" not in df.columns:
        return {}
    h = df.loc[df.label == "human", "n_words"].dropna().to_numpy(dtype=float)
    a = df.loc[df.label == "ai", "n_words"].dropna().to_numpy(dtype=float)
    if len(h) < 3 or len(a) < 3:
        return {}
    d = cohens_d(h, a)
    return {
        "human_median": float(np.median(h)),
        "ai_median": float(np.median(a)),
        "cohens_d": d,
        "severe": abs(d) > 0.8,
        "notable": abs(d) > 0.3,
    }


def separation_table(df: pd.DataFrame) -> pd.DataFrame:
    human = df[df.label == "human"]
    ai = df[df.label == "ai"]
    feats = [c for c in df.columns if c not in META_COLS]

    rows = []
    for f in feats:
        a = human[f].dropna().to_numpy(dtype=float)
        b = ai[f].dropna().to_numpy(dtype=float)
        if len(a) < 5 or len(b) < 5:
            continue
        # Mann-Whitney rather than a t-test: most of these features are rates and
        # fractions with skewed, bounded distributions, and normality is not on offer.
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append({
            "feature": f,
            "group": feature_group(f),
            "human_mean": a.mean(), "human_sd": a.std(ddof=1),
            "human_p25": np.percentile(a, 25), "human_p50": np.percentile(a, 50),
            "human_p75": np.percentile(a, 75),
            "ai_mean": b.mean(), "ai_sd": b.std(ddof=1),
            "ai_p50": np.percentile(b, 50),
            "cohens_d": cohens_d(a, b),
            "p_raw": p,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adj"] = benjamini_hochberg(out["p_raw"].to_numpy())
    out["abs_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_d", ascending=False).reset_index(drop=True)


def discriminative_model(df: pd.DataFrame, seed: int = 0) -> dict:
    """Cross-validated logistic fit. Reports which features carry the separation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Detector scores are excluded as predictors. They are outputs of the same kind of
    # model the features are meant to characterise, so feeding them in and reporting the
    # resulting AUC as evidence that the features separate the corpora is circular: the
    # model would largely be asking a detector and taking credit. They stay in the
    # separation table, where they are reported as an independent check, and in
    # validate_with_detectors, which is what they are for.
    feats = [c for c in df.columns
             if c not in META_COLS and not c.startswith("det_")]
    X = df[feats].replace([np.inf, -np.inf], np.nan)

    # Drop features that are undefined for most documents. Median-filling a column that
    # is NaN for one whole label would invent a value that separates the labels, which
    # is the opposite of what the NaN was recording.
    keep = [c for c in feats if X[c].notna().mean() >= 0.5]
    dropped = len(feats) - len(keep)
    feats = keep
    X = X[feats]
    X = X.fillna(X.median()).to_numpy(dtype=float)
    y = (df.label == "ai").astype(int).to_numpy()

    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"error": "need both labels and at least 20 documents"}

    # L2 is the default penalty. Naming it explicitly triggers a deprecation warning on
    # scikit-learn 1.8+, and the replacement spelling does not exist on older versions,
    # so the portable choice is to leave it at the default.
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=min(5, int(min(np.bincount(y)))), shuffle=True,
                         random_state=seed)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    pipe.fit(X, y)
    coef = pipe.named_steps["logisticregression"].coef_[0]
    weights = sorted(zip(feats, coef), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "auc_mean": float(auc.mean()),
        "auc_sd": float(auc.std()),
        "n_folds": int(cv.get_n_splits()),
        "n_features": len(feats),
        "n_dropped_sparse": dropped,
        "top_weights": [(f, float(w)) for f, w in weights[:25]],
    }


def permutation_null(df: pd.DataFrame, observed_auc: float, n_perm: int) -> dict:
    """Empirical null for the AUC, by refitting on shuffled labels.

    Needed because the sampling distribution of a cross-validated AUC depends on sample
    size and on how correlated the features are, and neither is known in advance. On a
    60 document corpus with this feature set the null AUC has a standard deviation of
    roughly 0.1, so an observed 0.68 is unremarkable; on a 1200 document corpus the same
    value would be decisive. Comparing against a null computed from the actual data
    removes the guesswork instead of hard coding a threshold that is wrong at most
    sample sizes.

    Returns the null distribution's centre and spread, and the fraction of permutations
    that matched or beat the observed value, which is an exact p-value.
    """
    if n_perm <= 0 or not np.isfinite(observed_auc):
        return {}

    rng = np.random.default_rng(0)
    nulls: list[float] = []
    for i in range(n_perm):
        shuffled = df.copy()
        shuffled["label"] = rng.permutation(shuffled["label"].to_numpy())
        m = discriminative_model(shuffled, seed=i)
        if "auc_mean" in m:
            nulls.append(m["auc_mean"])
        print(f"  permutation {i + 1}/{n_perm}", end="\r")
    print(" " * 40, end="\r")

    if not nulls:
        return {}
    arr = np.asarray(nulls)
    n_ge = int((arr >= observed_auc).sum())
    return {
        "n_perm": len(nulls),
        "null_mean": float(arr.mean()),
        "null_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "null_p95": float(np.percentile(arr, 95)),
        # Add-one correction: with n permutations the smallest reportable p is
        # 1/(n+1), and claiming p = 0 from a finite permutation set is wrong.
        "p_value": (n_ge + 1) / (len(arr) + 1),
    }


def validate_with_detectors(df: pd.DataFrame) -> list[dict]:
    """Check the open detectors against the known corpus labels.

    Two things are established here. First, whether each detector separates these
    corpora at all, reported as AUC. Second, its orientation, which is inferred from the
    data rather than assumed: an AUC below 0.5 means the score runs the other way, and
    saying so is far better than silently drawing inverted conclusions from it.
    """
    from sklearn.metrics import roc_auc_score

    out = []
    y = (df.label == "ai").astype(int).to_numpy()
    for col in [c for c in df.columns if c.startswith("det_") and not c.endswith("_error")]:
        s = df[col].to_numpy(dtype=float)
        ok = np.isfinite(s)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        auc = float(roc_auc_score(y[ok], s[ok]))
        out.append({
            "detector": col.replace("det_", ""),
            "auc_raw": auc,
            "auc_oriented": max(auc, 1 - auc),
            "higher_means_ai": auc >= 0.5,
            "n_scored": int(ok.sum()),
        })
    return out


def replication_check(full: pd.DataFrame, primary: pd.DataFrame,
                      sep: pd.DataFrame, top_k: int = 15) -> list[dict]:
    """Re-test the primary comparison's top features on each secondary corpus.

    This is the check that matters most for believing any of it. The primary comparison
    uses one generator, one prompt and one journal register, so a feature that separates
    there could be an artefact of any of the three. A feature that also separates in
    HC3 or RAID, built by other people with other generators, is far more likely to be
    a property of machine prose.

    Sign agreement is the test, not magnitude. Effect sizes will not transfer across
    registers and there is no reason to expect them to; the direction should.
    """
    held_out = sorted(set(full.source.dropna()) - set(primary.source.dropna()))
    top = sep.head(top_k)
    out = []

    for src in held_out:
        sub = full[full.source == src]
        if sub.label.nunique() < 2:
            continue
        h = sub[sub.label == "human"]
        a = sub[sub.label == "ai"]
        if len(h) < 10 or len(a) < 10:
            continue

        agree = 0
        tested = 0
        details = []
        for _, r in top.iterrows():
            f = r.feature
            if f not in sub.columns:
                continue
            av = h[f].dropna().to_numpy(dtype=float)
            bv = a[f].dropna().to_numpy(dtype=float)
            if len(av) < 5 or len(bv) < 5:
                continue
            d2 = cohens_d(av, bv)
            tested += 1
            same = (d2 > 0) == (r.cohens_d > 0)
            agree += int(same)
            details.append({"feature": f, "d_primary": float(r.cohens_d),
                            "d_secondary": float(d2), "same_sign": bool(same)})

        if tested:
            out.append({
                "source": src, "n_human": len(h), "n_ai": len(a),
                "tested": tested, "agree": agree,
                "agreement": agree / tested, "details": details,
            })
    return out


def write_style_md(sep: pd.DataFrame, path: Path, min_d: float, top_n: int) -> int:
    strong = sep[(sep.abs_d >= min_d) & (sep.p_adj < 0.05)]
    # Three groups are kept out of the policy, all for the same reason: a rule has to be
    # something a writer can act on.
    #
    #   function  "use 'the' 4% less often" is not a followable instruction
    #   detector  a detector score is an output, not a property of the prose
    #   lm        perplexity is a measurement, not an action. It usually separates the
    #             corpora most strongly, and it will dominate this list if allowed, but
    #             "raise your perplexity" tells a writer nothing. The features that
    #             *cause* it, sentence variation and word choice, are already here on
    #             their own terms.
    #
    # They all stay in separation.csv, which is where the measurement lives.
    strong = strong[~strong.group.isin(
        {"function", "detector", "lm", "formatting"})].head(top_n)

    lines = [
        "# STYLE.md",
        "",
        "Generated by `scripts/analyze.py`. Do not edit by hand; rerun the analysis.",
        "",
        "Every rule below came from a measured difference between human written and",
        "model written papers on matched topics. Each states the feature it came from,",
        "the target range taken from the middle half of the human corpus, and what the",
        "model corpus did instead. A rule you disagree with can be checked against",
        "`results/separation.csv` and dropped.",
        "",
        "Targets are ranges, not points. Writing to the exact human mean on every axis",
        "would itself be a kind of uniformity that no real document has.",
        "",
        "---",
        "",
    ]

    if strong.empty:
        lines += [
            "No feature reached the effect size threshold.",
            "",
            f"Nothing cleared |d| >= {min_d} with an adjusted p below 0.05. That is a",
            "real result, not a failure: on this corpus, at this sample size, the two",
            "kinds of prose are not far apart on the measures tested. Enlarging the",
            "corpora or adding a second generator would sharpen the test.",
            "",
        ]
    else:
        for _, r in strong.iterrows():
            name = r.feature
            gloss = GLOSS.get(name, name)
            direction = "higher" if r.cohens_d > 0 else "lower"
            lines += [
                f"### {gloss}",
                "",
                f"- Target: **{r.human_p25:.3g} to {r.human_p75:.3g}** "
                f"(human median {r.human_p50:.3g})",
                f"- Model corpus sits {direction}, at {r.ai_p50:.3g}",
                f"- `{name}` · Cohen's d = {r.cohens_d:+.2f} · adjusted p = {r.p_adj:.1e}",
                "",
            ]

    lines += [
        "---",
        "",
        "## How to use this",
        "",
        "Run `python scripts/score.py <draft>` to see where a draft sits on each axis.",
        "The report ranks features by how far the draft is from the human distribution,",
        "so the top few lines are the ones worth acting on.",
        "",
        "Two cautions on reading the numbers.",
        "",
        "These targets describe published chemistry and materials papers from before",
        "2020. They are not targets for a grant abstract, a cover letter or a talk.",
        "",
        "A feature that separates the corpora is not automatically a feature worth",
        "changing. Some of these differences are the model writing badly, and fixing",
        "them makes the prose better. Others are just differences. The effect size tells",
        "you the gap is real; it does not tell you the gap is a flaw.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return len(strong)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--style-out", default="STYLE.md")
    ap.add_argument("--min-d", type=float, default=0.5,
                    help="effect size floor for a feature to become a rule")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--sources", default="pmc,generated",
                    help="comma-separated sources for the primary comparison. The "
                         "default is the matched pair; secondary corpora have a "
                         "different register and pooling them would confound genre "
                         "with authorship. Pass 'all' to override.")
    ap.add_argument("--paired", action="store_true",
                    help="compare only against the human documents the AI corpus was "
                         "generated to mirror. Makes the length match exact even if "
                         "generation was incomplete.")
    ap.add_argument("--replicate", action="store_true",
                    help="also test the top features on each secondary source")
    ap.add_argument("--permutations", type=int, default=0,
                    help="build an empirical null for the AUC by refitting on shuffled "
                         "labels this many times. 20 is usually enough to tell a real "
                         "separation from small-sample noise. Costs one model fit each.")
    args = ap.parse_args()

    fpath = Path(args.features)
    if not fpath.exists():
        print(f"{fpath} not found. Run scripts/build_features.py first.", file=sys.stderr)
        return 1

    full = pd.read_csv(fpath)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.sources.strip().lower() == "all":
        df = full
        print("comparing across ALL sources; effect sizes below mix register with "
              "authorship")
    else:
        keep = {s.strip() for s in args.sources.split(",") if s.strip()}
        df = full[full.source.isin(keep)].copy()
        skipped = sorted(set(full.source.dropna()) - keep)
        print(f"primary comparison: {', '.join(sorted(keep))}")
        if skipped:
            print(f"  held out of the primary comparison: {', '.join(skipped)}")
        if df.empty:
            print(f"no documents match --sources {args.sources}", file=sys.stderr)
            return 1

    n_h = int((df.label == "human").sum())
    n_a = int((df.label == "ai").sum())
    print(f"loaded {len(df)} documents: {n_h} human, {n_a} ai")
    if n_h < 5 or n_a < 5:
        print("need at least 5 documents on each side", file=sys.stderr)
        return 1

    if args.paired:
        df = restrict_to_mirrored(df)
        n_h = int((df.label == "human").sum())
        n_a = int((df.label == "ai").sum())

    bal = length_balance(df)
    if bal:
        print()
        print(f"length balance: human median {bal['human_median']:,.0f} words, "
              f"ai median {bal['ai_median']:,.0f} words, d = {bal['cohens_d']:+.2f}")
        if bal["severe"]:
            print("  *** THE TWO LABELS DIFFER SUBSTANTIALLY IN LENGTH ***")
            print("  Many style features correlate with length, so the separation below")
            print("  may be word count wearing style's clothes. Length-match the corpora")
            print("  before believing any of it (see length_match in fetch_datasets.py).")
        elif bal["notable"]:
            print("  note: a modest length difference. Worth keeping in mind when a")
            print("  length-sensitive feature appears near the top.")
        else:
            print("  lengths are comparable; the comparison is not confounded by size.")

    sep = separation_table(df)
    sep.to_csv(outdir / "separation.csv", index=False)

    print(f"\ntop separating features ({len(sep)} tested):")
    print(f"  {'feature':38s} {'human':>9s} {'ai':>9s} {'d':>7s} {'p_adj':>9s}")
    for _, r in sep.head(20).iterrows():
        print(f"  {r.feature[:38]:38s} {r.human_p50:9.3g} {r.ai_p50:9.3g} "
              f"{r.cohens_d:+7.2f} {r.p_adj:9.1e}")

    by_group = (sep.groupby("group")["abs_d"]
                .agg(["mean", "max", "count"])
                .sort_values("max", ascending=False))
    print("\nseparation by feature group:")
    print(f"  {'group':14s} {'mean |d|':>9s} {'max |d|':>9s} {'n':>5s}")
    for g, r in by_group.iterrows():
        print(f"  {g:14s} {r['mean']:9.2f} {r['max']:9.2f} {int(r['count']):5d}")

    model = discriminative_model(df)
    null = {}
    if "error" in model:
        print(f"\ndiscriminative model skipped: {model['error']}")
    else:
        print(f"\ncross-validated separability: AUC {model['auc_mean']:.3f} "
              f"+/- {model['auc_sd']:.3f} over {model['n_folds']} folds")

        if args.permutations:
            print(f"  building empirical null from {args.permutations} "
                  f"label permutations...")
            null = permutation_null(df, model["auc_mean"], args.permutations)
        if null:
            print(f"  null AUC {null['null_mean']:.3f} +/- {null['null_sd']:.3f} "
                  f"(95th percentile {null['null_p95']:.3f})")
            print(f"  permutation p = {null['p_value']:.3f}")
            if null["p_value"] > 0.05:
                print("  the observed AUC is within what shuffled labels produce on "
                      "this corpus.")
                print("  Treat the feature ranking below as provisional, and get more "
                      "documents.")
        else:
            print("  no empirical null computed; pass --permutations 20 to get one.")
            print("  Without it an AUC is hard to read: at a few dozen documents the "
                  "null itself sits near 0.6.")

        print("  strongest weights (positive = pushes toward the AI label):")
        for f, w in model["top_weights"][:12]:
            print(f"    {f[:40]:40s} {w:+.3f}")

    detectors = validate_with_detectors(df)
    if detectors:
        print("\nopen detector validation:")
        for d in detectors:
            direction = "higher = AI" if d["higher_means_ai"] else "lower = AI"
            print(f"  {d['detector']:16s} AUC {d['auc_oriented']:.3f}  ({direction}, "
                  f"n={d['n_scored']})")
    else:
        print("\nno detector columns present (rerun build_features.py with --detectors)")

    replication = []
    if args.replicate:
        replication = replication_check(full, df, sep)
        if replication:
            print("\nreplication on held-out corpora (sign agreement on the top 15):")
            for rep in replication:
                print(f"  {rep['source']:12s} {rep['agree']}/{rep['tested']} features "
                      f"agree in direction  ({rep['agreement']:.0%}, "
                      f"n={rep['n_human']}h/{rep['n_ai']}a)")
                disagreed = [d["feature"] for d in rep["details"] if not d["same_sign"]]
                if disagreed:
                    print(f"    disagreed: {', '.join(disagreed[:6])}")
        else:
            print("\nno secondary corpus had both labels; nothing to replicate on")
            print("  fetch one with: python scripts/fetch_datasets.py --hc3 500")

    # The reference distribution score.py compares drafts against.
    human = df[df.label == "human"]
    reference = {
        "n_documents": n_h,
        # Recorded so score.py can tell whether a draft is comparable in length. Over a
        # third of features drift more than 20% between an 800 word excerpt and a full
        # paper, so a reference is only valid for drafts of roughly its own size.
        "median_words": float(human["n_words"].median()) if "n_words" in human else 0.0,
        "sources": sorted(human.source.dropna().unique().tolist()),
        "features": {
            f: {
                "mean": float(human[f].mean()),
                "sd": float(human[f].std(ddof=1)),
                "p05": float(human[f].quantile(0.05)),
                "p25": float(human[f].quantile(0.25)),
                "p50": float(human[f].quantile(0.50)),
                "p75": float(human[f].quantile(0.75)),
                "p95": float(human[f].quantile(0.95)),
            }
            for f in df.columns if f not in META_COLS and human[f].notna().sum() >= 5
        },
        "separating_features": sep.head(60)[["feature", "cohens_d", "p_adj"]]
        .to_dict("records"),
    }
    (outdir / "human_reference.json").write_text(
        json.dumps(reference, indent=2), encoding="utf-8")

    n_rules = write_style_md(sep, Path(args.style_out), args.min_d, args.top_n)

    summary = [
        "# Analysis summary",
        "",
        f"- {n_h} human documents, {n_a} model documents",
        (f"- length balance: human median {bal['human_median']:,.0f} w, ai median "
         f"{bal['ai_median']:,.0f} w, d = {bal['cohens_d']:+.2f}"
         + ("  **LENGTH CONFOUND**" if bal.get("severe") else ""))
        if bal else "- length balance: not computed",
        f"- {len(sep)} features tested",
        f"- {int(((sep.abs_d >= args.min_d) & (sep.p_adj < 0.05)).sum())} features with "
        f"|d| >= {args.min_d} and adjusted p < 0.05",
    ]
    if "error" not in model:
        summary.append(f"- cross-validated AUC {model['auc_mean']:.3f} "
                       f"+/- {model['auc_sd']:.3f}")
    if null:
        summary.append(f"- permutation null AUC {null['null_mean']:.3f} "
                       f"+/- {null['null_sd']:.3f} over {null['n_perm']} shuffles; "
                       f"p = {null['p_value']:.3f}")
    for d in detectors:
        summary.append(f"- detector {d['detector']}: AUC {d['auc_oriented']:.3f}")
    for rep in replication:
        summary.append(f"- replication on {rep['source']}: {rep['agree']}/"
                       f"{rep['tested']} top features agree in direction")
    summary += ["", "## Top 30 features by effect size", "",
                "| feature | group | human median | ai median | d | adj p |",
                "|---|---|---|---|---|---|"]
    for _, r in sep.head(30).iterrows():
        summary.append(f"| `{r.feature}` | {r.group} | {r.human_p50:.4g} | "
                       f"{r.ai_p50:.4g} | {r.cohens_d:+.2f} | {r.p_adj:.1e} |")
    (outdir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"\nwrote {outdir / 'separation.csv'}")
    print(f"wrote {outdir / 'human_reference.json'}")
    print(f"wrote {outdir / 'summary.md'}")
    print(f"wrote {args.style_out} ({n_rules} rules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
