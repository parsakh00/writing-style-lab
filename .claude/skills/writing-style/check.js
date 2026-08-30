// A line-for-line port of check.py, so the page can run the checker without a Python
// runtime. Same data files, same output. CI runs both on the same text and compares.
//
//   node check.js draft.md [--register paper|letter|docs] [--reference corpus|group] [--suggest]
//   import { report } from "./check.js"   (browser: pass the loaded data files)

const RE_WORD = /[A-Za-z][A-Za-z'-]*/g;
const RE_SENT = /(?<=[.!?])["')\]]*\s+(?=[A-Z0-9("'])/;
const RE_NUMERIC = /\S*\d\S*/g;
const RE_CONCESSIVE = /\b(?:though|although|however|whereas|albeit|nonetheless|nevertheless|admittedly|granted|conversely|whilst)\b/gi;
const RE_CONTRASTIVE = /(?:rather than|while (?:it|this|that|the)|on the other hand|whereas)/gi;
const RE_COUNTERWEIGHT = /(?:at the expense of|is not to say|trade-?off|in exchange for|that said|to be fair|the flip side|cuts both ways)/gi;
const RE_ATTENUATOR = /\b(?:for (?:now|the moment)|to some extent|in practice|arguably|it (?:remains|is worth|should be noted)|we would not)\b/gi;
const RE_QUOTED = /"[^"]{1,120}"|'[^']{1,120}'|“[^”]{1,120}”/g;

const FUNCTION_WORDS = new Set(("a about above after again against all also although always an and another any are " +
  "as at be because been before being below between both but by can cannot could did " +
  "do does down during each either few for from further had has have he her here his " +
  "how however i if in into is it its itself just may might more most much must my " +
  "neither no nor not of off on once one only or other our out over own same she " +
  "should since so some such than that the their them then there these they this " +
  "those though through thus to too under until up upon very was we were what when " +
  "where whether which while who whose why will with within would you your").split(" "));
const FIRST_PERSON = ["we", "our", "us", "ours", "ourselves", "i", "my", "me", "mine"];
const UNIVERSAL = new Set(["counterweight_rate", "attenuator_rate", "contrastive_rate", "concessive_rate",
  "balanced_sentence_frac", "long_word_rate", "numeric_token_rate"]);
const REGISTERS = { paper: null, letter: UNIVERSAL, docs: UNIVERSAL };
const SHOWN = [
  ["counterweight_rate", "stock hedges /1000w"], ["contrastive_rate", "contrastive constructions /1000w"],
  ["attenuator_rate", "attenuators /1000w"], ["concessive_rate", "concessives /1000w"],
  ["balanced_sentence_frac", "self-qualifying sentences"], ["passive_per_clause", "passive per clause"],
  ["first_person_rate", "first person /1000w"], ["sent_len_mean", "mean sentence length"],
  ["sent_len_iqr", "sentence length IQR"], ["punct_comma_rate", "commas /1000w"],
  ["numeric_token_rate", "numeric tokens /1000w"], ["long_word_rate", "long words /1000w"],
  ["nominalisation_rate", "nominalisation /1000w"],
];

// ---- formatting that matches Python's ----------------------------------------------
function pyFixed(v, d) {
  // Python rounds half to even on the exact binary value; toFixed rounds half up.
  const scaled = v * Math.pow(10, d);
  if (Math.abs(scaled % 1) === 0.5) {
    const floor = Math.floor(scaled);
    const n = floor % 2 === 0 ? floor : floor + 1;
    return (n / Math.pow(10, d)).toFixed(d);
  }
  return v.toFixed(d);
}
const pct = (v) => pyFixed(v * 100, 0) + "%";
const padR = (s, w) => s.length >= w ? s : s + " ".repeat(w - s.length);
const padL = (s, w) => s.length >= w ? s : " ".repeat(w - s.length) + s;
const commas = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
const words = (t) => t.match(RE_WORD) || [];
const count = (t, re) => (t.match(re) || []).length;
const pysplit = (s) => s.trim() === "" ? [] : s.trim().split(/\s+/);

// ---- the measures ----------------------------------------------------------------
export function dropQuoted(text) { return text.replace(RE_QUOTED, " "); }

export function stripMarkup(text) {
  text = text.replace(/```[\s\S]*?```/g, " ");
  text = text.replace(/`[^`]*`/g, " ");
  text = text.replace(/^\s*\|.*\|\s*$/gm, " ");
  text = text.replace(/^\s{0,3}#{1,6}\s+.*$/gm, " ");
  text = text.replace(/^\s*[-*+]\s+/gm, "");
  text = text.replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1");
  text = text.replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1");
  return text.replace(/[ \t]+/g, " ");
}

export function sentences(text) {
  const flat = text.replace(/\s*\n\s*/g, " ").trim();
  return flat.split(RE_SENT).filter(s => words(s).length >= 3);
}

export function measure(text) {
  const unquoted = dropQuoted(text);
  const w = words(text);
  const n = w.length || 1;
  const sents = sentences(text);
  const lens = sents.length ? sents.map(s => words(s).length).sort((a, b) => a - b) : [0];
  const rate = (c) => 1000.0 * c / n;
  const low = w.map(x => x.toLowerCase());
  let balanced = 0;
  for (const s of sents) {
    const q = dropQuoted(s);
    if (new RegExp(RE_CONCESSIVE.source, "i").test(q) || new RegExp(RE_COUNTERWEIGHT.source, "i").test(q)) balanced++;
  }
  const passive = count(text, /\b(?:is|are|was|were|be|been|being)\s+\w+ed\b/gi);
  const clauses = Math.max(sents.length, 1);
  let fp = 0; for (const x of low) if (FIRST_PERSON.includes(x)) fp++;
  return {
    counterweight_rate: rate(count(unquoted, RE_COUNTERWEIGHT)),
    contrastive_rate: rate(count(unquoted, RE_CONTRASTIVE)),
    attenuator_rate: rate(count(unquoted, RE_ATTENUATOR)),
    concessive_rate: rate(count(unquoted, RE_CONCESSIVE)),
    balanced_sentence_frac: balanced / clauses,
    passive_per_clause: passive / clauses,
    first_person_rate: rate(fp),
    sent_len_mean: lens.reduce((a, b) => a + b, 0) / lens.length,
    sent_len_iqr: lens[Math.floor(3 * lens.length / 4)] - lens[Math.floor(lens.length / 4)],
    punct_comma_rate: rate((text.match(/,/g) || []).length),
    numeric_token_rate: rate(count(text, RE_NUMERIC)),
    long_word_rate: rate(w.filter(x => x.length >= 8).length),
    nominalisation_rate: rate(count(text, /\b\w{4,}(?:tion|sion|ment|ness|ity|ance|ence|ism)s?\b/gi)),
    _n_words: w.length,
    _n_sentences: sents.length,
    _median_len: lens[Math.floor(lens.length / 2)],
  };
}


function introShape(text, sents, nWords, out) {
  const n = sents.length;
  if (n < 5) { out("\nintroduction: too short to score (under 5 sentences)"); return; }
  const first = sents[0];
  const OP = [
    ["importance or wide use (papers 11%)", /(?:play(?:s)? (?:a|an) (?:key|important|crucial|central|vital|major)|(?:has|have) (?:attracted|received|gained|drawn|garnered)|(?:is|are) (?:one of the most|widely|among the most|of (?:great|considerable))|great (?:attention|interest|potential)|(?:emerged|promising) (?:as|candidates?)|extensively (?:studied|investigated|used)|(?:has|have) been (?:widely|extensively))/i],
    ["recent growth (papers 5%)", /(?:in recent (?:years|decades)|recently|over the (?:past|last)|the last (?:decade|few years))/i],
    ["societal need (papers 6%)", /(?:the (?:need|demand|challenge)|increasing (?:demand|concern|levels?)|global (?:warming|energy|climate)|greenhouse gas|energy crisis|co2 emissions?|climate change|environmental|air pollution|clean(?:er)? energy)/i],
    ["definition (papers 4%)", /(?:(?:are|is) a (?:class|family|type|group|series) of|consist(?:s|ing)? of|composed of|constructed (?:from|by)|known as|refers? to)/i],
  ];
  let kind = "plain factual claim (papers 74%)";
  for (const [label, re2] of OP) { if (re2.test(first)) { kind = label; break; } }
  const GAP = /(?:remains? (?:unclear|unknown|challenging|elusive|an open|a challenge|poorly understood|limited|scarce|to be)|little (?:is known|attention|work|information)|few (?:studies|reports|works|attempts)|no (?:study|report|systematic|general)|has not (?:been|yet)|have not (?:been|yet)|not (?:yet|fully|well) (?:been )?(?:understood|explored|studied|established|investigated|addressed|clear)|is still (?:lacking|missing|unclear|unknown|debated)|lack of|open question|to date|(?:however|but|yet|unfortunately|despite)[^.]{0,120}(?:difficult|challeng|hinder|limit|problem|unclear|unknown|remains?|not been|little|few|scarce|hamper|suffer))/i;
  const PUR = /(?:in this (?:work|paper|study|article|contribution|letter)|here,? we|in the present (?:work|study|paper)|the (?:aim|purpose|goal|objective) of (?:this|the present)|this (?:work|paper|study) (?:presents|reports|describes|examines|investigates|addresses|focuses|aims)|we (?:report|present|propose|develop|investigate|examine|study|demonstrate|show|introduce|address|extend|apply|use|explore)\b)/i;
  let gi = null, pi = null;
  sents.forEach((x, i) => { if (gi === null && GAP.test(x)) gi = i; if (pi === null && PUR.test(x)) pi = i; });
  const ann = count(text, /(?:we (?:find|found|show|demonstrate|observe) that|our (?:results|findings|calculations|simulations) (?:show|reveal|indicate|suggest|demonstrate))/gi);
  const MARK = /\[(?:\d{1,3})(?:\s?[,\u2013-]\s?\d{1,3})*\]|\((?:[A-Z][A-Za-z-]+(?: et al\.?)?(?:,| and [A-Z][A-Za-z-]+)? ?\d{4}[a-z]?(?:; ?)?)+\)/g;
  const dens = 1000 * count(text, MARK) / Math.max(nWords, 1);
  out("\nintroduction shape (266 published introductions)");
  out(`  opener: ${kind}`);
  if (gi === null) out("  gap: none found (papers state one in 51%, as a concrete lack)");
  else out(`  gap: sentence ${gi + 1}, ${pct(gi / n)} of the way in (papers 18-68%)`);
  if (pi === null) out("  purpose statement: none found (papers 53%: 'In this work, we...')  <<");
  else out(`  purpose statement: sentence ${pi + 1}, ${pct(pi / n)} of the way in (papers 36-77%)${pi / n < 0.25 ? "  <<" : ""}`);
  out(`  length: ${nWords} words (papers 515-779); citations ${pyFixed(dens, 1)}/1000w (papers 9.7-44.2)${dens < 5 ? "  <<" : ""}`);
  if (text.includes("?")) out("  contains a literal question (papers 3%): the gap implies the question instead");
  if (ann) out(`  announces findings ${ann}x (papers do this in 8% of introductions)`);
}

function suggestSequences(text, sents, data, out) {
  const tri = data["trigrams.json"].trigrams;
  const cont = new Map();
  for (const g in tri) {
    const [a, b, d] = g.split(" "); const k = a + " " + b;
    if (!cont.has(k)) cont.set(k, new Map());
    const m = cont.get(k); m.set(d, (m.get(d) || 0) + tri[g]);
  }
  let total = 0, unatt = 0;
  out("\nsequence suggestions (papers 69-76% unattested on all triples)");
  sents.forEach((snt, i) => {
    const w = words(snt).map(x => x.toLowerCase());
    const grams = []; for (let j = 0; j < w.length - 2; j++) grams.push(w[j] + " " + w[j + 1] + " " + w[j + 2]);
    const miss = grams.filter(g => !(g in tri));
    total += grams.length; unatt += miss.length;
    const useful = miss.map(g => [g, cont.get(g.split(" ").slice(0, 2).join(" "))]).filter(([, t]) => t);
    if (!useful.length) return;
    out(`  [${i + 1}] ${snt.slice(0, 110)}`);
    for (const [g, top] of useful.slice(0, 6)) {
      const [a, b] = g.split(" ");
      const alts = [...top.entries()].sort((x, y) => y[1] - x[1]).slice(0, 3).map(([k, v]) => `${k} (${v})`).join(", ");
      out(`        '${g}' -> after '${a} ${b}' papers write: ${alts}`);
    }
  });
  if (total) out(`  unattested triples: ${unatt}/${total} = ${pct(unatt / total)}`);
}

// data: { "reference.json": {...}, "group_reference.json": {...}, "vocab.json": {...},
//         "formulas.json": {...}, "trigrams.json": {...}, ["sequences.json": {...}] }
export function report(text, data, { register = "paper", reference = "papers", top = 13, name = "draft", suggest = false, intro = false } = {}) {
  const lines = []; const out = (s = "") => lines.push(s);
  text = stripMarkup(text);
  const ref = data[{ papers: "combined_reference.json", corpus: "reference.json", group: "group_reference.json" }[reference] || "combined_reference.json"].features;
  const vocab = data["vocab.json"], formulas = data["formulas.json"];
  const m = measure(text);
  const sents = sentences(text);

  out(`${name}: ${commas(m._n_words)} words, ${m._n_sentences} sentences\n`);
  if (m._n_sentences === 0) { out("nothing to measure"); return lines.join("\n") + "\n"; }
  if (m._n_words < 300) out("under 300 words; these measures are noisy at this length\n");
  out(padR("", 38) + padL("draft", 9) + padL(reference, 16));
  out("-".repeat(64));
  const allowed = REGISTERS[register];
  if (allowed) { out(`register: ${register}. Genre-specific targets are not applied; ${allowed.size} universal measures shown.`); out(); }
  let off = 0, shown = 0;
  for (const [key, label] of SHOWN.slice(0, top)) {
    if (!(key in ref)) continue;
    if (allowed && !allowed.has(key)) continue;
    const lo = ref[key].p25, hi = ref[key].p75, v = m[key];
    shown++;
    const mark = (lo <= v && v <= hi) ? "" : "  <<";
    if (mark) off++;
    out(padR(label, 38) + padL(pyFixed(v, 2), 9) + padL(pyFixed(lo, 2), 8) + "-" + padR(pyFixed(hi, 2), 7) + mark);
  }
  out(`\n${shown - off}/${shown} inside the published range`);

  const total = vocab.total_words, freq = vocab.freq;
  const seen = new Map();
  for (const x of words(text)) { const l = x.toLowerCase(); seen.set(l, (seen.get(l) || 0) + 1); }
  const rare = [...seen.entries()].filter(([w]) => w.length >= 4 && !FUNCTION_WORDS.has(w) && 1e6 * (freq[w] || 0) / total < 1.0);
  if (rare.length) {
    rare.sort((a, b) => ((freq[a[0]] || 0) - (freq[b[0]] || 0)) || (b[1] - a[1]));
    out(`\nwords the corpus does not use (${rare.length}):`);
    for (const [w, c] of rare.slice(0, 12)) out(`  ${padR(w, 24)} used ${c}x   ${(freq[w] || 0) === 0 ? "absent" : "rare"}`);
    out("  Technical terms belong here. General-purpose words do not:");
    out("  papers quantify where those characterise.");
  }

  const low = words(text).map(x => x.toLowerCase());
  let known;
  if (data["sequences.json"]) known = new Set(data["sequences.json"].trigrams);
  else { known = new Set(); for (const g in data["trigrams.json"].trigrams) if (g.split(" ").filter(x => FUNCTION_WORDS.has(x)).length >= 2) known.add(g); }
  const conn = [];
  for (let j = 0; j < low.length - 2; j++) { const g = [low[j], low[j + 1], low[j + 2]]; if (g.filter(x => FUNCTION_WORDS.has(x)).length >= 2) conn.push(g.join(" ")); }
  if (conn.length) {
    const hit = conn.filter(g => known.has(g)).length / conn.length;
    out(`\nconnective sequences papers have used: ${pct(hit)} (papers 39-65%, p05-p95)${hit >= 0.39 ? "" : "  <<"}`);
    if (hit < 0.39) {
      const missing = [...new Set(conn)].filter(g => !known.has(g));
      out(`  sequences no paper in 6.4M words makes (${missing.length}), first 15:`);
      for (const g of missing.slice(0, 15)) out(`    ${g}`);
      out("  Rebuild the sentence around a sequence papers use; the formulas below are a start.");
    }
  }

  const VERB = /\b(?:is|are|was|were|has|have|had|can|may|could|should|will|shows?|gives?|leads?|results?|depends?|becomes?|remains?|increases?|decreases?|follows?|corresponds?|indicates?|suggests?|requires?|provides?|yields?|occurs?|forms?)\b/;
  const gaps = [];
  for (const snt of sents) { const mv = VERB.exec(snt); if (mv) gaps.push(pysplit(snt.slice(0, mv.index)).length); }
  if (gaps.length) {
    const late = gaps.filter(g => g > 12).length / gaps.length;
    const back = sents.filter(s => /^(?:This|These|Such|Here|It)\b/.test(s)).length / sents.length;
    out("\nstructure (Gopen and Swan)");
    out(`  sentences with >12 words before the verb: ${pct(late)} (papers 15-29%)${late > 0.35 ? "  <<" : ""}`);
    out(`  sentences opening on This/These/Such/Here/It: ${pct(back)} (papers 5-19%)${back < 0.03 ? "  <<" : ""}`);
  }

  const INTEGRAL = /\b(?:[A-Z][a-z]+ et al\.?|[A-Z][a-z]+ and (?:co-?workers|colleagues))[^.]{0,40}?\b(?:showed|show|reported|report|found|find|proposed|developed|observed|demonstrated|suggested|noted|derived|studied|calculated|computed|measured|obtained|presented|pointed out)\b|\b(?:work|study|results|decomposition|analysis|model) of [A-Z][a-z]+ and (?:co-?workers|colleagues)/g;
  const cites = text.match(INTEGRAL) || [];
  const MARK = /\[(?:\d{1,3}|n)(?:\s?[,–-]\s?\d{1,3})*\]|\((?:[A-Z][A-Za-z-]+(?: et al\.?)?(?:,| and [A-Z][A-Za-z-]+)? ?\d{4}[a-z]?(?:; ?)?)+\)/g;
  const nMarks = count(text, MARK);
  const dens = 1000 * nMarks / Math.max(m._n_words, 1);
  if (register === "paper" && m._n_words >= 300) out(`  citations: ${nMarks} = ${pyFixed(dens, 1)} per 1000 words (papers 9.9-17.2)${dens < 5 ? "  <<" : ""}`);
  const CL = /\b(?:which|that|because|although|whereas|while|when|if|since|so that|as|where)\b|, and\b|, but\b|; /g;
  if (sents.length) {
    const cl = sents.map(s => 1 + count(s, CL));
    const meanCl = cl.reduce((a, b) => a + b, 0) / cl.length, three = cl.filter(x => x >= 3).length / cl.length;
    out(`  clauses per sentence: ${pyFixed(meanCl, 2)} (papers 1.6-2.2); three or more: ${pct(three)} (papers 13-33%)${(meanCl > 2.3 || three > 0.36) ? "  <<" : ""}`);
  }
  if (cites.length) {
    out(`  author named to introduce a finding (${cites.length}; papers 4 in 103,000 words):`);
    for (const h of cites.slice(0, 5)) out(`    ${h.trim().slice(0, 70)}`);
    out("  State the finding in your words and attach the reference.");
  }

  if (sents.length) {
    const nS = sents.length;
    const colon = sents.filter(x => /[a-z\]]:\s+[a-z]/.test(x)).length / nS;
    const byag = sents.filter(x => /\b(?:is|are|was|were)\s+(?:\w+ly\s+)?\w+ed\s+by\s+(?:the|a|an|every|each|all|most|any)\b/.test(x)).length / nS;
    const dep = 1000 * count(text, /\bdepending on\b/g) / Math.max(m._n_words, 1);
    const thf = 1000 * count(text, /\b(?:is|are|was|were)\s+(?:therefore|thus)\b/g) / Math.max(m._n_words, 1);
    const flags = [];
    if (colon > 0.03) flags.push(`colon inside the sentence ${pct(colon)} (papers 1%)`);
    if (byag > 0.03) flags.push(`passive with a by-agent ${pct(byag)} (papers 1%)`);
    if (dep > 0.4) flags.push(`'depending on' ${pyFixed(dep, 1)}/1000w (papers 0.1)`);
    if (thf > 0.3) flags.push(`'is therefore' ${pyFixed(thf, 1)}/1000w (papers 0.03)`);
    if (flags.length) { out("  draft habits papers do not share:"); for (const f of flags) out(`    ${f}`); }
  }

  const RULES = [
    ["quantity noun + at/from/among (papers: of)", /\b(?:uptake|heat|spread|deviation|amount|density|capacity|loading|enthalpy|rate|value|values|distribution|coefficient|fraction|ratio|number)\s+(?:at|from|among)\b(?!\s+(?:which|that))/gi],
    ["comparison with bare 'experiment' (papers: the experimental value/data)", /\b(?:than|with|to|from)\s+experiment\b(?!al|s)/gi],
    ["a factor without a number", /\bby an? (?:similar|comparable|large|small|considerable) factor\b/gi],
    ["abstract-noun claim papers do not make", /\bis (?:a|an|the) (?:source|property|limitation|indication|reflection|feature|hallmark|sign) of\b/gi],
    ["'therefore' inside the verb (papers open the sentence with it)", /\b(?:is|are|was|were|has|have|can|could|would|should)\s+therefore\b/gi],
    ["non-canonical noun (training data, test data, Henry constant, Boltzmann weight)", /\b(?:training data|test data|henry constant|boltzmann weight)\b/gi],
    ["'at low/high coverage' or 'in the ... limit' where a variable was set", /\bat (?:low|high) coverage\b|\bin the (?:low|high|zero)[- ]\w+ limit\b/gi],
  ];
  const hits = [];
  for (const [label, re] of RULES) { const k = count(text, re); if (k) hits.push([label, k]); }
  if (hits.length) { out("  general rules (SKILL.md, from the suggestions):"); for (const [label, k] of hits) out(`    ${k}x  ${label}`); }

  if (intro) introShape(text, sents, m._n_words, out);
  if (suggest) suggestSequences(text, sents, data, out);

  const ftotal = formulas.total_words, fset = formulas.formulas;
  const present = new Set(); for (let j = 0; j < low.length - 2; j++) present.add(low[j] + " " + low[j + 1] + " " + low[j + 2]);
  const top60 = Object.entries(fset).sort((a, b) => b[1] - a[1]).slice(0, 60);
  const used = top60.filter(([g]) => present.has(g)).length;
  out(`\nconnective formulas: ${used}/60 used (${pct(used / 60)}; papers 7-13%)`);
  if (used / 60 < 0.07) {
    out("  heavily used in papers and missing here:");
    for (const [g, c] of top60.filter(([g]) => !present.has(g)).slice(0, 8)) out(`    ${padR(g, 26)}${padL(pyFixed(1e6 * c / ftotal, 0), 6)} per million`);
  }
  return lines.join("\n") + "\n";
}

// ---- command line (Node) ----------------------------------------------------------
if (typeof process !== "undefined" && process.argv && process.argv[1] && /check\.js$/.test(process.argv[1])) {
  const fs = await import("node:fs"); const path = await import("node:path"); const url = await import("node:url");
  const here = path.dirname(url.fileURLToPath(import.meta.url));
  const args = process.argv.slice(2);
  const opt = (flag, d) => { const i = args.indexOf(flag); return i >= 0 ? args[i + 1] : d; };
  const file = args.find(a => !a.startsWith("--") && !["paper", "letter", "docs", "papers", "corpus", "group"].includes(a));
  const data = {};
  for (const f of ["reference.json", "group_reference.json", "combined_reference.json", "vocab.json", "formulas.json", "trigrams.json", "sequences.json"]) {
    const p = path.join(here, "data", f); if (fs.existsSync(p)) data[f] = JSON.parse(fs.readFileSync(p, "utf8"));
  }
  const text = fs.readFileSync(file, "utf8");
  process.stdout.write(report(text, data, { register: opt("--register", "paper"), reference: opt("--reference", "papers"),
    suggest: args.includes("--suggest"), intro: args.includes("--intro"), name: path.basename(file) }));
}
