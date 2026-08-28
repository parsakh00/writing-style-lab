// Cloudflare Worker: polishes a draft with the skill's policy, holding the API key on
// the server side. The page sends {draft, register}; the worker adds the policy and the
// checker's report is computed on the page and passed along; the key never leaves here.
//
// Limits (all enforced here, not on the page):
//   3 polishes per visitor per day, by IP address
//   DAILY_TOTAL polishes per day for everyone together
//   MAX_CHARS characters of draft
//
// Bindings: ANTHROPIC_API_KEY (secret), RATE (KV namespace), ALLOWED_ORIGIN (var).

const MAX_CHARS = 12000;
const PER_IP_PER_DAY = 3;
const DAILY_TOTAL = 200;
const MODEL = "claude-sonnet-5";
const POLICY_URL = "https://parsakh00.github.io/writing-style-lab/tool/SKILL.md";

const SYSTEM_HEAD = "You revise scientific prose to the policy below. Work only with what the draft contains: every claim, number, citation marker and technical term stays exactly as given, and nothing is added from outside it, no fact, explanation, example, interpretation, qualifier or context the author did not write. Change register, phrasing, sentence structure and citation practice only; the content is the author's and is not yours to extend or correct. Never introduce a number, value, name or reference that is not in the draft: where the draft gives no number, keep its wording, and where a citation is missing, leave the sentence uncited rather than adding a placeholder. Citations: when the draft introduces a finding by naming its authors and carries a citation marker, for example 'Smith and coworkers found that X [12]', write the finding in the author's words with the same marker, 'X [12]', and keep the marker exactly as written, in its position after the claim it belongs to. When a finding names its authors and has no marker, keep the sentence as it is; do not add a marker, a placeholder or a name. Return the revised text and nothing else.\n\n";

function cors(env, extra = {}) {
  return { "access-control-allow-origin": env.ALLOWED_ORIGIN || "*", "access-control-allow-methods": "POST, OPTIONS",
           "access-control-allow-headers": "content-type", "access-control-expose-headers": "x-remaining", ...extra };
}
const json = (env, status, body, extra = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...cors(env, extra) } });

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: cors(env) });
    if (request.method === "GET" && new URL(request.url).pathname === "/health") {
      // Reports what is bound, never the values.
      const k = env.ANTHROPIC_API_KEY || "";
      return json(env, 200, { key: k ? `set, ${k.length} chars, starts ${k.slice(0, 11)}` : "missing",
                              origin: env.ALLOWED_ORIGIN || "missing", counters: env.RATE ? "bound" : "missing" });
    }
    if (request.method === "GET" && new URL(request.url).pathname === "/quota") {
      const day = new Date().toISOString().slice(0, 10);
      const ip = request.headers.get("cf-connecting-ip") || "unknown";
      const used = parseInt((await env.RATE.get(`v5:ip:${day}:${ip}`)) || "0", 10);
      return json(env, 200, { remaining: Math.max(PER_IP_PER_DAY - used, 0), limit: PER_IP_PER_DAY });
    }
    if (request.method !== "POST") return json(env, 405, { error: "POST only" });

    const origin = request.headers.get("origin") || "";
    if (env.ALLOWED_ORIGIN && origin !== env.ALLOWED_ORIGIN) return json(env, 403, { error: "origin not allowed" });

    let body;
    try { body = await request.json(); } catch { return json(env, 400, { error: "bad request" }); }
    const draft = String(body.draft || "").trim();
    const register = ["paper", "letter", "docs"].includes(body.register) ? body.register : "paper";
    const report = String(body.report || "").slice(0, 8000);
    if (draft.split(/\s+/).length < 5) return json(env, 400, { error: "paste a draft" });
    if (draft.length > MAX_CHARS) return json(env, 413, { error: `drafts are limited to ${MAX_CHARS} characters` });

    // Limits. KV keys expire at the end of the day they were made.
    const day = new Date().toISOString().slice(0, 10);
    const ip = request.headers.get("cf-connecting-ip") || "unknown";
    const ipKey = `v5:ip:${day}:${ip}`, totalKey = `v5:total:${day}`;
    const used = parseInt((await env.RATE.get(ipKey)) || "0", 10);
    const total = parseInt((await env.RATE.get(totalKey)) || "0", 10);
    if (used >= PER_IP_PER_DAY) return json(env, 429, { error: `this computer has used its ${PER_IP_PER_DAY} polishes for today` }, { "x-remaining": "0" });
    if (total >= DAILY_TOTAL) return json(env, 429, { error: "the daily limit for everyone has been reached; try tomorrow" }, { "x-remaining": "0" });

    const policy = await (await fetch(POLICY_URL, { cf: { cacheTtl: 3600 } })).text();
    const user = "Register: " + register + "\n\nThe checker's report on this draft:\n" + report + "\n\nThe draft:\n" + draft;
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01" },
      // Thinking is off: with it on, a long draft used the whole output budget on thought
      // and returned no text. The checker's report already says what to change.
      body: JSON.stringify({ model: MODEL, max_tokens: 8192, thinking: { type: "disabled" },
                             system: SYSTEM_HEAD + policy, messages: [{ role: "user", content: user }] }),
    });
    const j = await r.json();
    if (!r.ok) return json(env, 502, { error: j.error ? j.error.message : "upstream error" });

    const ttl = 86400;
    await env.RATE.put(ipKey, String(used + 1), { expirationTtl: ttl });
    await env.RATE.put(totalKey, String(total + 1), { expirationTtl: ttl });
    const text = (j.content || []).map(c => c.text || "").join("");
    // The reply's shape travels with the text, so an empty answer can be diagnosed
    // from the page rather than guessed at.
    const shape = { stop_reason: j.stop_reason, blocks: (j.content || []).map(c => c.type), usage: j.usage, model: j.model };
    return json(env, 200, { text, shape }, { "x-remaining": String(PER_IP_PER_DAY - used - 1) });
  },
};
