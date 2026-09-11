// Cloudflare Worker: polishes a draft with the skill's policy, holding the API key on
// the server side. The page sends {draft, register}; the worker adds the policy and the
// checker's report is computed on the page and passed along; the key never leaves here.
//
// Limits (all enforced here, not on the page):
//   3 polishes per visitor per day, by IP address
//   DAILY_TOTAL polishes per day for everyone together
//   MAX_WORDS words (and MAX_CHARS characters) of draft
//
// Bindings: ANTHROPIC_API_KEY (secret), RATE (KV namespace), ALLOWED_ORIGIN (var).

const MAX_WORDS = 5000;
const MAX_CHARS = 40000;
const PER_IP_PER_DAY = 3;
const DAILY_TOTAL = 200;
const MODEL = "claude-sonnet-5";
const POLICY_URL = "https://parsakh00.github.io/writing-style-lab/tool/SKILL.md";

const SYSTEM_HEAD = "You revise scientific prose to the policy below. Work only with what the draft contains: every claim, number, citation marker and technical term stays exactly as given, and nothing is added from outside it, no fact, explanation, example, interpretation, qualifier or context the author did not write. Change register, phrasing, sentence structure and citation practice only; the content is the author's and is not yours to extend or correct. Never introduce a number, value, name or reference that is not in the draft: where the draft gives no number, keep its wording, and where a citation is missing, leave the sentence uncited rather than adding a placeholder. Citations: when the draft introduces a finding by naming its authors and carries a citation marker, for example 'Smith and coworkers found that X [12]', write the finding in the author's words with the same marker, 'X [12]', and keep the marker exactly as written, in its position after the claim it belongs to. When a finding names its authors and has no marker, keep the sentence as it is; do not add a marker, a placeholder or a name. Before returning, read every sentence once more against this list, measured on 245 papers: no colon inside a sentence introducing evidence; no passive with its agent attached by 'by'; no 'so' as a clause connective; no 'depending on' at the end of a sentence; no 'is therefore' inside the verb; no claim carried by 'is a source of', 'is a property of' or 'is a limitation of'; 'at all pressures', not 'at every pressure'; 'overestimates' or 'underestimates the uptake', not 'overbinds'; 'the experimental value' or 'data', never bare 'experiment'; a quantity noun takes 'of'. Never add significance commentary: no 'paving the way', 'highlights the importance', 'plays a key role', 'opens new avenues'; state the result and stop. Keep every reported failure, surprise or discrepancy exactly as the author wrote it, and keep every cross-reference such as 'as mentioned above' or 'described below'. Return the revised text and nothing else.\n\n";

function cors(env, extra = {}) {
  return { "access-control-allow-origin": env.ALLOWED_ORIGIN || "*", "access-control-allow-methods": "POST, OPTIONS",
           "access-control-allow-headers": "content-type", "access-control-expose-headers": "x-remaining", ...extra };
}
const json = (env, status, body, extra = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...cors(env, extra) } });

async function sign(env, text) {
  // HMAC of the text with the API key as the secret; proves a second pass follows a
  // first pass on this exact text. Nothing about the key is recoverable from it.
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(env.ANTHROPIC_API_KEY), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(text));
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}

export default {
  async fetch(request, env, ctx) {
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
      const used = parseInt((await env.RATE.get(`v6:ip:${day}:${ip}`)) || "0", 10);
      return json(env, 200, { remaining: Math.max(PER_IP_PER_DAY - used, 0), limit: PER_IP_PER_DAY });
    }
    if (request.method !== "POST") return json(env, 405, { error: "POST only" });

    const origin = request.headers.get("origin") || "";
    if (env.ALLOWED_ORIGIN && origin !== env.ALLOWED_ORIGIN) return json(env, 403, { error: "origin not allowed" });

    let body;
    try { body = await request.json(); } catch { return json(env, 400, { error: "bad request" }); }
    const draft = String(body.draft || "").trim();
    const pass2 = body.pass === 2 && typeof body.token === "string";
    const register = ["paper", "letter", "docs"].includes(body.register) ? body.register : "paper";
    const report = String(body.report || "").slice(0, 8000);
    const nWords = draft.split(/\s+/).filter(Boolean).length;
    if (nWords < 5) return json(env, 400, { error: "paste a draft" });
    if (nWords > MAX_WORDS || draft.length > MAX_CHARS) return json(env, 413, { error: `drafts are limited to ${MAX_WORDS} words` });

    // Limits. KV keys expire at the end of the day they were made.
    const day = new Date().toISOString().slice(0, 10);
    const ip = request.headers.get("cf-connecting-ip") || "unknown";
    const ipKey = `v6:ip:${day}:${ip}`, totalKey = `v6:total:${day}`;
    const used = parseInt((await env.RATE.get(ipKey)) || "0", 10);
    const total = parseInt((await env.RATE.get(totalKey)) || "0", 10);
    if (pass2) {
      // A second pass revises the first pass's own output against the checker's
      // findings. It is free, and it is accepted only with the token the first pass
      // issued for exactly that text, so it cannot be used as a second free polish.
      if ((await sign(env, draft)) !== body.token) return json(env, 403, { error: "second pass without a first" });
    } else if (used >= PER_IP_PER_DAY) return json(env, 429, { error: `this computer has used its ${PER_IP_PER_DAY} polishes for today` }, { "x-remaining": "0" });
    if (total >= DAILY_TOTAL) return json(env, 429, { error: "the daily limit for everyone has been reached; try tomorrow" }, { "x-remaining": "0" });

    const policy = await (await fetch(POLICY_URL, { cf: { cacheTtl: 3600 } })).text();
    const user = pass2
      ? "Register: " + register + "\n\nThis text is your own revision. The checker still finds the constructions below in it. Rewrite only the sentences those findings name, and copy every other sentence character for character. The lines reading 'papers write:' are corpus statistics, never words to insert: reword the sentence naturally so the flagged sequence disappears, and where no natural rewording exists, keep the sentence as it is. Every rewritten sentence must still read as ordinary prose, with every number, unit and technical term unchanged:\n" + report + "\n\nThe text:\n" + draft
      : "Register: " + register + "\n\nThe checker's report on this draft:\n" + report + "\n\nThe draft:\n" + draft;
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01" },
      // Thinking is off: with it on, a long draft used the whole output budget on thought
      // and returned no text. The checker's report already says what to change. The
      // answer streams, so the page can show the revision as it is written.
      body: JSON.stringify({ model: MODEL, max_tokens: 16384, thinking: { type: "disabled" }, stream: true,
                             system: SYSTEM_HEAD + policy, messages: [{ role: "user", content: user }] }),
    });
    if (!r.ok) {
      let msg = "upstream error";
      try { const j = await r.json(); if (j.error) msg = j.error.message; } catch {}
      return json(env, 502, { error: msg });
    }
    if (!pass2) {
      const ttl = 86400;
      await env.RATE.put(ipKey, String(used + 1), { expirationTtl: ttl });
      await env.RATE.put(totalKey, String(total + 1), { expirationTtl: ttl });
    }
    const remaining = pass2 ? Math.max(PER_IP_PER_DAY - used, 0) : PER_IP_PER_DAY - used - 1;

    // The upstream stream is re-sent to the page one text piece at a time. The closing
    // event carries the token for the second pass and the reply's shape, so an empty
    // answer can be diagnosed from the page rather than guessed at.
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const enc = new TextEncoder();
    const send = (obj) => writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n"));
    const pump = (async () => {
      const dec = new TextDecoder();
      const reader = r.body.getReader();
      let buf = "", text = "", stopReason = null, usage = null, modelName = null;
      const blocks = [];
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop();
          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            let ev;
            try { ev = JSON.parse(line.slice(5)); } catch { continue; }
            if (ev.type === "content_block_start") blocks.push(ev.content_block && ev.content_block.type);
            else if (ev.type === "content_block_delta" && ev.delta && ev.delta.type === "text_delta") {
              text += ev.delta.text;
              await send({ delta: ev.delta.text });
            } else if (ev.type === "message_start" && ev.message) modelName = ev.message.model;
            else if (ev.type === "message_delta") { if (ev.delta) stopReason = ev.delta.stop_reason; if (ev.usage) usage = ev.usage; }
            else if (ev.type === "error") await send({ error: ev.error ? ev.error.message : "the stream broke" });
          }
        }
        await send({ done: true, token: await sign(env, text),
                     shape: { stop_reason: stopReason, blocks, usage, model: modelName } });
      } catch (e) {
        try { await send({ error: String(e) }); } catch {}
      }
      try { await writer.close(); } catch {}
    })();
    if (ctx) ctx.waitUntil(pump);
    return new Response(readable, { status: 200, headers: { "content-type": "text/event-stream", "cache-control": "no-store",
                                                            ...cors(env, { "x-remaining": String(remaining) }) } });
  },
};
