# The polish service

A Cloudflare Worker that holds the API key and forwards a draft, with the skill's
policy, to the model. The page never sees the key.

Limits, enforced here: three polishes per computer per day, two hundred per day in
total, twelve thousand characters per draft. Requests are accepted only from the
project's page, and only the fixed polishing prompt is ever sent.

`worker.js` is the whole service; `wrangler.toml` names its bindings: the secret
`ANTHROPIC_API_KEY`, the variable `ALLOWED_ORIGIN`, and a KV namespace `RATE` for the
daily counters.
