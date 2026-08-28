# The polish service

A Cloudflare Worker that holds the API key and forwards a draft, with the skill's
policy, to the model. The page never sees the key. Limits: three polishes per computer
per day, two hundred per day in total, twelve thousand characters per draft.

## Deploy, once (about ten minutes, free tier)

Everything below happens in the Cloudflare dashboard; nothing needs installing.

1. Create an account at dash.cloudflare.com if you have none.
2. **Workers & Pages → Create → Create Worker.** Name it `writing-style-polish`, Deploy,
   then **Edit code**: replace the contents with `worker.js` from this folder, and Deploy.
3. **Storage & Databases → KV → Create namespace**, named `RATE`.
4. Back in the worker: **Settings → Bindings → Add → KV namespace**: variable name `RATE`,
   namespace `RATE`.
5. **Settings → Variables and Secrets → Add**: type *Secret*, name `ANTHROPIC_API_KEY`,
   value your key. Add another, type *Text*, name `ALLOWED_ORIGIN`, value
   `https://parsakh00.github.io`.
6. Copy the worker's URL from its overview page; it looks like
   `https://writing-style-polish.<your-subdomain>.workers.dev`. That URL goes into
   `POLISH_URL` in `scripts/build_site.py`.

Set a monthly spend limit on the key in the Anthropic console as well. The worker's
limits bound how many polishes can happen; the spend limit bounds what they can cost.

## Test it

```
curl -X POST https://writing-style-polish.<your-subdomain>.workers.dev \
  -H "content-type: application/json" -H "origin: https://parsakh00.github.io" \
  -d '{"draft":"The uptake from the sample was higher than experiment. We set the value according to the data.","register":"paper"}'
```

A JSON `{"text": "..."}` comes back, with an `x-remaining` header counting down from 3.
