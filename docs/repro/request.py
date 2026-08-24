import json, sys, urllib.request
port = sys.argv[1]; ntok = int(sys.argv[2])
prompt = "Explain how a B-tree index speeds up range queries in a relational database, step by step."
body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": ntok, "temperature": 0, "top_k": 1, "seed": 1234}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
except Exception as e:
    print(f"REQUEST_FAILED {type(e).__name__}"); sys.exit(1)
t = d.get("timings", {})
print(f"tg={t.get('predicted_per_second'):.3f} n={t.get('predicted_n')} "
      f"draft={t.get('draft_n_accepted')}/{t.get('draft_n')}")
