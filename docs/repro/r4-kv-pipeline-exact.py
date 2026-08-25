# Greedy server output, hashed. Run through r4-kv-pipeline-exact.sh.
import hashlib, json, sys, urllib.request

PORT, NTOK, NPARA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

SHORT = [
    "Explain how a B-tree index speeds up range queries in a relational database, step by step.",
    "Write a Python function that merges two sorted lists, then explain its complexity.",
    "Summarise the tradeoffs between optimistic and pessimistic concurrency control.",
    "Describe, in order, what happens when a CPU takes a page fault on a memory-mapped file.",
]
# a long, deterministic filler so that decode runs against a deep KV window
PARA = ("A B-tree index stores keys in sorted order across a shallow, balanced tree. "
        "Range queries descend once to the first qualifying leaf and then walk the leaf "
        "chain sequentially, so the cost is one descent plus the size of the range. ")
DEEP = [
    "Summarise the text above in exactly five sentences.",
    "List three claims the text above makes, then say which is weakest and why.",
]

def ask(label, prompt):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": NTOK, "temperature": 0, "top_k": 1, "seed": 1234}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
    except Exception as e:
        print(f"{label} REQUEST_FAILED {type(e).__name__}")
        return False
    m = d["choices"][0]["message"]
    # reasoning models put most of the generation in reasoning_content; hash both
    body_text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    t = d.get("timings", {})
    print(f"{label} {hashlib.sha256(body_text.encode()).hexdigest()[:16]} "
          f"prompt_n={t.get('prompt_n')} n={t.get('predicted_n')} "
          f"tg={t.get('predicted_per_second'):.3f}")
    return True

ok = True
for i, q in enumerate(SHORT):
    ok &= ask(f"short{i}", q)
for i, q in enumerate(DEEP):
    ok &= ask(f"deep{i} ", (PARA * NPARA) + "\n\n" + q)
sys.exit(0 if ok else 1)
