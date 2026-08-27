# Greedy server output, hashed, over several prefill corpora and prefill lengths.
# Run through r4-kv-pipeline-exact.sh. Compare the hashes across pipeline depths and against a
# build of the parent commit: the pipelined path must reproduce the ordered path exactly.
import hashlib, json, sys, urllib.request

PORT = sys.argv[1]
LENGTHS = [int(x) for x in sys.argv[2].split(",")]   # approximate prefill tokens

# Four corpora with different token statistics, so that the deliveries being pipelined are not
# always the same shape of content: prose, source code, structured records, and dialogue.
CORPORA = {
    "prose": ("A B-tree index stores keys in sorted order across a shallow, balanced tree. "
              "Range queries descend once to the first qualifying leaf and then walk the leaf "
              "chain sequentially, so the cost is one descent plus the size of the range. "),
    "code":  ("static int walk_leaf_chain(struct btree *t, uint64_t lo, uint64_t hi, "
              "int (*cb)(void *, uint64_t), void *ctx) {\n"
              "    struct leaf *l = btree_descend(t, lo);\n"
              "    while (l && l->keys[0] <= hi) {\n"
              "        for (int i = 0; i < l->n; i++) { if (l->keys[i] > hi) return 0; "
              "cb(ctx, l->keys[i]); }\n"
              "        l = l->next;\n    }\n    return 0;\n}\n"),
    "records": ('{"id":%d,"region":"eu-central","bytes":918273,"status":"ok",'
                '"latency_ms":12.75,"tags":["index","range","btree"]}\n'),
    "dialogue": ("Q: Why does the planner prefer a sequential scan here?\n"
                 "A: Because the predicate matches most of the table, and random leaf access "
                 "would cost more than reading every page once.\n"),
}

QUESTIONS = {
    "prose":    "Summarise the text above in exactly five sentences.",
    "code":     "Describe what the function above does, then name one bug it could hide.",
    "records":  "How many distinct fields does each record above have, and what are they?",
    "dialogue": "State the single claim the answers above keep returning to.",
}

def filler(name, target_tokens):
    unit = CORPORA[name]
    # roughly four characters to the token; the exact prefill length is reported per task
    reps = max(1, (target_tokens * 4) // len(unit % 0 if "%d" in unit else unit))
    if "%d" in unit:
        return "".join(unit % i for i in range(reps))
    return unit * reps

def nonce(name, length):
    # The server restores a cached prefix from an earlier task, and a restored window is not
    # numerically the same as a freshly prefilled one, so two tasks that share a long prefix stop
    # measuring the code under test. This makes every task's prefix unique, and it is derived from
    # the task rather than drawn at random so that a control run produces comparable hashes.
    h = hashlib.sha256(f"{name}/{length}".encode()).hexdigest()[:32]
    return f"Session {h}. Ignore this line.\n\n"

def ask(label, prompt, ntok, want_prefill):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": ntok, "temperature": 0, "top_k": 1, "seed": 1234}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=14400) as r:
            d = json.load(r)
    except Exception as e:
        print(f"{label} REQUEST_FAILED {type(e).__name__}", flush=True)
        return False
    m = d["choices"][0]["message"]
    # reasoning models put most of the generation in reasoning_content; hash both
    text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    t = d.get("timings", {})
    # a reused prefix shows up as a prompt_n far below the prompt actually sent; the hash it
    # produces is not comparable to a fresh prefill, so say so rather than reporting it silently
    prompt_n = t.get("prompt_n") or 0
    reused = prompt_n < want_prefill // 2
    print(f"{label:<18} {hashlib.sha256(text.encode()).hexdigest()[:16]} "
          f"prompt_n={prompt_n:<7} n={t.get('predicted_n'):<4} "
          f"pp={t.get('prompt_per_second'):8.2f} tg={t.get('predicted_per_second'):7.3f}"
          f"{'  CACHE_REUSE' if reused else ''}", flush=True)
    return not reused

ok = True
for length in LENGTHS:
    ntok = 256 if length <= 4096 else 128
    for name in CORPORA:
        prompt = nonce(name, length) + filler(name, length) + "\n\n" + QUESTIONS[name]
        ok &= ask(f"{name}@{length}", prompt, ntok, length)
sys.exit(0 if ok else 1)
