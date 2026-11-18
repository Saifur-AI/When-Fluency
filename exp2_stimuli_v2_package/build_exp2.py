"""Experiment 2: transplant real stuttering patterns into written job-interview answers.

For each answer we build 7 versions:
  F   fluent original
  S3, S8, S15   stuttered at about 3, 8 and 15 %SS
  T3, T8, T15   typical-disfluency control: "um"/"uh" at the same positions and count as S3/S8/S15

Stutter positions are nested (the S3 events are a subset of S8, which are a subset of S15), so the dose
contrast changes only the amount of stuttering. Event types and repetition counts are sampled from the
Valente-coded events aligned to FluencyBank Timestamped (stutter_inventory.json); the position mix follows
the observed share of stutters on function words (about 17%).
"""
import argparse, json, random, re
import cmudict
import pandas as pd

CMU = cmudict.dict()
FUNCTION = set("""a an the and or but so if of to in on at by for with from as is are was were be been it its
this that these those i me my we our us you your he him his she her they them their there here not no do did
does have has had will would can could should just than then also about into out up""".split())
CONTINUANT = set("sfvzmnlrhwyaeiou")
LEVELS = [3, 8, 15]
FILLERS = ["um", "uh"]


def syllables(word):
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w:
        return 0
    if w in CMU:
        return max(1, sum(ch[-1].isdigit() for ch in CMU[w][0]))
    w2 = w.replace("'", "")
    n = len(re.findall(r"[aeiouy]+", w2))
    if w2.endswith("e") and not w2.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def flesch(text):
    sents = max(1, len(re.findall(r"[.!?]", text)))
    words = re.findall(r"[A-Za-z']+", text)
    syl = sum(syllables(w) for w in words)
    return round(206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words), 2)


def split_token(tok):
    """'Hello,' -> ('', 'Hello', ',')"""
    m = re.match(r"^([^A-Za-z0-9']*)([A-Za-z0-9'-]+)([^A-Za-z0-9']*)$", tok)
    return m.groups() if m else ("", tok, "")


def first_syllable(w):
    m = re.match(r"[^aeiouy]*[aeiouy]+", w.lower())
    s = m.group(0) if m else w[0].lower()
    return s if len(s) < len(w) else w[0].lower()


def render(word, kind, reps):
    low = word.lower()
    if word == "I":
        frag, low = "I", "I"
    else:
        frag = None
    if kind == "ISR":
        f = frag or low[0]
        out = "-".join([f] * reps + [low])
    elif kind == "SR":
        f = frag or first_syllable(low)
        out = "-".join([f] * reps + [low])
    elif kind == "B":
        out = f"{frag or low[0]}... {low}"
    elif kind == "P":
        out = (frag or low[0]) * 3 + low[1:] if word != "I" else "Iii"
    elif kind == "MUR":
        out = f"{low}, {low}"
    else:
        out = low
    # keep the original capitalisation of the target word
    if word[0].isupper():
        out = out[0].upper() + out[1:]
    return out


def sample_kind(rng, shares, word):
    kinds = list(shares)
    weights = [shares[k] for k in kinds]
    for _ in range(20):
        k = rng.choices(kinds, weights)[0]
        if k == "P" and word[0].lower() not in CONTINUANT:
            continue  # prolongation needs a sound that can be held
        return k
    return "ISR"


def sample_reps(rng, iters):
    vals = [int(x) for x in iters]
    w = list(iters.values())
    return min(rng.choices(vals, w)[0], 5)


def build(answer_id, qid, question, text, inv, seed):
    rng = random.Random(seed)
    toks = text.split()
    parts = [split_token(t) for t in toks]
    syl = sum(syllables(p[1]) for p in parts)
    n_max = round(syl * LEVELS[-1] / 100)
    # weighted order of candidate positions: about 17% of stutters fall on function words
    idx = [i for i, p in enumerate(parts) if re.match(r"[A-Za-z]", p[1])]
    fn = [i for i in idx if parts[i][1].lower() in FUNCTION]
    ct = [i for i in idx if i not in fn]
    w = {i: (0.17 / max(len(fn), 1)) if i in fn else (0.83 / max(len(ct), 1)) for i in idx}
    order, pool = [], list(idx)
    while pool:
        i = rng.choices(pool, [w[j] for j in pool])[0]
        pool.remove(i)
        order.append(i)
    # avoid stuttering on two neighbouring words where possible
    chosen = []
    for i in order:
        if len(chosen) == n_max:
            break
        if all(abs(i - j) > 1 for j in chosen):
            chosen.append(i)
    for i in order:
        if len(chosen) == n_max:
            break
        if i not in chosen:
            chosen.append(i)
    events = {}
    for i in chosen:
        k = sample_kind(rng, inv["kind_share"], parts[i][1])
        reps = sample_reps(rng, inv["iterations"]["SR" if k == "SR" else "ISR"]) if k in ("ISR", "SR") else 1
        events[i] = (k, reps)

    rows = [dict(answer_id=answer_id, question_id=qid, question=question, version="F", version_label="fluent",
                 level=0, text=text, positions="", event_types="")]
    for lvl in LEVELS:
        n = round(syl * lvl / 100)
        pos = sorted(chosen[:n])
        s_toks, t_toks = [], []
        fi = 0
        for i, (pre, word, post) in enumerate(parts):
            if i in pos:
                k, reps = events[i]
                s_toks.append(pre + render(word, k, reps) + post)
                filler = FILLERS[fi % 2]; fi += 1
                if pre == "" and word[0].isupper() and word != "I":
                    t_toks.append(filler.capitalize() + " " + word[0].lower() + word[1:] + post)
                else:
                    t_toks.append(pre + filler + " " + word + post)
            else:
                s_toks.append(pre + word + post)
                t_toks.append(pre + word + post)
        types = ";".join(f"{i}:{events[i][0]}x{events[i][1]}" for i in pos)
        rows.append(dict(answer_id=answer_id, question_id=qid, question=question, version=f"S{lvl}",
                         version_label=f"stuttered {lvl}%SS", level=lvl, text=" ".join(s_toks),
                         positions=" ".join(map(str, pos)), event_types=types))
        rows.append(dict(answer_id=answer_id, question_id=qid, question=question, version=f"T{lvl}",
                         version_label=f"typical-disfluency control {lvl}%SS", level=lvl, text=" ".join(t_toks),
                         positions=" ".join(map(str, pos)), event_types=""))
    for r in rows:
        r["n_events"] = len(r["positions"].split()) if r["positions"] else 0
        r["syllables_fluent"] = syl
        r["pss_actual"] = round(100 * r["n_events"] / syl, 2) if r["version"].startswith("S") else 0.0
        r["word_count"] = len(re.findall(r"[A-Za-z0-9']+", r["text"]))
        r["readability"] = flesch(r["text"])
        r["seed"] = seed
        r["source"] = "new_v2"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+")
    ap.add_argument("--inventory")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    inv = json.load(open(a.inventory))
    data = {}
    for f in a.answers:
        data.update(json.load(open(f)))
    rows = []
    for qid in sorted(data):
        q = data[qid]
        for n, text in enumerate(q["answers"], 1):
            aid = f"X2-{qid}-{n:02d}"
            rows.extend(build(aid, qid, q["question"], text, inv, seed=a.seed * 1000 + int(qid[1:]) * 100 + n))
    order = ["F", "S3", "S8", "S15", "T3", "T8", "T15"]
    df = pd.DataFrame(rows)
    df["version"] = pd.Categorical(df.version, order, ordered=True)
    df = df.sort_values(["answer_id", "version"])
    df.to_csv(a.out, index=False)
    print(df.groupby("version", observed=True).agg(n=("text", "size"), pss=("pss_actual", "mean"),
                                                    words=("word_count", "mean")).round(2))


if __name__ == "__main__":
    main()
