"""Build the extended FluencyBank stimulus table (Experiment 1).

Inputs
  --romana   FluencyBank Timestamped segment file (segid, segstart, segend, verbatim, intended)
  --valente  folder with Valente interview .cha files (%xstu tier, times in ms)
  --old      previous stimulus table (its excerpts are kept as seeds)
Output
  one row per excerpt per version (C1-C6) plus an alignment log.
"""
import argparse, difflib, glob, os, re
import cmudict
import pandas as pd

FILLERS = {"um", "uh", "er", "erm", "ah", "eh", "hm", "hmm", "mm", "mhm", "uhm", "umm"}
CORE = ("SR", "ISR", "MUR", "P", "B")
PROMPT_RE = re.compile(
    r"\bquestion (one|two|three|four|five|six|number|\d)\b|^(so )?(the )?(first|second|third|fourth|fifth|sixth|last|next) question"
    r"|please (talk|describe)|you can talk about|what do (you|i) think causes|if you (didn't|did not) stutter"
    r"|what else would you|to help them learn|successful communication means|that's all the questions",
    re.I)
EXPLICIT_RE = re.compile(
    r"stutter|stammer|stutt|disfluen|fluen|speech|therap|patholog|\bslp\b|\bnsa\b|\bblock(s|ed|ing)?\b|"
    r"\bfriends\b.{0,20}\bconvention|support group|self.help", re.I)
IMPLICIT_RE = re.compile(
    r"\b(talk|talking|talked|speak|speaking|spoke|communicat\w*|present\w*|phone|nervous|anxi\w*|fear\w*|"
    r"afraid|scared|shy|embarrass\w*|confiden\w*|introduc\w*|say my name|my name|words?|ordering|order food)\b", re.I)
TOPIC_KW = {
    "work": r"\b(work|job|career|office|boss|coworker|co-worker|company|manager|client|customer|business|employ\w*|"
            r"meeting|team|project|interview|vendor|paperwork)\b",
    "study": r"\b(school|class|college|university|teacher|student|degree|study|studied|exam|grade|program|mba|campus|professor)\b",
    "family": r"\b(family|mom|mother|dad|father|parent\w*|brother|sister|son|daughter|kids?|children|wife|husband|"
              r"grand\w*|aunt|uncle|cousin|married)\b",
    "hobbies": r"\b(hobby|hobbies|sport\w*|game\w*|music|guitar|racing|car|play|playing|read|reading|travel\w*|"
               r"run|running|church|volunteer\w*|dj|cook\w*|fish\w*|hike|hiking|movie\w*)\b",
}
CMU = cmudict.dict()


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
    if not words:
        return float("nan")
    syl = sum(syllables(w) for w in words)
    return round(206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words), 2)


def load_valente(folder):
    rows = []
    for f in glob.glob(os.path.join(folder, "*.cha")):
        spk = os.path.basename(f)[:-4]
        for line in open(f, encoding="utf-8", errors="replace"):
            if not line.startswith("%xstu"):
                continue
            body = (line.split("\t", 1)[-1] if "\t" in line else line[6:]).replace("\x15", " ")
            for code, t0, t1 in re.findall(r"(\S+)\s+(-?\d+)_(\d+)", body):
                core = code.split(";")[0].upper()
                parts = re.split(r"[+:,]", core)
                kinds = [p for p in parts if p in CORE]
                if not kinds:
                    continue  # secondary-only behaviour (e.g. "_;V;1")
                rows.append(dict(spk=spk, code=core, kind=kinds[0], t0=int(t0) / 1000, t1=int(t1) / 1000))
    return pd.DataFrame(rows)


def tok(s):
    return [t for t in re.findall(r"[a-z0-9']+", str(s).lower()) if t] if isinstance(s, str) else []


def align_segment(seg, events):
    """Return C1 render pieces, C2 tokens, C5 render pieces and counts for one segment."""
    v, t = tok(seg.verbatim), tok(seg.intended)
    sm = difflib.SequenceMatcher(a=v, b=t, autojunk=False)
    # spans: list of (v_start, v_end, t_insert_index)
    spans, keep = [], [None] * len(v)  # keep[i] = t index for fluent tokens
    raw = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                keep[i1 + k] = j1 + k
        elif op in ("delete", "replace"):
            raw.append((i1, i2))
    # split each disfluent stretch into filler runs, a revision part and a trailing repetition part
    for i1, i2 in raw:
        groups, k = [], i1
        while k < i2:
            f = v[k] in FILLERS
            e = k
            while e < i2 and (v[e] in FILLERS) == f:
                e += 1
            groups.append((k, e, f)); k = e
        for g1, g2, is_f in groups:
            if is_f:
                spans.append(dict(i1=g1, i2=g2, kind="filler")); continue
            nxt = v[g2] if g2 < len(v) else ""
            if nxt in FILLERS:  # look past fillers for the target word
                nxt = next((v[x] for x in range(g2, len(v)) if v[x] not in FILLERS), "")
            cut = g2
            while cut > g1 and nxt and (nxt.startswith(v[cut - 1]) or v[cut - 1] == nxt) and cut - 1 >= g1:
                cut -= 1
                if v[cut] == nxt and cut < g2 - 1:
                    break
            if cut > g1:
                spans.append(dict(i1=g1, i2=cut, kind="revision"))
            if cut < g2:
                spans.append(dict(i1=cut, i2=g2, kind="sld_candidate"))
    for s in spans:
        s["mid"] = (s["i1"] + s["i2"]) / 2
        s["sld"] = False
        nk = next((x for x in range(s["i2"], len(v)) if keep[x] is not None), None)
        s["j"] = keep[nk] if nk is not None else len(t)
    # place Valente events on the verbatim token axis by linear time interpolation
    dur = max(seg.segend - seg.segstart, 1e-6)
    n = max(len(v), 1)
    evs = []
    for e in events:
        pos = ((e["t0"] + e["t1"]) / 2 - seg.segstart) / dur * n
        evs.append(dict(e, pos=min(max(pos, 0), n - 0.5), span=None))
    # greedy matching: closest event-span pairs first, non-filler spans only
    tol = max(3.0, 0.35 * n)
    pairs = sorted(
        ((abs(e["pos"] - s["mid"]) + (0 if s["kind"] == "sld_candidate" else 1.5), ei, si)
         for ei, e in enumerate(evs) for si, s in enumerate(spans) if s["kind"] != "filler"),
        key=lambda x: x[0])
    used_e, used_s = set(), set()
    for d, ei, si in pairs:
        if ei in used_e or si in used_s or d > tol:
            continue
        used_e.add(ei); used_s.add(si)
        evs[ei]["span"] = si
        spans[si]["sld"] = True
        spans[si]["code"] = evs[ei]["kind"]
    # unrepresented events (blocks / prolongations with no text) attach to the next fluent token
    fluent_idx = [k for k in range(len(v)) if keep[k] is not None]
    marks = {}  # v index -> event kind rendered on that token
    for e in evs:
        if e["span"] is not None or not fluent_idx:
            continue
        target = next((k for k in fluent_idx if k >= e["pos"]), fluent_idx[-1])
        if target in marks:
            target = next((k for k in fluent_idx if k > target and k not in marks), None)
            if target is None:
                continue
        marks[target] = e["kind"]
    # render C1 from the verbatim stream, and C5 (v3 rule) alongside it: C1 with every stuttering-like event
    # replaced by one filler slot (FILL) at the same position; typical disfluencies, including Valente MUR
    # (pre-registered as typical), are kept exactly as in C1
    span_at = {s["i1"]: s for s in spans}
    out, out5, k, sld_before = [], [], 0, []  # sld_before: t indices preceded by a stutter event (v2 C5)
    while k < len(v):
        if k in span_at:
            s = span_at[k]
            toks = v[s["i1"]:s["i2"]]
            nxt_k = s["i2"]
            nxt_w = v[nxt_k] if nxt_k < len(v) else ""
            typical = s.get("code") == "MUR" and typical_rep(toks, nxt_w)
            if s["sld"] and s["kind"] == "sld_candidate" and nxt_k < len(v) and nxt_k not in span_at:
                piece = "-".join(toks + [render_word(v[nxt_k], marks.get(nxt_k))])
                out.append(piece)
                if typical:
                    out5.append(piece)
                else:
                    out5.append(FILL)
                    out5.extend(c5_word(v[nxt_k], marks.get(nxt_k)))
                if keep[nxt_k] is not None:
                    sld_before.append(keep[nxt_k])
                k = nxt_k + 1
                continue
            if s["kind"] == "filler":
                out.extend(toks)
                out5.extend(toks)
            elif s["sld"]:
                if s.get("code") == "B":
                    piece = " ".join(toks) + "..."
                elif s["kind"] == "sld_candidate" or len(toks) == 1:
                    piece = "-".join(toks) + "-"
                else:
                    piece = " ".join(toks) + ","
                out.append(piece)
                if typical:
                    out5.append(piece)
                elif s["kind"] == "revision" and len(toks) > 1:
                    # the event was matched to a multi-word revision: keep the revision words (typical)
                    # and put the filler where the stutter was, at the end of the span
                    out5.extend([" ".join(toks) + ",", FILL])
                else:
                    out5.append(FILL)
                sld_before.append(s["j"])
            else:
                out.append(" ".join(toks) + ",")
                out5.append(" ".join(toks) + ",")
            k = s["i2"]
            continue
        w = v[k]
        if k in marks:
            out.append(render_word(w, marks[k]))
            out5.extend(c5_word(w, marks[k]))
            if keep[k] is not None:
                sld_before.append(keep[k])
        else:
            out.append(w)
            out5.append(w)
        k += 1
    n_sld = sum(s["sld"] for s in spans) + len(marks)
    n_td = sum(1 for s in spans if not s["sld"])
    return dict(c1=out, c5v3=out5, c2=t, sld_before=sorted(sld_before), n_events=len(evs), n_matched=len(used_e) + len(marks),
                n_text=len(used_e), n_placed=len(marks),
                n_sld=n_sld, n_td=n_td)


FILL = "\x00FILL\x00"  # filler slot in C5 (v3); build_excerpt fills it with alternating um / uh


def c5_word(w, kind):
    """C5 (v3) tokens for a fluent word carrying a Valente event placed by timing."""
    if kind is None:
        return [w]
    if kind == "MUR" and syllables(w) > 1:  # multisyllabic word repetition: typical, kept as in C1
        return [render_word(w, kind)]
    return [FILL, w]


def typical_rep(toks, target):
    """Pre-registered boundary rule (Ambrose & Yairi): a repetition is typical only if it repeats a
    multisyllabic whole word or a phrase; part-word and single-syllable word repetitions are stuttering-like."""
    if len(set(toks)) > 1 and all(t in CMU and (len(t) > 1 or t in ("a", "i")) for t in toks):
        return True  # phrase repetition
    return all(t == target for t in toks) and syllables(target) > 1


def render_word(w, kind):
    if kind == "B":
        return f"{w[0]}... {w}"
    if kind == "P" and w[0].isalpha():
        return w[0] * 3 + w
    if kind in ("SR", "ISR") and w[0].isalpha():
        # coded by Valente but not visible in the verbatim text: render as a first-sound repetition
        onset = re.match(r"[^aeiouy]*[aeiouy]?", w).group(0) if kind == "SR" else w[0]
        onset = onset if 0 < len(onset) < len(w) else w[0]
        return f"{onset}-{onset}-{w}" if kind == "ISR" else f"{onset}-{w}"
    if kind == "MUR":
        return f"{w}, {w}"
    return w


def sentence(tokens):
    if not tokens:
        return ""
    s = " ".join(tokens)
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\bi'(m|ve|d|ll)\b", lambda m: "I'" + m.group(1), s)
    s = re.sub(r"\bi-", "I-", s)
    s = re.sub(r"-i\b", "-I", s)
    s = s.rstrip(",")
    return s[0].upper() + s[1:] + "."


def build_excerpt(segs, ev_df, disclose=False, c5_mode="v2"):
    """c5_mode 'v2': C2 with fillers at the stuttering positions.
    c5_mode 'v3': C1 with each stuttering-like event replaced by one filler, typical disfluencies kept."""
    c1, c2, c5 = [], [], []
    stats = dict(n_events=0, n_matched=0, n_text=0, n_placed=0, n_sld=0, n_td=0, n_c5_fillers=0)
    fill = ["um", "uh"]
    fi = 0
    for seg in segs.itertuples():
        evs = ev_df[(ev_df.t1 > seg.segstart - 0.05) & (ev_df.t0 < seg.segend + 0.05)]
        evs = evs[((evs.t0 + evs.t1) / 2 >= seg.segstart - 0.05) & ((evs.t0 + evs.t1) / 2 <= seg.segend + 0.05)]
        r = align_segment(seg, evs.to_dict("records"))
        for key in stats:
            stats[key] += r.get(key, 0)
        if r["c1"]:
            c1.append(sentence(r["c1"]))
        if r["c2"]:
            c2.append(sentence(r["c2"]))
        if c5_mode == "v3":
            t5 = []
            for x in r["c5v3"]:
                if x == FILL:
                    x = fill[fi % 2]; fi += 1; stats["n_c5_fillers"] += 1
                t5.append(x)
            if t5:
                c5.append(sentence(t5))
        elif r["c2"]:
            t5 = list(r["c2"])
            for j in sorted(r["sld_before"], reverse=True):
                t5.insert(min(j, len(t5)), fill[fi % 2]); fi += 1
            c5.append(sentence(t5))
    return " ".join(c1), " ".join(c2), " ".join(c5), stats


def classify(text):
    if EXPLICIT_RE.search(text):
        return "mentions"
    return "implicit" if IMPLICIT_RE.search(text) else "clean"


def topic_of(text):
    scores = {k: len(re.findall(p, text, re.I)) for k, p in TOPIC_KW.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--romana"); ap.add_argument("--valente"); ap.add_argument("--old"); ap.add_argument("--out")
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--extra", default="", help="comma-separated first segids of extra windows to add")
    ap.add_argument("--labels", default=None, help="manual topic review: excerpt_id,manual_review")
    a = ap.parse_args()

    df = pd.read_csv(a.romana)
    df["spk"] = df.segid.str.rsplit("_", n=1).str[0]
    df["verbatim"] = df.verbatim.fillna("")
    df["intended"] = df.intended.fillna("")
    df["wi"] = df.intended.map(lambda s: len(tok(s)))
    df["prompt"] = df.verbatim.str.contains(PROMPT_RE.pattern, flags=re.I, regex=True) | df.intended.str.contains(PROMPT_RE.pattern, flags=re.I, regex=True)
    df["explicit"] = df.intended.str.contains(EXPLICIT_RE.pattern, flags=re.I, regex=True) | df.verbatim.str.contains(EXPLICIT_RE.pattern, flags=re.I, regex=True)
    df = df.sort_values(["spk", "segstart"]).reset_index(drop=True)
    ev = load_valente(a.valente)

    old = pd.read_csv(a.old, encoding="utf-8-sig")
    old1 = old[old.version == "C1"]
    seeds = {r.excerpt_id: r.segids.split() for r in old1.itertuples()}
    used = {s for ids in seeds.values() for s in ids}

    # candidate windows: consecutive non-prompt, non-explicit segments, gap <= 3 s, 40-180 intended words
    cands = []
    for spk, d in df.groupby("spk", sort=False):
        cur, wc, last, q = [], 0, None, ""

        def flush():
            nonlocal cur, wc
            if 40 <= wc <= 180:
                cands.append(dict(spk=spk, segids=list(cur), wc=wc, question=q))
            cur, wc = [], 0

        for r in d.itertuples():
            if r.prompt:
                flush(); q = r.intended or r.verbatim; last = r.segend; continue
            if r.explicit or r.segid in used or r.wi == 0 and not r.verbatim.strip():
                flush(); last = r.segend; continue
            if last is not None and r.segstart - last > 3:
                flush()
            if cur and wc + r.wi > 150:
                flush()
            cur.append(r.segid); wc += r.wi; last = r.segend
        flush()
    cand = pd.DataFrame(cands)
    seg_ix = df.set_index("segid")

    def text_of(ids):
        return " ".join(seg_ix.loc[ids, "intended"])

    cand["neutral"] = cand.segids.map(lambda ids: classify(text_of(ids)))

    def n_events(row):
        t0, t1 = seg_ix.loc[row.segids, "segstart"].min(), seg_ix.loc[row.segids, "segend"].max()
        e = ev[(ev.spk == row.spk)]
        mid = (e.t0 + e.t1) / 2
        return int(((mid >= t0) & (mid <= t1)).sum())

    cand["n_ev"] = cand.apply(n_events, axis=1)
    cand = cand[cand.n_ev > 0]  # an excerpt with no coded stuttering cannot test C1 vs C2
    cand.assign(text=cand.segids.map(text_of), segids=cand.segids.map(" ".join)).to_csv(a.out.replace(".csv", "_candidates.csv"), index=False)
    cand["pref"] = cand.wc.between(60, 150).astype(int) * 2 + (cand.neutral == "clean").astype(int)

    # selection: keep seeds, then fill each speaker up to the cap, best-scored first
    chosen = []
    for eid, ids in seeds.items():
        chosen.append(dict(excerpt_id=eid, spk=eid.split("#")[0], segids=ids, previous=True, question=""))
    for spk, d in cand.groupby("spk"):
        have = sum(1 for c in chosen if c["spk"] == spk)
        for r in d.sort_values(["pref", "wc"], ascending=False).head(max(0, a.cap - have)).itertuples():
            chosen.append(dict(excerpt_id=None, spk=spk, segids=r.segids, previous=False, question=r.question))
    for sid in [x for x in a.extra.split(",") if x]:
        hit = cand[cand.segids.map(lambda ids: ids[0] == sid)]
        if len(hit) and not any(c["segids"][0] == sid for c in chosen):
            r = hit.iloc[0]
            chosen.append(dict(excerpt_id=None, spk=r.spk, segids=r.segids, previous=False, question=r.question))
    # name new excerpts after the first segment index so ids are stable
    for c in chosen:
        if c["excerpt_id"] is None:
            c["excerpt_id"] = f"{c['spk']}#n{int(c['segids'][0].rsplit('_', 1)[1]):03d}"

    rows, log = [], []
    roles = ["office_analyst", "client_facing"]
    role_ix = {}
    for c in sorted(chosen, key=lambda c: (c["spk"], c["segids"][0])):
        segs = seg_ix.loc[c["segids"]].reset_index()
        e = ev[ev.spk == c["spk"]]
        c1, c2, c5, st = build_excerpt(segs, e)
        syl = sum(syllables(w) for w in re.findall(r"[A-Za-z']+", c2))
        # core Valente events inside the excerpt's time span
        t0, t1 = segs.segstart.min(), segs.segend.max()
        pss = round(100 * st["n_events"] / syl, 2) if syl else float("nan")
        disc = "Before we start, I should mention that I stutter. "
        neut = "Before we start, I should mention that I'm left-handed. "
        versions = [("C1", "stuttered", c1), ("C2", "fluent", c2), ("C3", "stuttered + disclosure", disc + c1),
                    ("C4", "fluent + disclosure", disc + c2), ("C5", "typical-disfluency control", c5),
                    ("C6", "stuttered + neutral disclosure", neut + c1)]
        role = roles[role_ix.get("n", 0) % 2]
        role_ix["n"] = role_ix.get("n", 0) + 1
        neutral = classify(text_of(c["segids"]))
        q = c["question"]
        if c["previous"]:
            q = old1.set_index("excerpt_id").loc[c["excerpt_id"], "question"]
            q = "" if pd.isna(q) else q
        for code, label, txt in versions:
            rows.append(dict(excerpt_id=c["excerpt_id"], speaker_id=c["spk"], version=code, version_label=label,
                             text=txt, word_count=len(re.findall(r"[A-Za-z0-9']+", txt)), pss=pss,
                             syllables_c2=syl, readability=flesch(txt), role=role, topic=topic_of(c2),
                             neutral_review=neutral, question=q,
                             question_mentions_stuttering=bool(re.search(r"stutter", str(q), re.I)),
                             n_valente_events=st["n_events"], n_sld=st["n_sld"], n_td=st["n_td"],
                             start_s=round(t0, 2), end_s=round(t1, 2), segids=" ".join(c["segids"]),
                             in_previous_set=c["previous"]))
        log.append(dict(excerpt_id=c["excerpt_id"], events=st["n_events"], matched=st["n_matched"],
                        matched_to_text=st["n_text"], placed_by_time=st["n_placed"]))
    out = pd.DataFrame(rows)
    if a.labels:
        lab = pd.read_csv(a.labels)
        out = out.merge(lab, on="excerpt_id", how="left")
        # seeds keep the earlier neutral review; unlabelled new windows were read and judged to be about stuttering
        out["manual_review"] = out.manual_review.fillna(
            out.neutral_review.where(out.in_previous_set, "stuttering_content"))
        if "old_review" in out:
            pass
        # the three earlier excerpts that answer a stuttering question are dropped (agreed plan); for new windows the
        # content itself was reviewed by hand, so the prompt flag is kept as a moderator instead
        q_drop = out.question_mentions_stuttering & out.in_previous_set
        usable = out.manual_review.isin(["clean", "implicit"]) & ~q_drop
        out["exp1_primary"] = usable
        # cap usable excerpts per speaker, keeping earlier excerpts first, then clean, then longer
        first = out[out.version == "C2"].assign(u=usable[out.version == "C2"])
        first = first[first.u].assign(cl=lambda d: (d.manual_review == "clean").astype(int))
        first = first.sort_values(["speaker_id", "in_previous_set", "cl", "word_count"], ascending=[True, False, False, False])
        keep_ids = set(first.groupby("speaker_id").head(a.cap).excerpt_id)
        out["exp1_primary"] = out.excerpt_id.isin(keep_ids)
        out["exclusion_reason"] = ""
        out.loc[out.manual_review == "stuttering_content", "exclusion_reason"] = "content is about stuttering"
        out.loc[out.manual_review == "exclude_sensitive", "exclusion_reason"] = "sensitive personal content"
        out.loc[q_drop, "exclusion_reason"] = "answers a stuttering question"
        out.loc[usable & ~out.exp1_primary, "exclusion_reason"] = "over speaker cap"
        out = out.drop(columns=["neutral_review"]).rename(columns={"manual_review": "neutral_review"})
    out.to_csv(a.out, index=False)
    pd.DataFrame(log).to_csv(a.out.replace(".csv", "_alignment_log.csv"), index=False)
    lg = pd.DataFrame(log)
    print("excerpts", out.excerpt_id.nunique(), "speakers", out.speaker_id.nunique(), "rows", len(out))
    print("events", lg.events.sum(), "rendered", lg.matched.sum(), "matched to transcript disfluency", lg.matched_to_text.sum(),
          "placed by timing", lg.placed_by_time.sum())


if __name__ == "__main__":
    main()
