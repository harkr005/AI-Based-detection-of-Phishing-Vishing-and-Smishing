"""Scambait calls -> 16 kHz mono wav + our-Whisper transcripts. The corpus the
FTC robocalls cannot be.

WHY. `llm_acts_7b.json` says it plainly: of 1,353 FTC robocalls, 78% contain no
scam act at all, and credential/payment/remote fire on 17 calls between them. A
robocall is the bait ("press 1 to speak to an agent"); the fraud happens after
the transfer, and that half was never recorded. So the acts the taxonomy was
built for have almost no training examples.

NDSS 2024, "An analysis of scam baiting calls" (arxiv 2307.01965), built 341
calls / 90 h from six public YouTube channels, and its scam-type breakdown is
exactly the missing half: SSN 138, refund 110, tech support 63, gift card 12.
Their dataset link was never released, so this reproduces the RECIPE, not the
download - same channels, same vetting idea.

Three properties nothing else public has: real live two-party conversation, the
audio is public so it goes through OUR faster-whisper (identical provenance to
the 27 calls and HarperValley), and the pipeline already exists.

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER FROM THIS: scam-baiters are not real
victims, so the VICTIM half is performed. It is usable here because act_model
scores what the CALLER does and the scammer half is genuine - say that out loud
rather than letting an examiner find it. The baiter also steers the call, which
is a second reason not to read the victim side.

Usage:  python -m scratch._scambait --fetch [--per N]
        python -m scratch._scambait --transcribe [--limit N]
Both are resumable: they skip what is already on disk.
"""
import json
import random
import re
import sys
import time

import data                       # before torch

ROOT = data.DATA / "vishing" / "scambait"
INDEX = ROOT / "index.json"
DROPPED = ROOT / "dropped.json"
TRANS = ROOT / "transcripts.json"

# The six channels the NDSS paper drew from, with its per-channel call counts.
CHANNELS = {
    "Boda Scambaits": ("BodaScambaits", 129),
    "Scammer Jammer": ("scammerjammer", 70),
    "Scammer Payback": ("ScammerPayback", 63),
    "scambait tv": ("scambaittv", 33),
    "IRLrosie": ("IRLrosie", 18),
    "Rinoa Poison": ("RinoaPoison", 17),
}
# 2026-10-24: seven more English scam-baiting channels, 20 calls each target.
# These are a TEST set only (act_model.SCAMBAIT_NEW_CHANNELS): never judged,
# never trained on, fixed before any of their calls was scored.
NEW_CHANNELS = {
    "Kitboga": ("KitbogaShow", 20),
    "Jim Browning": ("JimBrowning", 20),
    "Scambaiter": ("Scambaiter", 20),
    "NanoBaiter": ("NanoBaiter", 20),
    "Scam Sandwich": ("ScamSandwich", 20),
    "Deeveeaar": ("Deeveeaar", 20),
    "ScammerRevolts": ("ScammerRevolts", 20),
}
CHANNELS.update(NEW_CHANNELS)

# The paper's vetting rule as a title filter: keep the calls where the baiter
# plays it straight, drop the entertainment. Cheap and imperfect - it reads the
# title, not the call - so `dropped.json` keeps what it threw away for review.
DROP = re.compile(r"\b(prank|troll|rage|rages|raging|angry|destroy|deleted|"
                  r"hack(ed|ing)?|revenge|karaoke|song|sings?|rickroll|meme|"
                  r"funny|scream|cat|dog|compilation|highlight|reaction|react|"
                  r"shorts?|live|stream|q&a|podcast|vlog|announcement)\b", re.I)
MIN_DUR, MAX_DUR = 180, 2400      # a conversation, not a clip and not a stream


def enumerate_channel(handle, want):
    """Flat listing -> [(id, title, dur)], newest first."""
    from yt_dlp import YoutubeDL
    opts = {"extract_flat": "in_playlist", "quiet": True, "no_warnings": True,
            "playlistend": want * 4}      # headroom for what the filter drops
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/@{handle}/videos",
                                download=False)
    return [(e["id"], e.get("title", ""), e.get("duration") or 0)
            for e in info.get("entries", []) if e]


def fetch(per=None):
    from yt_dlp import YoutubeDL
    import imageio_ffmpeg
    ROOT.mkdir(parents=True, exist_ok=True)
    index = json.load(open(INDEX)) if INDEX.exists() else []
    dropped = json.load(open(DROPPED)) if DROPPED.exists() else []
    have = {r["id"] for r in index} | {r["id"] for r in dropped}
    opts = {"format": "bestaudio/best", "quiet": True, "no_warnings": True,
            "outtmpl": str(ROOT / "%(id)s.%(ext)s"),
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                "preferredcodec": "wav"}],
            "postprocessor_args": {"extractaudio": ["-ar", "16000", "-ac", "1"]}}
    t0 = time.time()
    for name, (handle, paper_n) in CHANNELS.items():
        # --new: only the 2026-10-24 test channels. Topping up an original
        # channel would change the fixed held-out set (Scammer Jammer) mid-study.
        if "--new" in sys.argv and name not in NEW_CHANNELS:
            continue
        # Per channel, and counting what is already indexed: a rerun that
        # ignores this fetches the paper's count AGAIN. It did, once - the
        # corpus reached 258 Boda calls against a target of 129.
        want = (per or paper_n) - sum(1 for r in index if r["channel"] == name)
        if want <= 0:
            print(f"{name}: target already met", flush=True)
            continue
        try:
            vids = enumerate_channel(handle, want)
        except Exception as e:                       # a dead channel is not fatal
            print(f"{name}: listing failed ({e})", flush=True)
            continue
        keep, drop_now = [], []
        for vid, title, dur in vids:
            if vid in have:
                continue
            why = ("title" if DROP.search(title)
                   else "duration" if not MIN_DUR <= dur <= MAX_DUR else None)
            if why:
                drop_now.append({"id": vid, "title": title, "dur": dur, "why": why})
                continue
            keep.append((vid, title, dur))
            if len(keep) >= want:
                break
        dropped += drop_now
        print(f"{name}: {len(vids)} listed, {len(drop_now)} vetted out, "
              f"downloading {len(keep)}", flush=True)
        for k, (vid, title, dur) in enumerate(keep, 1):
            try:
                with YoutubeDL(opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            except Exception as e:
                dropped.append({"id": vid, "title": title, "dur": dur,
                                "why": f"download: {type(e).__name__}"})
                continue
            wav = ROOT / f"{vid}.wav"
            if not wav.exists():                     # postprocessor did not run
                dropped.append({"id": vid, "title": title, "dur": dur,
                                "why": "no wav"})
                continue
            for stale in ROOT.glob(f"{vid}.*"):      # keep the wav, drop the m4a
                if stale != wav:
                    stale.unlink()
            index.append({"id": vid, "title": title, "dur": dur, "channel": name,
                          "url": f"https://www.youtube.com/watch?v={vid}",
                          "label": 1, "mock": False,
                          "path": str(wav)})
            json.dump(index, open(INDEX, "w"), indent=1)
            json.dump(dropped, open(DROPPED, "w"), indent=1)
            if k % 5 == 0 or k == len(keep):
                print(f"  {k}/{len(keep)}  {(time.time() - t0) / 60:.1f} min",
                      flush=True)
    json.dump(dropped, open(DROPPED, "w"), indent=1)
    hrs = sum(r["dur"] for r in index) / 3600
    print(f"\n{len(index)} calls, {hrs:.1f} h  ({len(dropped)} dropped)")


def transcribe(limit=10 ** 6):
    import voice_model as vm
    index = json.load(open(INDEX))
    done = json.load(open(TRANS)) if TRANS.exists() else {}
    todo = [r for r in index if r["id"] not in done]
    # Seeded shuffle so a partial run is a CHANNEL-BALANCED sample, not the
    # first N of whichever channel happens to sit at the top of the index.
    random.Random(0).shuffle(todo)
    todo = todo[:limit]
    print(f"{len(index)} calls, {len(done)} done, transcribing {len(todo)}",
          flush=True)
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        words = vm.transcribe(ROOT / f"{r['id']}.wav")
        done[r["id"]] = " ".join(w for _, _, w in words)
        json.dump(done, open(TRANS, "w"))
        if k % 5 == 0 or k == len(todo):
            el = (time.time() - t0) / 60
            print(f"  {k}/{len(todo)}  {el:.1f} min  "
                  f"eta {el / k * (len(todo) - k):.0f} min", flush=True)
    n = sorted(len(t.split()) for t in done.values())
    if n:
        print(f"{len(done)} transcripts, median {n[len(n) // 2]} words, "
              f"{sum(1 for x in n if x < 20)} under 20 words")


if __name__ == "__main__":
    def arg(flag, default=None):
        return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    if "--fetch" in sys.argv:
        fetch(per=arg("--per"))
    if "--transcribe" in sys.argv:
        transcribe(limit=arg("--limit", 10 ** 6))
    if not {"--fetch", "--transcribe"} & set(sys.argv):
        print(__doc__)
