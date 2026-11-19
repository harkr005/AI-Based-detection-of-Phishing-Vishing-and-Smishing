"""Does the 2026-09-23 mail-parser fix help or hurt the SHIPPED email model (v3)?

v3 was trained on text from the OLD data._pot_body, which kept <style>/<script>
contents (CSS/JS) as body text and returned "" for an empty text/plain beside a
full text/html. The fix changes 43.8% of personal-mail rows (median -64% length)
and 47.5% of phishing_pot. Live Gmail is now parsed the NEW way, so this reads
v3 on both parsings of the same mail:

  - Inbox ham 2021-26, the box no model trained on -> false-positive rate
  - promotions/updates ham 2025-26, the box v3 trained on
  - phishing_pot sample -> how much real phishing still reads as non-ham

The old parser's text comes from _inbox_eval.py's cache, which predates the fix.
PRIVATE DATA: rates only, no message text.

Run: python scratch/_parser_fix_eval.py
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402  (before torch)
import pandas as pd  # noqa: E402
import email_model as em  # noqa: E402

OLD_CACHE = data.PERSONAL / "_self_6000.parquet"          # old parser (pre-fix)
NEW_CACHE = data.PERSONAL / "_self_6000_parserfix.parquet"
POT_N = 2000
HAM = em.LABELS.index("ham")
PRE_FIX = "4442226"  # last commit with the old parser


def old_data():
    """data.py as of PRE_FIX, pointed at the real dataset folders."""
    src = subprocess.run(["git", "show", f"{PRE_FIX}:data.py"], capture_output=True,
                         text=True, encoding="utf-8", check=True,
                         cwd=Path(data.__file__).parent).stdout
    path = Path(tempfile.mkdtemp()) / "data_prefix.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("data_prefix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in ("DATA", "POT", "SA_RAW", "PERSONAL", "INDIA", "KB"):
        setattr(mod, k, getattr(data, k))
    return mod


def slices(d):
    ham = d[(d.label_3 == "ham") & d.date.dt.year.between(2021, 2026)]
    return [("Inbox ham 2021-26 (unseen box)", ham[ham.mbox == "Inbox"]),
            ("promo/updates ham 2025-26 (seen)",
             ham[ham.mbox.isin({"Category promotions", "Category updates"})
                 & (ham.date.dt.year >= 2025)])]


def not_ham(tok, model, texts):
    return 100 * (em.predict(tok, model, texts)[0].argmax(1) != HAM).mean()


if __name__ == "__main__":
    assert OLD_CACHE.exists(), "run scratch/_inbox_eval.py once first (old-parser cache)"
    old = pd.read_parquet(OLD_CACHE)
    if NEW_CACHE.exists():
        new = pd.read_parquet(NEW_CACHE)
    else:
        new = data.load_personal_mail(person="self", limit=6000)
        new.to_parquet(NEW_CACHE)

    # 2025+ only: v3 and v4 both TRAIN on phishing_pot <=2024 (_train_unified).
    pot = data.load_phishing_pot()
    pot_new = pot[pot.date.dt.year >= 2025]
    pot_new = pot_new.sample(min(POT_N, len(pot_new)), random_state=0)
    pot_old = old_data().load_phishing_pot().loc[pot_new.index]
    assert (pot_old.file.values == pot_new.file.values).all()

    root = Path(data.__file__).resolve().parent / "models"
    cols = [("v3 old-parse", "email_distilbert_v3", "old"),
            ("v3 new-parse", "email_distilbert_v3", "new")]
    if (root / "email_distilbert_v4" / "model.pt").exists():
        cols.append(("v4 new-parse", "email_distilbert_v4", "new"))

    rows = [(label, a, b) for (label, a), (_, b) in zip(slices(old), slices(new))]
    rows.append(("phishing_pot 2025+ (all phishing)", pot_old, pot_new))
    res = {}
    for tag, name, which in cols:
        em.OUT = root / name
        em.load.cache_clear()
        tok, model = em.load()
        for label, a, b in rows:
            res[(label, tag)] = not_ham(tok, model, (a if which == "old" else b).text)

    print()
    print(f"{'% NOT ham':38s} {'n':>6s}" + "".join(f"{t:>15s}" for t, _, _ in cols))
    for label, a, b in rows:
        print(f"{label:38s} {len(b):>6d}" + "".join(f"{res[(label, t)]:>14.1f}%" for t, _, _ in cols))
