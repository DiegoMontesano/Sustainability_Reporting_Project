"""
Ferrari Sustainability Report — Quality Analysis
================================================
Implements:
  1. Loughran-McDonald (2011) dictionary-based tone analysis
  2. Gunning Fog Index readability measure (Li, 2008)
  3. Vague language frequency (impression management proxy)
  4. LDA topic modelling (Latent Dirichlet Allocation)

Sources:
  - Annual Report 2025: Ferrari_Annual_Report_2025.pdf
    → Sustainability Statement: pp. 174-303 (lines 8800-16360 of extracted text)
  - Annual Report 2024: Ferrari_Annual_Report_2024.pdf
    → Sustainability Statement: pp. 163-294 (lines 11317-21412 of extracted text)
  - LM Master Dictionary: Loughran-McDonald_MasterDictionary_1993-2025.xlsx

Dependencies:
    pip install pandas openpyxl scikit-learn pdfminer.six
    apt-get install poppler-utils  (for pdftotext)
"""

import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ── 0. CONFIGURATION ──────────────────────────────────────────────────────────

PDF_2025       = "Ferrari_Annual_Report_2025.pdf"
PDF_2024       = "Ferrari_Annual_Report_2024.pdf"
LM_DICT_PATH   = "Loughran-McDonald_MasterDictionary_1993-2025.xlsx"

# Correct line boundaries for sustainability section only
# 2025: pp. 174-303  |  2024: pp. 163-294
LINES_2025 = (8800, 16360)
LINES_2024 = (11317, 21412)

N_TOPICS       = 10
LDA_SEED       = 42
LDA_MAX_ITER   = 30
VOCAB_SIZE     = 2000
MIN_PARA_WORDS = 30
MIN_CLEAN_WORDS = 10


# ── 1. TEXT EXTRACTION ────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str, out_path: str) -> list[str]:
    """Extract text from PDF using pdftotext -layout, return as list of lines."""
    subprocess.run(["pdftotext", "-layout", pdf_path, out_path], check=True)
    with open(out_path, "r", encoding="utf-8") as f:
        return f.readlines()


def get_sustainability_text(lines: list[str], start: int, end: int) -> str:
    """Slice the sustainability section from the full extracted text."""
    return "".join(lines[start:end])


# ── 2. LM DICTIONARY ─────────────────────────────────────────────────────────

def load_lm_dictionary(path: str) -> dict[str, set]:
    """Load Loughran-McDonald master dictionary, return sets per category."""
    lm = pd.read_excel(path)
    categories = {
        "negative":     set(lm[lm["Negative"]     != 0]["Word"].str.upper()),
        "positive":     set(lm[lm["Positive"]     != 0]["Word"].str.upper()),
        "uncertainty":  set(lm[lm["Uncertainty"]  != 0]["Word"].str.upper()),
        "litigious":    set(lm[lm["Litigious"]    != 0]["Word"].str.upper()),
        "strong_modal": set(lm[lm["Strong_Modal"] != 0]["Word"].str.upper()),
        "weak_modal":   set(lm[lm["Weak_Modal"]   != 0]["Word"].str.upper()),
    }
    print(f"LM dictionary loaded. Neg={len(categories['negative'])}, "
          f"Pos={len(categories['positive'])}, Unc={len(categories['uncertainty'])}")
    return categories


# ── 3. FOG INDEX ──────────────────────────────────────────────────────────────

def count_syllables(word: str) -> int:
    """Estimate syllable count using vowel-group heuristic."""
    word = word.lower()
    count, prev_vowel = 0, False
    for ch in word:
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # Adjust for silent trailing 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_fog_index(text: str) -> dict:
    """
    Compute Gunning Fog Index = 0.4 × (avg sentence length + % complex words).
    Complex word = ≥3 syllables. Reference: Li (2008).
    """
    tokens    = re.findall(r"[A-Za-z]+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.split()) >= 3]
    n         = len(tokens)

    avg_sent_len  = n / len(sentences) if sentences else 0
    complex_words = [t for t in tokens if count_syllables(t) >= 3]
    pct_complex   = 100 * len(complex_words) / n if n else 0
    fog_index     = 0.4 * (avg_sent_len + pct_complex)

    return {
        "total_words":      n,
        "total_sentences":  len(sentences),
        "avg_sentence_len": round(avg_sent_len, 1),
        "complex_words_pct": round(pct_complex, 1),
        "fog_index":        round(fog_index, 2),
    }


# ── 4. TONE & VAGUE LANGUAGE ANALYSIS ────────────────────────────────────────

VAGUE_TERMS = {
    "SIGNIFICANT", "SIGNIFICANTLY", "CERTAIN", "VARIOUS", "SEVERAL",
    "ONGOING", "SUBSTANTIAL", "SUBSTANTIALLY", "APPROPRIATE", "GENERALLY",
    "APPROXIMATELY", "EXPECTED", "BELIEVE", "BELIEVES", "ADEQUATE", "REASONABLY",
}


def analyse_tone(text: str, lm_dict: dict[str, set]) -> dict:
    """
    Compute word category proportions per L&M (2011).
    Returns proportions as % of total words.
    """
    tokens = re.findall(r"[A-Za-z]+", text)
    upper  = [t.upper() for t in tokens]
    n      = len(upper)

    counts = {cat: sum(1 for w in upper if w in words)
              for cat, words in lm_dict.items()}
    counts["vague"] = sum(1 for w in upper if w in VAGUE_TERMS)

    proportions = {cat: round(100 * c / n, 3) for cat, c in counts.items()}

    # Tone balance: pos / (pos + neg)
    pos = counts["positive"]
    neg = counts["negative"]
    proportions["pos_neg_ratio"] = round(100 * pos / (pos + neg), 1) if (pos + neg) else 0

    # Top words per category
    top_words = {
        cat: Counter(w for w in upper if w in words).most_common(10)
        for cat, words in lm_dict.items()
    }

    return {"n": n, "counts": counts, "proportions": proportions, "top_words": top_words}


def print_lm_results(label: str, fog: dict, tone: dict) -> None:
    """Pretty-print results for one year."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total words:          {tone['n']:,}")
    print(f"  Fog Index:            {fog['fog_index']}   (≤17 ideal)")
    print(f"  Avg sentence length:  {fog['avg_sentence_len']} words")
    print(f"  Complex words %:      {fog['complex_words_pct']}%")
    print()
    for cat in ["negative", "positive", "uncertainty", "litigious",
                "strong_modal", "weak_modal", "vague"]:
        pct = tone["proportions"][cat]
        print(f"  {cat:<15}: {pct:.3f}%")
    print(f"  Pos/(Pos+Neg):        {tone['proportions']['pos_neg_ratio']}%  (<60% = no flag)")
    print()
    print("  Top negative words:", tone["top_words"]["negative"][:5])
    print("  Top positive words:", tone["top_words"]["positive"][:5])
    print("  Top uncertainty:   ", tone["top_words"]["uncertainty"][:5])


# ── 5. LDA TOPIC MODELLING ────────────────────────────────────────────────────

STOPWORDS = set("""
a about above across after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further get
got had hadn't has hasn't have haven't having he he'd he'll he's her here hers
herself him himself his how i if in into is isn't it its itself me more most
my no nor not of off on once only or other our out over own same she should so
some such than that the their them then there these they this those through to
too under until up very was we were what when where which while who will with
would you your yourself
ferrari group company report reporting year annual pursuant related accordance
including referred applicable december january february march april may june july
august september october november table annex section chapter page following refer
please note also moreover furthermore however therefore thus hence whereby whereas
figure number per cent percentage point basis million billion thousand euro
esrs csrd such shall may must within without between among across information
disclosure data provided available set forth indicated described shown presented
prepared standards requirements below previous current respective given excluding
""".split())


def clean_paragraph(text: str) -> str:
    """Lowercase, remove non-alpha, drop stopwords and short tokens."""
    text   = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = [t for t in text.split() if len(t) > 3 and t not in STOPWORDS]
    return " ".join(tokens)


def run_lda(text: str, label: str,
            n_topics: int = N_TOPICS,
            vocab_size: int = VOCAB_SIZE) -> tuple:
    """
    Run LDA on sustainability statement text.
    Returns (topic_weights, lda_model, feature_names, vectorizer).
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text)
                  if len(p.strip().split()) >= MIN_PARA_WORDS]
    cleaned    = [c for c in (clean_paragraph(p) for p in paragraphs)
                  if len(c.split()) >= MIN_CLEAN_WORDS]
    print(f"\n{label}: {len(paragraphs)} paragraphs → {len(cleaned)} usable after cleaning")

    # Vectorise
    vectorizer = CountVectorizer(
        max_df=0.85,
        min_df=3,
        max_features=vocab_size,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(cleaned)
    print(f"  DTM shape: {dtm.shape}")

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=LDA_SEED,
        max_iter=LDA_MAX_ITER,
        learning_method="batch",
    )
    lda.fit(dtm)

    feature_names  = vectorizer.get_feature_names_out()
    doc_topics     = lda.transform(dtm)
    topic_weights  = doc_topics.mean(axis=0)

    print(f"\n  {'T#':<4} {'Prevalence':>10}   Top 8 keywords")
    print(f"  {'-'*65}")
    for i in range(n_topics):
        top_idx  = lda.components_[i].argsort()[-8:][::-1]
        top_kws  = [feature_names[j] for j in top_idx]
        print(f"  T{i+1:<3} {100*topic_weights[i]:>9.1f}%   {' | '.join(top_kws)}")

    return topic_weights, lda, feature_names, vectorizer


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

def main():

    # ── Step 1: Extract text from PDFs ────────────────────────────────────────
    print("Extracting text from PDFs...")
    lines_2025 = extract_pdf_text(PDF_2025, "/tmp/ferrari_2025.txt")
    lines_2024 = extract_pdf_text(PDF_2024, "/tmp/ferrari_2024.txt")

    text_2025 = get_sustainability_text(lines_2025, *LINES_2025)
    text_2024 = get_sustainability_text(lines_2024, *LINES_2024)

    print(f"\n2025 sustainability section: {len(re.findall(r'[A-Za-z]+', text_2025)):,} words "
          f"(pp. 174-303, lines {LINES_2025[0]}-{LINES_2025[1]})")
    print(f"2024 sustainability section: {len(re.findall(r'[A-Za-z]+', text_2024)):,} words "
          f"(pp. 163-294, lines {LINES_2024[0]}-{LINES_2024[1]})")

    # ── Step 2: Load LM dictionary ─────────────────────────────────────────────
    print("\nLoading Loughran-McDonald dictionary...")
    lm_dict = load_lm_dictionary(LM_DICT_PATH)

    # ── Step 3: Fog Index ──────────────────────────────────────────────────────
    print("\nComputing Fog Index...")
    fog_2025 = compute_fog_index(text_2025)
    fog_2024 = compute_fog_index(text_2024)

    # ── Step 4: Tone analysis ──────────────────────────────────────────────────
    print("\nRunning LM tone analysis...")
    tone_2025 = analyse_tone(text_2025, lm_dict)
    tone_2024 = analyse_tone(text_2024, lm_dict)

    print_lm_results("2025 — Sustainability Statement (pp. 174-303)", fog_2025, tone_2025)
    print_lm_results("2024 — Sustainability Statement (pp. 163-294)", fog_2024, tone_2024)

    # ── Year-on-year comparison ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  YEAR-ON-YEAR COMPARISON (2024 → 2025)")
    print(f"{'='*60}")
    metrics = [
        ("Fog Index",       fog_2024["fog_index"],          fog_2025["fog_index"]),
        ("Avg sent. length",fog_2024["avg_sentence_len"],   fog_2025["avg_sentence_len"]),
        ("Complex words %", fog_2024["complex_words_pct"],  fog_2025["complex_words_pct"]),
        ("Negative %",      tone_2024["proportions"]["negative"],    tone_2025["proportions"]["negative"]),
        ("Positive %",      tone_2024["proportions"]["positive"],    tone_2025["proportions"]["positive"]),
        ("Uncertainty %",   tone_2024["proportions"]["uncertainty"], tone_2025["proportions"]["uncertainty"]),
        ("Litigious %",     tone_2024["proportions"]["litigious"],   tone_2025["proportions"]["litigious"]),
        ("Pos/(Pos+Neg)",   tone_2024["proportions"]["pos_neg_ratio"], tone_2025["proportions"]["pos_neg_ratio"]),
        ("Vague terms %",   tone_2024["proportions"]["vague"],       tone_2025["proportions"]["vague"]),
    ]
    for label, v24, v25 in metrics:
        delta = round(v25 - v24, 3)
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {label:<22}  2024={v24}  2025={v25}  {arrow}{abs(delta)}")

    # ── Step 5: LDA topic modelling ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LDA TOPIC MODELLING")
    print(f"{'='*60}")

    w25, lda25, feat25, vec25 = run_lda(text_2025, "2025 Sustainability Statement")
    w24, lda24, feat24, vec24 = run_lda(text_2024, "2024 Sustainability Statement")

    print("\nDone. All results printed above.")
    print("To export results to Excel, run the build_corrected_excel.py script.")


if __name__ == "__main__":
    main()
