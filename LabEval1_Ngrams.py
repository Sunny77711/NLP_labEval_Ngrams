"""
N-Gram Perplexity Exercise — Topic: SHELTER
=============================================
Corpus (2 sentiment-contrasted sentences):
    Positive : <s> Good shelters offer warm beds </s>
    Negative : <s> Weak shelters collapse during storms </s>

Test sentence:
    <s> Good shelters collapse during storms </s>

We build Unigram, Bigram and Trigram models with Laplace (Add-1)
smoothing, print the count / probability matrices, and compute the
perplexity of the test sentence under each model.
"""

import math
from collections import defaultdict
import pandas as pd
from nltk.util import ngrams

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ------------------------------------------------------------------
# 1. CORPUS
# ------------------------------------------------------------------
positive_doc = "<s> Good shelters offer warm beds </s>".split()
negative_doc = "<s> Weak shelters collapse during storms </s>".split()
corpus = [positive_doc, negative_doc]

# Vocabulary in order of first appearance (mirrors the reference exercise style)
vocab = []
for doc in corpus:
    for w in doc:
        if w not in vocab:
            vocab.append(w)
V = len(vocab)

print("Vocabulary:", vocab)
print("|V| =", V)

# ------------------------------------------------------------------
# 2. UNIGRAM MODEL
# ------------------------------------------------------------------
unigram_counts = defaultdict(int)
N_tokens = 0
for doc in corpus:
    for w in doc:
        unigram_counts[w] += 1
        N_tokens += 1

unigram_probs = {w: (unigram_counts[w] + 1) / (N_tokens + V) for w in vocab}

uni_df = pd.DataFrame(
    {"Count": [unigram_counts[w] for w in vocab],
     "P_laplace": [round(unigram_probs[w], 3) for w in vocab]},
    index=vocab,
)
print("\n=== UNIGRAM TABLE (N =", N_tokens, ") ===")
print(uni_df)

# ------------------------------------------------------------------
# 3. BIGRAM MODEL
# ------------------------------------------------------------------
bigram_counts = defaultdict(lambda: defaultdict(int))
row_totals_bi = defaultdict(int)
for doc in corpus:
    for w1, w2 in ngrams(doc, 2):
        bigram_counts[w1][w2] += 1
        row_totals_bi[w1] += 1

bi_count_df = pd.DataFrame(
    [[bigram_counts[w1][w2] for w2 in vocab] for w1 in vocab],
    index=vocab, columns=vocab,
)
bi_count_df["RowTotal"] = [row_totals_bi[w1] for w1 in vocab]

def bigram_prob(w1, w2):
    return (bigram_counts[w1][w2] + 1) / (row_totals_bi[w1] + V)

bi_prob_df = pd.DataFrame(
    [[round(bigram_prob(w1, w2), 3) for w2 in vocab] for w1 in vocab],
    index=vocab, columns=vocab,
)

print("\n=== BIGRAM COUNT MATRIX ===")
print(bi_count_df)
print("\n=== BIGRAM LAPLACE PROBABILITY MATRIX ===")
print(bi_prob_df)

# ------------------------------------------------------------------
# 4. TRIGRAM MODEL  (sparse -> shown as history -> next-word table)
# ------------------------------------------------------------------
trigram_counts = defaultdict(lambda: defaultdict(int))
history_totals_tri = defaultdict(int)
for doc in corpus:
    for w1, w2, w3 in ngrams(doc, 3):
        trigram_counts[(w1, w2)][w3] += 1
        history_totals_tri[(w1, w2)] += 1

def trigram_prob(w1, w2, w3):
    hist = (w1, w2)
    return (trigram_counts[hist][w3] + 1) / (history_totals_tri[hist] + V)

rows = []
for hist, nexts in trigram_counts.items():
    for w3, c in nexts.items():
        rows.append({
            "History (w1,w2)": hist,
            "Next word": w3,
            "Count": c,
            "HistoryTotal": history_totals_tri[hist],
            "P_laplace": round(trigram_prob(hist[0], hist[1], w3), 3),
        })
tri_df = pd.DataFrame(rows)
print("\n=== TRIGRAM COUNT / PROBABILITY TABLE (seen trigrams) ===")
print(tri_df.to_string(index=False))

# ------------------------------------------------------------------
# 5. PERPLEXITY
# ------------------------------------------------------------------
def perplexity(tokens, n):
    """Compute Laplace-smoothed perplexity for a wrapped, tokenized sentence."""
    if n == 1:
        words = tokens[1:]              # <s> is not "predicted" by a unigram model
        logp = sum(math.log(unigram_probs[w]) for w in words)
        N = len(words)
    else:
        grams = list(ngrams(tokens, n))
        logp = 0.0
        for g in grams:
            if n == 2:
                p = bigram_prob(g[0], g[1])
            elif n == 3:
                p = trigram_prob(g[0], g[1], g[2])
            else:
                raise ValueError("n must be 1, 2, or 3")
            logp += math.log(p)
        N = len(grams)
    return math.exp(-logp / N), N

test_sentence = "<s> Good shelters collapse during storms </s>".split()
print("\n================ TEST SENTENCE ================")
print(" ".join(test_sentence))

for n, name in [(1, "Unigram"), (2, "Bigram"), (3, "Trigram")]:
    pp, N = perplexity(test_sentence, n)
    print(f"{name:8s} -> Perplexity = {pp:.3f}   (N = {N})")
