# NLP_labEval_Ngrams
# N-Gram Language Model with Perplexity

A simple NLP project that implements **Unigram, Bigram, and Trigram** language models using a small sentiment-based corpus on the topic **"Shelter"**. The models use **Laplace (Add-1) Smoothing** and compute the **perplexity** of a test sentence.

## Features

- Builds Unigram, Bigram, and Trigram models
- Generates count and probability tables/matrices
- Applies Laplace (Add-1) smoothing
- Calculates perplexity for a test sentence
- Uses **NLTK** and **Pandas**

## Requirements

- Python 3
- NLTK
- Pandas

Install dependencies:

```bash
pip install nltk pandas
```

## Run

```bash
python ngram_perplexity.py
```

## Dataset

Training corpus:

```
<s> Good shelters offer warm beds </s>
<s> Weak shelters collapse during storms </s>
```

Test sentence:

```
<s> Good shelters collapse during storms </s>
```

## Output

The program prints:

- Vocabulary
- Unigram count & probability table
- Bigram count & probability matrices
- Trigram count & probability table
- Perplexity of the test sentence for Unigram, Bigram, and Trigram models
