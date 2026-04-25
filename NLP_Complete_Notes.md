# 📘 NLP Complete Exam Notes
### Covers Full ETE Syllabus | Formulas + Examples + Numericals

---

## TABLE OF CONTENTS
1. [NLP Introduction & Phases](#1-nlp-introduction--phases)
2. [Semantics & Pragmatics](#2-semantics--pragmatics)
3. [Ambiguity in NLP](#3-ambiguity-in-nlp)
4. [Morphology & Text Preprocessing](#4-morphology--text-preprocessing)
5. [Corpus & Corpus Linguistics](#5-corpus--corpus-linguistics)
6. [Edit Distance — Levenshtein Algorithm](#6-edit-distance--levenshtein-algorithm)
7. [N-gram Language Models](#7-n-gram-language-models)
8. [Smoothing Techniques](#8-smoothing-techniques)
9. [Entropy, Cross-Entropy & Perplexity](#9-entropy-cross-entropy--perplexity)
10. [Bag of Words (BoW) & TF-IDF](#10-bag-of-words-bow--tf-idf)
11. [Parts of Speech (POS) Tagging](#11-parts-of-speech-pos-tagging)
12. [Word Sense Disambiguation (WSD)](#12-word-sense-disambiguation-wsd)
13. [Hidden Markov Model (HMM) & Viterbi](#13-hidden-markov-model-hmm--viterbi)
14. [Parsing — CFG, CNF, CYK, PCFG](#14-parsing--cfg-cnf-cyk-pcfg)
15. [Deep Learning in NLP — CNN, RNN, LSTM](#15-deep-learning-in-nlp--cnn-rnn-lstm)
16. [Real-World NLP Applications](#16-real-world-nlp-applications)

---

# 1. NLP Introduction & Phases

## What is NLP?
Natural Language Processing (NLP) is a subfield of Artificial Intelligence that gives computers the ability to understand, interpret, and generate human language.

**Definition:** NLP is a theoretically motivated range of computational techniques for analyzing and representing naturally occurring text/speech at one or more levels of linguistic analysis, for the purpose of achieving human-like language processing.

## Two Core Components
| Component | Full Form | Direction | Task |
|-----------|-----------|-----------|------|
| NLU | Natural Language Understanding | Text → Meaning | Comprehension |
| NLG | Natural Language Generation | Meaning → Text | Generation |

## Phases / Stages of NLP (Bottom-Up)

```
Raw Text
   ↓
1. Phonological Analysis     → sounds/phonemes
   ↓
2. Morphological Analysis    → word structure (roots, affixes)
   ↓
3. Lexical Analysis          → word meanings, POS tagging
   ↓
4. Syntactic Analysis        → grammar/parse trees
   ↓
5. Semantic Analysis         → literal meaning
   ↓
6. Discourse Analysis        → meaning across sentences
   ↓
7. Pragmatic Analysis        → meaning in context/intent
   ↓
Final Understanding
```

### Each Phase Explained:

**1. Phonological Analysis** — Deals with sounds. Maps phonemes (sound units) to words.

**2. Morphological Analysis** — Studies internal word structure.
- Example: "unhappiness" → un + happy + ness

**3. Lexical Analysis** — Assigns Part-of-Speech tags.
- Example: "run" → verb; "fast" → adjective or adverb

**4. Syntactic Analysis (Parsing)** — Checks grammatical structure using parse trees.
- Example: "The dog bites the man" → NP + VP structure

**5. Semantic Analysis** — Determines the literal meaning.
- Example: "I saw a bat" → is it an animal or cricket bat?

**6. Discourse Analysis** — Resolves references across sentences.
- Example: "Ram went to the store. He bought milk." → He = Ram

**7. Pragmatic Analysis** — Understands intent behind words.
- Example: "Can you pass the salt?" → Not asking about ability, it's a request.

---

# 2. Semantics & Pragmatics

## Semantics
Semantics deals with the **literal meaning** of words, phrases, and sentences.

**Types:**
- **Lexical Semantics** — meaning of individual words
- **Compositional Semantics** — meaning of phrases/sentences built from word meanings

**Example:**
- "The bank was steep." (river bank — geographic)
- "The bank was closed." (financial institution)

Semantics tries to determine which meaning is correct based on context.

## Pragmatics
Pragmatics deals with **meaning in context** — what the speaker actually intends.

It goes beyond the literal meaning of a sentence.

**Examples:**
| Sentence | Literal Meaning | Pragmatic Meaning |
|----------|----------------|-------------------|
| "Can you open the window?" | Are you capable? | Please open it |
| "It's cold in here." | Statement about temperature | Please turn up the heat |
| "Nice haircut!" (sarcastic) | Compliment | Insult |

## Key Difference
- **Semantics** = What does the sentence literally mean?
- **Pragmatics** = What does the speaker mean by saying it?

## Challenges in NLP
- Computers struggle with sarcasm, irony, metaphor
- Context-dependence is hard to model
- World knowledge is needed for pragmatic understanding

---

# 3. Ambiguity in NLP

**Definition:** Ambiguity = when a word, phrase, or sentence can be understood in more than one way.

Ambiguity is the **biggest challenge in NLP**.

## Types of Ambiguity

### 3.1 Lexical Ambiguity
A **single word** has multiple meanings or can be multiple parts of speech.

**Sub-types:**
- **Homonymy** — same word, completely different meanings
  - "Bank" = river bank OR financial institution
  - "Bat" = cricket bat OR flying animal

- **Polysemy** — same word, related meanings
  - "Head" = body part, head of department, head of a coin

- **Part-of-Speech ambiguity** — same word, different grammatical roles
  - "Silver" → noun (silver medal), adjective (silver hair), verb (silvered the mirror)

**Resolution:** POS Tagging, Word Sense Disambiguation (WSD)

---

### 3.2 Syntactic Ambiguity (Structural Ambiguity)
The **grammatical structure** of a sentence is ambiguous.

**Two types:**

**a) Attachment Ambiguity** — unclear what a phrase modifies
> "The man saw the girl with the telescope."
- Reading 1: The man used a telescope to see the girl.
- Reading 2: The girl had a telescope.

> "I ate pizza with a fork and knife."
- With what? Fork+knife are attached to eating or to pizza?

**b) Scope Ambiguity** — unclear how far a modifier/quantifier applies
> "Old men and women were taken to safety."
- Reading 1: (Old men) and (women) → only men are old
- Reading 2: Old (men and women) → both are old

> "Every man loves a woman."
- Reading 1: For every man, there exists some woman he loves
- Reading 2: There is one particular woman that every man loves

**Resolution:** Parsing, Probabilistic Grammar (PCFG)

---

### 3.3 Semantic Ambiguity
Even after syntax is resolved, **meaning** is still ambiguous.

> "Seema loves her mother and Sriya does too."
- Reading 1: Sriya loves Seema's mother
- Reading 2: Sriya loves her own mother

> "The car hit the pole while it was moving."
- It = car (logical) OR pole (grammatically valid)

**Resolution:** Requires world knowledge + context

---

### 3.4 Anaphoric Ambiguity (Referential Ambiguity)
A pronoun or reference could refer to **more than one earlier noun**.

> "The horse ran up the hill. It was very steep. It soon got tired."
- First "It" = hill (steep applies to surfaces)
- Second "It" = horse (tired applies to animate beings)

**Resolution:** Coreference Resolution, Discourse Analysis

---

### 3.5 Pragmatic Ambiguity
The **intended meaning** (speaker's intent) is unclear.

> "Can you close the door?" — Request or genuine question about ability?

**Summary Table:**
| Type | Level | Example | Resolution |
|------|-------|---------|------------|
| Lexical | Word | "bank" | WSD, POS tagging |
| Syntactic | Sentence structure | "I saw man with telescope" | Parsing |
| Semantic | Meaning | "Sriya loves her mother" | World knowledge |
| Anaphoric | Reference | "It was tired" | Coreference resolution |
| Pragmatic | Intent | "Can you pass salt?" | Context modeling |

---

# 4. Morphology & Text Preprocessing

## 4.1 Morphology

**Morphology** = study of the internal structure of words.

**Morpheme** = smallest meaningful unit of language.

**Example:**
- "unhappiness" → **un** (prefix) + **happy** (root) + **ness** (suffix)
- "dogs" → **dog** (root) + **s** (suffix for plural)

### Types of Morphemes:
| Type | Description | Example |
|------|-------------|---------|
| Free morpheme | Can stand alone | "dog", "run" |
| Bound morpheme | Cannot stand alone | "-ing", "un-", "-ness" |
| Root/Stem | Core meaning | "happy" in "unhappy" |
| Affix | Added to root | prefix, suffix, infix |

### Types of Morphology:
**a) Inflectional Morphology** — Changes grammatical form, NOT the word class or core meaning.
- dog → dogs (plural)
- run → ran (past tense)
- tall → taller (comparative)
- write → writing (progressive)

**b) Derivational Morphology** — Creates a new word, often changes word class.
- happy (adj) → happiness (noun)
- teach (verb) → teacher (noun)
- national (adj) → nationalize (verb)

---

## 4.2 Text Preprocessing Steps

### Step 1: Tokenization
Splitting text into smaller units (tokens = words or subwords).

**Input:** "I love NLP!"
**Output:** ["I", "love", "NLP", "!"]

**Challenges:**
- "New York" → 1 token or 2?
- "don't" → "do" + "n't" or keep as is?
- Punctuation handling

**Types:**
- Word tokenization: split by spaces/punctuation
- Sentence tokenization: split into sentences
- Subword tokenization: used in modern models (BPE)

---

### Step 2: Sentence Segmentation
Splitting a paragraph into individual sentences.

**Challenge:** Period (.) used for abbreviations too.
- "Dr. Smith works at U.S.A." — don't split here!
- "He works hard. He is successful." — split here!

---

### Step 3: Stopword Removal
Remove common words that carry little meaning.

**Stopwords:** "is", "the", "a", "and", "of", "in" ...

**Example:**
- Input: "The cat is on the mat"
- Output: ["cat", "mat"]

---

### Step 4: Stemming
Reducing a word to its **root/stem** by chopping off suffixes (may not be a real word).

**Algorithm: Porter Stemmer** (most common)

**Examples:**
| Word | Stem |
|------|------|
| running | run |
| happiness | happi |
| studies | studi |
| troubling | troubl |
| fishing | fish |

**Note:** Stems may not be real dictionary words ("happi").

---

### Step 5: Lemmatization
Reducing a word to its **base/dictionary form** (lemma) using vocabulary and morphological analysis.

**Examples:**
| Word | Lemma |
|------|-------|
| running | run |
| better | good |
| studies | study |
| went | go |
| am/is/are | be |

### Stemming vs Lemmatization:
| Feature | Stemming | Lemmatization |
|---------|----------|---------------|
| Speed | Fast | Slower |
| Accuracy | Lower | Higher |
| Result | May not be real word | Always real word |
| Method | Rule-based chopping | Dictionary lookup |
| Example | "studies" → "studi" | "studies" → "study" |

---

### Step 6: Normalization
Converting text to a standard form.
- Lowercase: "NLP" → "nlp"
- Expand contractions: "don't" → "do not"
- Remove special characters: "@#$%"
- Number normalization: "100" → "one hundred"

---

# 5. Corpus & Corpus Linguistics

## What is a Corpus?
A **corpus** (plural: corpora) is a large, structured collection of text used for linguistic analysis and NLP training.

**Examples:**
- Brown Corpus (1 million words of American English)
- BNC — British National Corpus
- Wikipedia dumps

## Corpus Design Principles
When building a corpus, consider:
1. **Representativeness** — covers the language variety needed
2. **Balance** — different genres/domains
3. **Size** — large enough for statistical significance
4. **Sampling** — random, stratified
5. **Annotation** — POS tags, parse trees, named entities

## Types of Corpora:
| Type | Description |
|------|-------------|
| Monolingual | One language |
| Bilingual / Parallel | Two languages (for translation) |
| Annotated | With POS tags, parse trees |
| Raw | Unannotated plain text |
| Specialized | Domain-specific (medical, legal) |

## Corpus Annotation
Adding linguistic information to raw text:
- **POS Annotation:** "The/DT dog/NN runs/VBZ"
- **Parse trees:** syntactic structure
- **Named Entity tags:** Person, Location, Organization
- **Semantic roles:** who did what to whom

## Collocations
**Collocation** = words that frequently occur together and whose meaning together differs from individual words.

**Examples:**
- "strong tea" (NOT "powerful tea")
- "make a decision" (NOT "do a decision")
- "heavy rain" (NOT "strong rain")

### Measuring Collocations:

**1. Frequency Count** — just count how often words appear together (simple but noisy).

**2. Pointwise Mutual Information (PMI)**

```
PMI(x, y) = log₂ [ P(x,y) / (P(x) × P(y)) ]
```

- High PMI → words appear together more than by chance
- PMI = 0 → independent
- Negative PMI → appear together less than by chance

**Example:**
- P("strong") = 0.01, P("tea") = 0.005, P("strong tea") = 0.002
- PMI = log₂(0.002 / (0.01 × 0.005)) = log₂(0.002/0.00005) = log₂(40) ≈ 5.32

High PMI → "strong tea" is a genuine collocation.

**3. t-test** — statistical test for significance of co-occurrence.

---

# 6. Edit Distance — Levenshtein Algorithm

## What is Edit Distance?
The **minimum number of single-character edits** (insertions, deletions, substitutions) needed to transform one string into another.

**Operations:**
| Operation | Cost | Example |
|-----------|------|---------|
| Insertion | 1 | "" → "a" |
| Deletion | 1 | "a" → "" |
| Substitution | 1 | "a" → "b" |

**Applications:**
- Spell checking ("teh" → "the")
- DNA sequence alignment
- Plagiarism detection
- Machine translation evaluation

## Algorithm — Dynamic Programming

**Formula:**

```
If s1[i] == s2[j]:
    dp[i][j] = dp[i-1][j-1]         (no cost, characters match)
Else:
    dp[i][j] = 1 + min(
        dp[i-1][j],     → deletion
        dp[i][j-1],     → insertion
        dp[i-1][j-1]    → substitution
    )
```

**Base Cases:**
- dp[i][0] = i (delete all characters of s1)
- dp[0][j] = j (insert all characters of s2)

---

## Numerical Example 1: "kitten" → "sitting"

|   |   | s | i | t | t | i | n | g |
|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| k | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| i | 2 | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| t | 3 | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| t | 4 | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| e | 5 | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| n | 6 | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

**Edit Distance = 3**

Operations:
1. k → s (substitution)
2. e → i (substitution)
3. insert g at end

---

## Numerical Example 2: "SUNDAY" → "SATURDAY"

|   |   | S | A | T | U | R | D | A | Y |
|---|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| S | 1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| U | 2 | 1 | 1 | 2 | 2 | 3 | 4 | 5 | 6 |
| N | 3 | 2 | 2 | 2 | 3 | 3 | 4 | 5 | 6 |
| D | 4 | 3 | 3 | 3 | 3 | 4 | 3 | 4 | 5 |
| A | 5 | 4 | 3 | 4 | 4 | 4 | 4 | 3 | 4 |
| Y | 6 | 5 | 4 | 4 | 5 | 5 | 5 | 4 | 3 |

**Edit Distance = 3**

---

# 7. N-gram Language Models

## What is a Language Model?
A **language model** assigns a probability to a sequence of words.

**Goal:** P(w₁ w₂ w₃ ... wₙ) = ?

**Why?** Useful for:
- Speech recognition
- Machine translation
- Spelling correction
- Text generation

## Chain Rule of Probability

```
P(w₁w₂...wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁w₂) × ... × P(wₙ|w₁...wₙ₋₁)
```

**Problem:** As n grows, we never see the full history in training data.

## Markov Assumption
Approximate: only look at last N-1 words (not the full history).

```
P(wₙ | w₁...wₙ₋₁) ≈ P(wₙ | wₙ₋ₙ₊₁...wₙ₋₁)
```

## Types of N-gram Models

| Model | N | Looks at | Formula |
|-------|---|----------|---------|
| Unigram | 1 | No context | P(w) |
| Bigram | 2 | Previous 1 word | P(w₂\|w₁) |
| Trigram | 3 | Previous 2 words | P(w₃\|w₁w₂) |

## MLE (Maximum Likelihood Estimation) Formula

**Bigram:**
```
P(wₙ | wₙ₋₁) = Count(wₙ₋₁, wₙ) / Count(wₙ₋₁)
```

**Unigram:**
```
P(w) = Count(w) / Total words in corpus
```

## Sentence Probability (Bigram)
Add `<s>` (start) and `</s>` (end) markers:

```
P(<s> w₁ w₂ ... wₙ </s>) = P(w₁|<s>) × P(w₂|w₁) × ... × P(</s>|wₙ)
```

---

## ★ FULL NUMERICAL EXAMPLE (from your notes)

**Corpus:**
```
"I love NLP"
"I love AI"
"I study NLP"
```

### Step 1: Add markers and extract bigrams

```
<s> I love NLP </s>
<s> I love AI </s>
<s> I study NLP </s>
```

**Bigram Counts:**
| Bigram | Count |
|--------|-------|
| (<s>, I) | 3 |
| (I, love) | 2 |
| (I, study) | 1 |
| (love, NLP) | 1 |
| (love, AI) | 1 |
| (study, NLP) | 1 |
| (NLP, </s>) | 2 |
| (AI, </s>) | 1 |

**Unigram Counts:**
| Word | Count |
|------|-------|
| <s> | 3 |
| I | 3 |
| love | 2 |
| study | 1 |
| NLP | 2 |
| AI | 1 |
| </s> | 3 |

### Step 2: Calculate Bigram Probabilities

```
P(I | <s>)    = C(<s>, I)    / C(<s>)    = 3/3 = 1.0
P(love | I)   = C(I, love)   / C(I)      = 2/3 ≈ 0.67
P(study | I)  = C(I, study)  / C(I)      = 1/3 ≈ 0.33
P(NLP | love) = C(love, NLP) / C(love)   = 1/2 = 0.5
P(AI | love)  = C(love, AI)  / C(love)   = 1/2 = 0.5
P(NLP | study)= C(study,NLP) / C(study)  = 1/1 = 1.0
P(</s> | NLP) = C(NLP, </s>) / C(NLP)    = 2/2 = 1.0
P(</s> | AI)  = C(AI, </s>)  / C(AI)     = 1/1 = 1.0
```

### Step 3: Sentence Probability — "I love NLP"

```
P(<s> I love NLP </s>)
= P(I|<s>) × P(love|I) × P(NLP|love) × P(</s>|NLP)
= 1.0 × (2/3) × (1/2) × 1.0
= 1.0 × 0.67 × 0.5 × 1.0
= 0.335
```

**Answer: P("I love NLP") ≈ 0.335**

### Step 4: Sentence Probability — "I study NLP"

```
P(<s> I study NLP </s>)
= P(I|<s>) × P(study|I) × P(NLP|study) × P(</s>|NLP)
= 1.0 × (1/3) × 1.0 × 1.0
= 0.333
```

---

# 8. Smoothing Techniques

## Why Smoothing?
N-gram models assign **zero probability** to unseen word sequences.

**Problem:** P(w | context) = 0 if (context, w) was never seen in training.

Zero probabilities crash entire sentence probability (multiplying any number by 0 = 0).

**Solution:** Smoothing = steal small probability from seen events and give to unseen ones.

---

## 8.1 Add-One Smoothing (Laplace Smoothing)

**Idea:** Add 1 to every count (even unseen bigrams).

**Formula:**

```
P_Laplace(wₙ | wₙ₋₁) = (C(wₙ₋₁, wₙ) + 1) / (C(wₙ₋₁) + V)
```

Where **V = vocabulary size** (number of unique words).

**Example:**
- Corpus: "I love NLP", "I love AI", "I study NLP"
- V = 7 (I, love, NLP, AI, study, <s>, </s>)
- C(I, love) = 2, C(I) = 3

```
P_Laplace(love | I) = (2 + 1) / (3 + 7) = 3/10 = 0.3
```

Compare to MLE: P(love | I) = 2/3 = 0.667

**Limitation:** Over-smooths — gives too much probability mass to unseen events, reduces probabilities of frequent events significantly.

---

## 8.2 Good-Turing Smoothing

**Idea:** Use the count of things you've seen **once** to estimate the count of things you've **never seen**.

**Key Formula:**

```
c* = (c + 1) × N(c+1) / N(c)
```

Where:
- **c** = original count of a bigram
- **c*** = adjusted (smoothed) count
- **N(c)** = number of bigrams that appear exactly c times

### Intuition:
- Things seen once help estimate things never seen.
- "If I saw 10 new birds last week, maybe I'll see a few more I've never seen before."

### Full Example:

Suppose we count bigrams and find:

| Count (c) | Number of bigrams with that count N(c) |
|-----------|----------------------------------------|
| 0 | unknown (these are unseen) |
| 1 | 10 |
| 2 | 5 |
| 3 | 4 |
| 4 | 2 |

**For unseen bigrams (c=0):**
```
c* = (0+1) × N(1) / N(0)
```
We estimate N(0) as total possible bigrams − seen bigrams.

**For bigrams seen once (c=1):**
```
c* = (1+1) × N(2) / N(1) = 2 × 5 / 10 = 1.0
```

**For bigrams seen twice (c=2):**
```
c* = (2+1) × N(3) / N(2) = 3 × 4 / 5 = 2.4
```

**For bigrams seen 3 times (c=3):**
```
c* = (3+1) × N(4) / N(3) = 4 × 2 / 4 = 2.0
```

**Interpretation:** Adjust down all counts. This frees up probability for unseen events.

**Probability of unseen bigram:**
```
P*(unseen) = N(1) / N_total
```
Where N_total = total number of bigram tokens.

---

## 8.3 Comparison of Smoothing Methods

| Method | Formula | Advantage | Disadvantage |
|--------|---------|-----------|--------------|
| No smoothing (MLE) | C(w,c)/C(c) | Simple | Zero probs |
| Add-one (Laplace) | (C+1)/(N+V) | Simple | Over-smooths |
| Good-Turing | c* = (c+1)N(c+1)/N(c) | More accurate | Complex |
| Kneser-Ney | Advanced backoff | Best performance | Most complex |

---

# 9. Entropy, Cross-Entropy & Perplexity

## 9.1 Entropy

**Entropy** = measure of uncertainty/randomness in a probability distribution.

**Formula:**
```
H(X) = -∑ P(xᵢ) × log₂ P(xᵢ)
```

- High entropy = high uncertainty (distribution is flat/uniform)
- Low entropy = low uncertainty (one outcome dominates)

**Units:** bits (when using log₂)

**Example:**
Fair coin: P(H) = P(T) = 0.5
```
H = -(0.5 × log₂0.5 + 0.5 × log₂0.5)
  = -(0.5 × (-1) + 0.5 × (-1))
  = -(-0.5 - 0.5)
  = 1 bit
```

Biased coin: P(H) = 0.9, P(T) = 0.1
```
H = -(0.9 × log₂0.9 + 0.1 × log₂0.1)
  = -(0.9 × (-0.152) + 0.1 × (-3.322))
  = -(−0.137 − 0.332)
  = 0.469 bits
```

Less uncertainty → lower entropy.

## 9.2 Cross-Entropy

**Cross-Entropy** = measures how well a model (Q) approximates the true distribution (P).

**Formula:**
```
H(P, Q) = -∑ P(x) × log₂ Q(x)
```

- P = true distribution
- Q = model's predicted distribution
- Cross-entropy ≥ Entropy always

**Interpretation:**
- If Q = P exactly: H(P,Q) = H(P) (best possible)
- If Q is wrong: H(P,Q) > H(P)

**In Language Models:**
```
H(P, Q) = -(1/N) × ∑ log₂ Q(wᵢ | wᵢ₋₁)
```
Where N = number of words in test corpus.

A lower cross-entropy means the model better predicts the test data.

## 9.3 Perplexity

**Perplexity** = the most common way to evaluate language models.

**Intuition:** If a model has perplexity K, it's as "confused" as if it were randomly choosing among K equally likely words at each step.

**Formula:**
```
PP(W) = 2^H(P,Q)
```

Or equivalently for a test sequence W = w₁w₂...wₙ:
```
PP(W) = P(w₁w₂...wₙ)^(-1/N)
```

Or using bigrams:
```
PP(W) = [∏ 1/P(wᵢ|wᵢ₋₁)]^(1/N)
```

**Lower perplexity = better model.**

### Numerical Example:

Test sentence: "I love NLP" (3 words)

From our corpus:
```
P(I | <s>) = 1.0
P(love | I) = 2/3
P(NLP | love) = 1/2
P(</s> | NLP) = 1.0
```

Probability of sentence (including </s>):
```
P = 1.0 × 0.667 × 0.5 × 1.0 = 0.335
```

N = 4 (counting </s>)

```
PP = (0.335)^(-1/4) = (1/0.335)^(1/4) = (2.985)^(0.25) ≈ 1.314
```

**Relation:** PP = 2^H
- Low perplexity ≈ Low cross-entropy ≈ Good model

### Typical Perplexity Values:
| Model Type | Perplexity |
|------------|-----------|
| Random (V=10,000) | 10,000 |
| Unigram | ~1000 |
| Bigram | ~200 |
| Trigram | ~100 |
| LSTM/Transformer | ~20-50 |

---

# 10. Bag of Words (BoW) & TF-IDF

## 10.1 Bag of Words (BoW)

**Idea:** Represent a document as a vector of word counts, ignoring order.

**Steps:**
1. Build vocabulary from all documents
2. For each document, count occurrences of each vocabulary word
3. Represent document as a count vector

**Example:**

Documents:
- Doc1: "I love NLP"
- Doc2: "I love AI"
- Doc3: "NLP is amazing"

Vocabulary: [I, love, NLP, AI, is, amazing] (6 words)

| Document | I | love | NLP | AI | is | amazing |
|----------|---|------|-----|----|----|---------|
| Doc1 | 1 | 1 | 1 | 0 | 0 | 0 |
| Doc2 | 1 | 1 | 0 | 1 | 0 | 0 |
| Doc3 | 0 | 0 | 1 | 0 | 1 | 1 |

**Limitations:**
- Ignores word order ("dog bites man" = "man bites dog")
- High dimensionality
- Doesn't capture word importance

---

## 10.2 TF-IDF (Term Frequency — Inverse Document Frequency)

**Idea:** Give higher weight to words that are **frequent in a document** but **rare across all documents**.

### Term Frequency (TF):
```
TF(t, d) = count of term t in document d / total terms in document d
```

### Inverse Document Frequency (IDF):
```
IDF(t) = log(N / df(t))
```
Where:
- N = total number of documents
- df(t) = number of documents containing term t

### TF-IDF Score:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Intuition:**
- Common words like "the", "is" → high TF but low IDF → low TF-IDF
- Important rare words → high TF × high IDF → high TF-IDF

---

### ★ Full Numerical Example:

**Documents:**
- Doc1: "the cat sat on the mat"
- Doc2: "the cat sat"
- Doc3: "the dog barked"

N = 3 documents

**Step 1: Calculate TF**

For "cat" in Doc1:
```
TF(cat, Doc1) = 1/6 = 0.167
```

For "cat" in Doc2:
```
TF(cat, Doc2) = 1/3 = 0.333
```

**Step 2: Calculate IDF**

"cat" appears in Doc1 and Doc2 → df("cat") = 2
```
IDF(cat) = log(3/2) = log(1.5) = 0.176
```

"the" appears in all 3 docs → df("the") = 3
```
IDF(the) = log(3/3) = log(1) = 0
```
→ "the" has zero importance (appears everywhere)

"dog" appears only in Doc3 → df("dog") = 1
```
IDF(dog) = log(3/1) = log(3) = 0.477
```

**Step 3: Calculate TF-IDF**

```
TF-IDF(cat, Doc1) = 0.167 × 0.176 = 0.029
TF-IDF(cat, Doc2) = 0.333 × 0.176 = 0.059
TF-IDF(dog, Doc3) = (1/3) × 0.477 = 0.159
TF-IDF(the, Doc1) = 0.333 × 0 = 0
```

**Conclusion:** "dog" is the most distinctive word in Doc3. "the" is meaningless.

---

# 11. Parts of Speech (POS) Tagging

## What is POS Tagging?
Assigning a grammatical tag (noun, verb, adjective, etc.) to each word in a sentence.

**Example:**
```
"The    dog    runs    fast"
 DT     NN     VBZ     RB
```

## Common POS Tags (Penn Treebank Tagset):
| Tag | Meaning | Example |
|-----|---------|---------|
| NN | Noun, singular | dog, car |
| NNS | Noun, plural | dogs, cars |
| NNP | Proper noun | London, John |
| VB | Verb, base form | run, eat |
| VBZ | Verb, 3rd person singular | runs, eats |
| VBD | Verb, past tense | ran, ate |
| VBG | Verb, gerund | running |
| JJ | Adjective | big, fast |
| RB | Adverb | quickly, very |
| DT | Determiner | the, a |
| IN | Preposition | in, on, at |
| CC | Coordinating conjunction | and, but, or |
| PRP | Personal pronoun | I, he, she |

## Why is POS Tagging Hard?
Many words are ambiguous:
- "book" → noun (a book) OR verb (book a ticket)
- "can" → noun (tin can), verb (can do), modal (can run)
- "fast" → adjective (fast car), adverb (runs fast), verb (to fast = not eat)

## Approaches to POS Tagging:

### 1. Rule-Based
- Hand-written linguistic rules
- "If word ends in '-ing' and preceded by 'is', tag as VBG"
- Accurate but requires expert knowledge

### 2. Statistical (HMM-based)
- Uses probability: P(tag | word) and P(tag | previous tag)
- Viterbi algorithm to find best tag sequence
- See Section 13 for HMM details

### 3. Deep Learning
- RNN/LSTM/Transformer-based taggers
- Best accuracy currently

## Phrase Structure
Sentences are made of **phrases**:

| Phrase | Abbreviation | Example |
|--------|-------------|---------|
| Noun Phrase | NP | "the big dog" |
| Verb Phrase | VP | "runs quickly" |
| Prepositional Phrase | PP | "in the park" |
| Adjective Phrase | AdjP | "very happy" |
| Adverb Phrase | AdvP | "quite quickly" |

**Sentence structure:**
```
S → NP VP
NP → DT JJ NN
VP → VB NP | VB PP
PP → IN NP
```

---

# 12. Word Sense Disambiguation (WSD)

## What is WSD?
The task of determining which **sense** (meaning) of a word is used in a given context.

**Example:**
> "I went to the bank."
- Sense 1: river bank
- Sense 2: financial institution

WSD needs to decide based on context.

**Applications:** Machine Translation, Information Retrieval, Question Answering

## Types of WSD Approaches:

### A. Knowledge-Based (Dictionary-Based)
Uses dictionaries/thesaurus (like WordNet) — no training data needed.

### B. Supervised
Uses labeled training data (word + correct sense).

### C. Unsupervised
Clusters word usages without labels.

---

## 12.1 Lesk Algorithm (Dictionary-Based)

**Idea:** Find the sense whose dictionary definition has the **maximum overlap** with the context words.

**Algorithm:**
```
For each sense s of the target word:
    overlap = |definition(s) ∩ context_words|
Return sense with maximum overlap
```

### Numerical Example:

**Target word:** "bank"
**Context:** "I deposited money in the bank near the river"
**Context words:** {deposited, money, bank, near, river}

**WordNet definitions:**
- Sense 1 (financial): "a financial institution that accepts deposits and makes loans"
  - Def words: {financial, institution, accepts, deposits, makes, loans}
- Sense 2 (river): "sloping land beside a body of water"
  - Def words: {sloping, land, beside, body, water}

**Overlap Calculation:**
- Sense 1 ∩ Context = {deposits} → overlap = 1
- Sense 2 ∩ Context = {river} (if "river" ≈ "water") → overlap = 1

If we consider "money" → deposits are financial:
- Sense 1 wins → "bank" = financial institution ✓

---

## 12.2 Advanced Lesk Algorithm

**Problem with Basic Lesk:** Only compares target word definition with context.

**Advanced Lesk:** Also considers definitions of **context words** (neighbors).

**Enhancement:**
```
overlap(sense_s, context) = |def(s) ∩ (context ∪ ⋃ def(context_words))|
```

Also compares definitions of neighboring words against each other — more information utilized.

**Other enhancements:**
- Weight overlaps (longer common phrases get higher weight)
- Use WordNet hypernyms/hyponyms for broader matching
- tf-idf weighting of definition words

---

## 12.3 Supervised WSD

**Approach:** Train a classifier using labeled examples.

**Features used:**
- Surrounding words (window of ±2-3 words)
- POS tags of surrounding words
- Collocations
- Syntactic relations

**Algorithms:** Naive Bayes, SVM, Neural Networks

**Advantage:** High accuracy
**Disadvantage:** Needs large annotated corpus (knowledge acquisition bottleneck)

---

## 12.4 Unsupervised WSD

**Approach:** Cluster occurrences of a word by their context (without labels).

**Assumption:** Occurrences in the same cluster = same sense.

**Algorithm: Word Sense Induction**
1. Collect all sentences containing target word
2. Represent each by context vector
3. Cluster contexts (K-means, etc.)
4. Each cluster = one sense

**Advantage:** No labeled data needed
**Disadvantage:** Clusters may not align with dictionary senses

---

## 12.5 WSD Approaches Summary

| Approach | Data Needed | Accuracy | Key Algorithm |
|----------|-------------|----------|---------------|
| Dictionary-based | Dictionary | Medium | Lesk |
| Supervised | Labeled corpus | High | SVM, NN |
| Unsupervised | Unlabeled corpus | Lower | Clustering |

---

# 13. Hidden Markov Model (HMM) & Viterbi

## What is HMM?

A **Hidden Markov Model** is a statistical model where:
- The system is in one of several **hidden states** (not directly observable)
- Each state emits an **observable output** with some probability
- States transition to other states with **transition probabilities**

**Used in NLP for:** POS tagging, Speech Recognition, Named Entity Recognition

## HMM Components (λ = (A, B, π)):

| Symbol | Name | Description |
|--------|------|-------------|
| Q = {q₁,...,qN} | States | Hidden states (e.g., POS tags) |
| O = {o₁,...,oT} | Observations | Observable outputs (e.g., words) |
| A | Transition Matrix | P(qⱼ\|qᵢ) — prob of going from state i to state j |
| B | Emission Matrix | P(oₖ\|qᵢ) — prob of emitting word k from state i |
| π | Initial Probabilities | P(q₁) — prob of starting in each state |

## Three Problems in HMM:

| Problem | Question | Algorithm |
|---------|----------|-----------|
| Evaluation | What is P(O\|λ)? | Forward Algorithm |
| Decoding | What is best state sequence? | Viterbi Algorithm |
| Learning | How to learn A, B, π? | Baum-Welch |

---

## Viterbi Algorithm (Decoding)

**Goal:** Find the most likely sequence of hidden states given observations.

**Formula:**
```
Initialization (t=1):
    v₁(j) = πⱼ × bⱼ(o₁)

Recursion (t > 1):
    vₜ(j) = max_i [vₜ₋₁(i) × aᵢⱼ] × bⱼ(oₜ)

Termination:
    Best final state = argmax_j [vₜ(j)]
    Backtrack to find full sequence
```

Where:
- vₜ(j) = probability of best path ending in state j at time t
- πⱼ = initial probability of state j
- aᵢⱼ = transition P(j|i)
- bⱼ(oₜ) = emission P(oₜ|j)

---

## ★ Numerical Example 1: Weather HMM (from your notes)

**States:** Sunny (S), Cloudy (C), Rainy (R)
**Observations:** Happy (H), Sad (Sa)

**Transition Matrix A:**
```
        Sunny  Cloudy  Rainy
Sunny  [0.33   0.67    0  ]
Rainy  [ 0     0.33   0.67]
Cloudy [0.67    0     0.33]
```

**Emission Matrix B:**
```
        Happy  Sad
Sunny  [0.5    0.5 ]
Cloudy [0.67   0.33]
Rainy  [0.33   0.67]
```

**Initial Probabilities π:**
```
P(Sunny) = 0.4, P(Cloudy) = 0.3, P(Rainy) = 0.3
```

**Observation Sequence:** [Happy, Sad]

---

### Step 1: Initialization (t=1, O₁ = Happy)

```
v₁(Sunny) = π(Sunny) × P(Happy|Sunny) = 0.4 × 0.5 = 0.200
v₁(Rainy) = π(Rainy) × P(Happy|Rainy) = 0.3 × 0.33 = 0.099
v₁(Cloudy) = π(Cloudy) × P(Happy|Cloudy) = 0.3 × 0.67 = 0.201
```

### Step 2: Recursion (t=2, O₂ = Sad)

**For state Sunny at t=2:**
```
v₂(Sunny) = max[
    v₁(Sunny) × P(Sunny→Sunny),
    v₁(Rainy) × P(Rainy→Sunny),
    v₁(Cloudy) × P(Cloudy→Sunny)
] × P(Sad|Sunny)

= max[0.200×0.33, 0.099×0, 0.201×0.67] × 0.5
= max[0.066, 0, 0.13467] × 0.5
= 0.13467 × 0.5
= 0.067335    ← came from Cloudy
```

**For state Rainy at t=2:**
```
v₂(Rainy) = max[
    v₁(Rainy) × P(Rainy→Rainy),
    v₁(Sunny) × P(Sunny→Rainy),
    v₁(Cloudy) × P(Cloudy→Rainy)
] × P(Sad|Rainy)

= max[0.099×0.33, 0.200×0.67, 0.201×0] × 0.67
= max[0.03267, 0.134, 0] × 0.67
= 0.134 × 0.67
= 0.08978    ← came from Sunny
```

**For state Cloudy at t=2:**
```
v₂(Cloudy) = max[
    v₁(Cloudy) × P(Cloudy→Cloudy),
    v₁(Sunny) × P(Sunny→Cloudy),
    v₁(Rainy) × P(Rainy→Cloudy)
] × P(Sad|Cloudy)

= max[0.201×0.33, 0.200×0, 0.099×0.67] × 0.33
= max[0.06633, 0, 0.06603] × 0.33
= 0.06633 × 0.33
= 0.021889    ← came from Cloudy
```

### Step 3: Termination

```
max[v₂(Sunny), v₂(Rainy), v₂(Cloudy)]
= max[0.067335, 0.08978, 0.021889]
= 0.08978 → State = Rainy (t=2)
```

**Backtrack t=1:** The Rainy state at t=2 came from Sunny.
- v₁ max among all states: max[0.200, 0.099, 0.201] = 0.201 → Cloudy (t=1)

**Most Likely State Sequence:**
```
t=1: Cloudy (Observation: Happy)
t=2: Rainy  (Observation: Sad)
```

**Path:** Cloudy → Rainy

---

## ★ Numerical Example 2: POS Tagging with Viterbi

**HMM for POS Tagging:**
- States: Noun (N), Verb (V)
- Observations: "fish", "sleep"

**Given:**
```
Initial: P(N) = 0.6, P(V) = 0.4

Transition:
P(N→N) = 0.3, P(N→V) = 0.7
P(V→N) = 0.8, P(V→V) = 0.2

Emission:
P(fish|N) = 0.6, P(fish|V) = 0.4
P(sleep|N) = 0.4, P(sleep|V) = 0.6
```

**Observation:** ["fish", "sleep"]

### Step 1: Initialization (t=1, word = "fish")

```
v₁(N) = P(N) × P(fish|N) = 0.6 × 0.6 = 0.36
v₁(V) = P(V) × P(fish|V) = 0.4 × 0.4 = 0.16
```

### Step 2: Recursion (t=2, word = "sleep")

**For Noun at t=2:**
```
v₂(N) = max[v₁(N)×P(N→N), v₁(V)×P(V→N)] × P(sleep|N)
       = max[0.36×0.3, 0.16×0.8] × 0.4
       = max[0.108, 0.128] × 0.4
       = 0.128 × 0.4           ← came from V
       = 0.0512
```

**For Verb at t=2:**
```
v₂(V) = max[v₁(N)×P(N→V), v₁(V)×P(V→V)] × P(sleep|V)
       = max[0.36×0.7, 0.16×0.2] × 0.6
       = max[0.252, 0.032] × 0.6
       = 0.252 × 0.6           ← came from N
       = 0.1512
```

### Step 3: Termination

```
max[v₂(N), v₂(V)] = max[0.0512, 0.1512] = 0.1512 → Verb (t=2)
```

**Backtrack:** V at t=2 came from N at t=1.

**Most Likely POS sequence:**
```
fish → Noun
sleep → Verb
```

**Answer: [Noun, Verb]**

---

# 14. Parsing — CFG, CNF, CYK, PCFG

## What is Parsing?
Parsing = analyzing a sentence to determine its grammatical structure (syntax tree).

**Types of Parsing:**
| Type | Direction | Method |
|------|-----------|--------|
| Top-Down | Root → leaves | Start with S, expand rules |
| Bottom-Up | Leaves → root | Start with words, reduce to S |
| Chart Parsing | Both | CYK Algorithm |

---

## 14.1 Context-Free Grammar (CFG)

A CFG has 4 components: **G = (N, Σ, R, S)**

| Symbol | Name | Meaning |
|--------|------|---------|
| N | Non-terminals | Syntactic categories: S, NP, VP, ... |
| Σ | Terminals | Actual words: dog, runs, the, ... |
| R | Rules | Production rules: NP → DT NN |
| S | Start symbol | Usually S (sentence) |

**Example Grammar:**
```
S  → NP VP
NP → DT NN | DT JJ NN | NNP
VP → VB NP | VB PP | VBZ
PP → IN NP
DT → "the" | "a"
NN → "dog" | "cat" | "ball"
NNP → "John" | "London"
VB → "chases" | "sees"
VBZ → "runs"
IN → "in" | "on"
JJ → "big" | "small"
```

**Parsing Example:** "The dog chases a cat"

```
         S
        / \
      NP   VP
     / \   / \
   DT  NN VB  NP
   |   |  |  / \
  the dog chases DT NN
                 |  |
                 a  cat
```

---

## 14.2 CFG to Chomsky Normal Form (CNF)

**CNF Rules:** Every production must be EITHER:
- **A → B C** (exactly two non-terminals), OR
- **A → a** (exactly one terminal)

**Why CNF?** Required for CYK algorithm.

### Steps to Convert CFG to CNF:

**Step 1: Add new start symbol**
```
S' → S
```
(Only if S appears on right side of any rule)

**Step 2: Remove ε-productions (A → ε)**
- Find all nullable symbols (those that can derive ε)
- Remove ε rules, add alternatives without nullable symbols
- Example: If A → ε and B → A C, add B → C

**Step 3: Remove unit productions (A → B)**
- A → B, B → CD becomes A → CD directly
- Remove the chain rule

**Step 4: Fix long productions (A → B C D...)**
- Break into binary: A → B X, X → C D

**Step 5: Fix mixed productions (A → a B)**
- Create new non-terminal: Xₐ → a, then A → Xₐ B

---

### Numerical Example: Convert to CNF

**Original Grammar:**
```
S → ABC
S → a
A → aA | a
B → bB | b
C → cC | c
```

**Step 1: Break long production S → ABC**
```
S → AX
X → BC
```

**Step 2: Fix terminal in mixed productions**
```
A → YₐA | a   where Yₐ → a
B → YᵦB | b   where Yᵦ → b
C → YcC | c   where Yc → c
```

**Final CNF:**
```
S → AX | a
X → BC
A → YₐA | a
B → YᵦB | b
C → YcC | c
Yₐ → a
Yᵦ → b
Yc → c
```

---

## 14.3 CYK Algorithm (Cocke-Younger-Kasami)

**Purpose:** Parses a sentence using a CFG in CNF. Checks if sentence is valid and finds parse tree.

**Type:** Bottom-up, dynamic programming.

**Complexity:** O(n³ × |G|) where n = sentence length, |G| = number of rules.

### CYK Table:
- Table[i][j] = set of non-terminals that can derive the substring from position i to j.

**Formula:**
```
Table[i][j] = {A | A → B C ∈ Grammar,
               B ∈ Table[i][k], C ∈ Table[k+1][j],
               for some k: i ≤ k < j}
```

**Base case (single words):**
```
Table[i][i] = {A | A → wᵢ ∈ Grammar}
```

---

### ★ Numerical Example: CYK Parsing

**Grammar (CNF):**
```
S → NP VP
NP → Det N
VP → V NP | V
Det → "the"
N → "cat" | "dog"
V → "chased"
```

**Sentence:** "the cat chased the dog" (5 words)

```
w₁=the, w₂=cat, w₃=chased, w₄=the, w₅=dog
```

**Step 1: Fill diagonal (single words)**
```
Table[1][1] = {Det}    ← "the" → Det
Table[2][2] = {N}      ← "cat" → N
Table[3][3] = {V}      ← "chased" → V
Table[4][4] = {Det}    ← "the" → Det
Table[5][5] = {N}      ← "dog" → N
```

**Step 2: Spans of length 2**
```
Table[1][2]: Can Det N → anything? NP → Det N ✓
             Table[1][2] = {NP}

Table[2][3]: Can N V → anything? No rule.
             Table[2][3] = {}

Table[3][4]: Can V Det → anything? No rule.
             Table[3][4] = {}

Table[4][5]: Can Det N → anything? NP → Det N ✓
             Table[4][5] = {NP}
```

**Step 3: Spans of length 3**
```
Table[1][3]: 
  k=1: Table[1][1]=Det, Table[2][3]={} → nothing
  k=2: Table[1][2]=NP, Table[3][3]=V → no rule NP V
  Table[1][3] = {}

Table[2][4]:
  k=2: Table[2][2]=N, Table[3][4]={} → nothing
  k=3: Table[2][3]={}, Table[4][4]=Det → nothing
  Table[2][4] = {}

Table[3][5]:
  k=3: Table[3][3]=V, Table[4][5]=NP → VP → V NP ✓
  Table[3][5] = {VP}
```

**Step 4: Spans of length 4**
```
Table[1][4] = {} (no useful combinations)
Table[2][5]:
  k=2: Table[2][2]=N, Table[3][5]=VP → no rule
  k=3: Table[2][4]={}, ...
  Table[2][5] = {}
```

**Step 5: Full sentence span**
```
Table[1][5]:
  k=1: Table[1][1]=Det, Table[2][5]={} → nothing
  k=2: Table[1][2]=NP, Table[3][5]=VP → S → NP VP ✓
  k=3: Table[1][3]={}, ...
  Table[1][5] = {S}
```

**S ∈ Table[1][5] → Sentence is VALID! ✓**

**Parse Tree:**
```
        S
       / \
     NP   VP
    / \   / \
  Det  N  V  NP
  |    |  |  / \
 the  cat |  Det N
      chased |   |
            the dog
```

---

## 14.4 PCFG — Probabilistic Context-Free Grammar

**Idea:** Attach probabilities to grammar rules.

**Constraint:** Probabilities of all rules for a non-terminal must sum to 1.

**Example:**
```
S → NP VP     [1.0]
NP → Det N    [0.6]
NP → NNP      [0.4]
VP → V NP     [0.7]
VP → V        [0.3]
```

**Probability of a parse tree:**
```
P(tree) = ∏ P(rule used at each node)
```

**Example:**
```
P(S→NP VP) × P(NP→Det N) × P(VP→V NP) × P(NP→Det N)
= 1.0 × 0.6 × 0.7 × 0.6
= 0.252
```

**Why PCFG?** When a sentence has multiple parse trees (ambiguity), PCFG picks the most probable one.

---

## 14.5 Top-Down vs Bottom-Up Parsing

**Top-Down Parsing:**
- Starts with goal symbol S
- Expands rules until reaching terminals
- Matches words left to right
- Problem: May try wrong rules many times

**Bottom-Up Parsing:**
- Starts with words (terminals)
- Reduces to non-terminals using rules
- Builds up to S
- Better for ambiguous grammars

---

# 15. Deep Learning in NLP — CNN, RNN, LSTM

## 15.1 CNN for NLP (Convolutional Neural Network)

Originally for images, but adapted for text.

**How it works for NLP:**
1. Represent words as vectors (word embeddings)
2. Apply convolution filters (sliding windows) over word sequences
3. Capture local n-gram features
4. Max-pooling to get most important features
5. Classify

**Use cases:** Text classification, Sentiment analysis

**Example:**
- Filter of size 2 → captures bigram features
- Filter of size 3 → captures trigram features

**Advantage:** Fast, parallel computation
**Disadvantage:** Fixed window size, doesn't capture long-range dependencies

---

## 15.2 RNN — Recurrent Neural Network

**Motivation:** Language is sequential — meaning depends on order and context.

**Problem with regular neural networks:**
- Cannot handle sequences of variable length
- Cannot remember previous inputs
- "I love NLP" ≠ just 3 independent word vectors

### RNN Architecture:

```
hₜ = f(hₜ₋₁, xₜ)
```

More specifically:
```
hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁)
yₜ = Wₕᵧ·hₜ
```

Where:
- xₜ = input at time t (word vector)
- hₜ = hidden state at time t (memory)
- hₜ₋₁ = previous hidden state
- yₜ = output at time t
- Wₓₕ = input weight matrix
- Wₕₕ = recurrent weight matrix (shared across time steps)
- Wₕᵧ = output weight matrix

### RNN Types:
| Type | Input → Output | Use Case |
|------|---------------|----------|
| One-to-one | 1 → 1 | Image classification |
| One-to-many | 1 → sequence | Image captioning |
| Many-to-one | Sequence → 1 | Sentiment analysis |
| Many-to-many | Sequence → sequence | Machine translation |

### Training: Backpropagation Through Time (BPTT)
- Unroll RNN for all time steps
- Compute cross-entropy loss
- Backpropagate through all time steps
- Same weights used at each step → gradients combined

---

## ★ RNN Numerical Example (from your notes)

**Task:** Next character prediction for "hello"
- Vocabulary: {h, e, l, o} (size 4)
- Input encoding: one-hot vectors

**Architecture:** 4 inputs, 3 hidden neurons, 4 outputs

**Weight matrices:**
```
Wₓₕ = 3×4 matrix (connects input to hidden)
Wₕₕ = 1×1 = [0.427043] (recurrent connection, simplified)
Wₕᵧ = 4×3 matrix (connects hidden to output)
```

**Step 1: Input "h" = [1, 0, 0, 0]ᵀ**

```
Wₓₕ × x_h:
[0.287027]
[0.902874]
[0.537524]
```

**Recurrent:** Wₕₕ × hₜ₋₁ + bias (first step: hₜ₋₁=0, use bias=0.567):
```
[0.567001]
[0.567001]
[0.567001]
```

**Current state hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁):**
```
hₜ = tanh([0.287027+0.567001, 0.902874+0.567001, 0.537524+0.567001])
   = tanh([0.854028, 1.469875, 1.104525])
   = [0.693168, 0.899554, 0.802118]
```

**Output yₜ = Wₕᵧ × hₜ:**
```
yₜ = [1.797, 1.049, 0.801, 1.041]
```

**Softmax (probability over next character):**
```
P = softmax(yₜ) → probabilities for [h, e, l, o]
```

The character with highest probability is predicted as next letter.

---

### RNN Limitations:
- **Vanishing Gradient:** Gradients shrink as they propagate back → can't learn long-range dependencies
- **Exploding Gradient:** Gradients grow exponentially → training becomes unstable
- **Short-term memory:** Can't remember things from far back in sequence

---

## 15.3 LSTM — Long Short-Term Memory

**Motivation:** Fix RNN's vanishing gradient problem. Remember information over long sequences.

**Key Idea:** LSTM has **two states** instead of one:
1. **Cell state (Cₜ)** — long-term memory (highway for information)
2. **Hidden state (hₜ)** — short-term memory (output)

### LSTM has 3 Gates:

#### Gate 1: Forget Gate
**What to remove from cell state?**
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```
- Output between 0 and 1
- 0 = completely forget, 1 = completely remember
- Sigmoid activation (σ)

#### Gate 2: Input Gate
**What new information to add to cell state?**
```
iₜ = σ(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```
- iₜ = how much to update (sigmoid: 0 to 1)
- C̃ₜ = candidate values to add (tanh: -1 to 1)

#### Update Cell State:
```
Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ
```
- Forget some old info (fₜ × Cₜ₋₁)
- Add new info (iₜ × C̃ₜ)

#### Gate 3: Output Gate
**What to output from cell state?**
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ × tanh(Cₜ)
```

### Complete LSTM Equations:
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     ← Forget gate
iₜ = σ(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)     ← Input gate
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  ← Candidate values
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ         ← Update cell state
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     ← Output gate
hₜ = oₜ ⊙ tanh(Cₜ)                ← Hidden state output
```
(⊙ = element-wise multiplication)

### LSTM vs RNN:
| Feature | RNN | LSTM |
|---------|-----|------|
| Memory | Short-term only | Short + Long term |
| Vanishing Gradient | Suffers | Mostly solved |
| Gates | None | 3 (Forget, Input, Output) |
| States | 1 (hₜ) | 2 (Cₜ, hₜ) |
| Performance on long sequences | Poor | Good |

### LSTM Example Intuition:

> "Sengar likes to eat **samosas** on every Sunday, which is a popular **cuisine** in -------."

- LSTM remembers "samosas" from far back
- Forget gate discards irrelevant info (Sunday, every)
- Input gate stores "cuisine" as important
- Predicts: India (because samosas = Indian cuisine)

---

# 16. Real-World NLP Applications

| Application | Description | Techniques Used |
|-------------|-------------|-----------------|
| **Machine Translation** | Translate between languages (Google Translate) | Seq2Seq, Transformer |
| **Speech Recognition** | Convert speech to text (Siri, Alexa) | HMM, RNN, CTC |
| **Text-to-Speech (TTS)** | Convert text to audio | Concatenative synthesis, Neural TTS |
| **Sentiment Analysis** | Determine opinion (positive/negative) | BoW, CNN, LSTM |
| **Named Entity Recognition (NER)** | Find names, places, dates | HMM, CRF, BERT |
| **Question Answering** | Answer questions from text | BERT, Transformers |
| **Spell Checking** | Correct misspellings | Edit Distance, Language Models |
| **Summarization** | Condense long text | Extractive, Abstractive |
| **Chatbots** | Conversational AI (ChatGPT) | Transformers, RLHF |
| **POS Tagging** | Tag grammatical roles | HMM, CRF, Neural |
| **Parsing** | Find syntax structure | CFG, PCFG |
| **Information Retrieval** | Search engines | TF-IDF, BM25, Dense Retrieval |

---

# 📝 QUICK REVISION CHEAT SHEET

## Key Formulas at a Glance:

```
BIGRAM MLE:    P(w|prev) = C(prev,w) / C(prev)

ADD-ONE:       P(w|prev) = (C(prev,w)+1) / (C(prev)+V)

GOOD-TURING:   c* = (c+1) × N(c+1) / N(c)

ENTROPY:       H = -∑ P(x) log₂P(x)

CROSS-ENTROPY: H(P,Q) = -∑ P(x) log₂Q(x)

PERPLEXITY:    PP = 2^H  OR  P(W)^(-1/N)

TF:            TF(t,d) = count(t,d) / total_words(d)
IDF:           IDF(t) = log(N/df(t))
TF-IDF:        TF(t,d) × IDF(t)

EDIT DISTANCE: dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)

PMI:           log₂[P(x,y)/(P(x)×P(y))]

VITERBI INIT:  v₁(j) = π(j) × b(j, o₁)
VITERBI REC:   vₜ(j) = max_i[vₜ₋₁(i)×a(i,j)] × b(j,oₜ)

RNN:           hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁)
               yₜ = Wₕᵧhₜ

LSTM FORGET:   fₜ = σ(Wf·[hₜ₋₁,xₜ] + bf)
LSTM INPUT:    iₜ = σ(Wᵢ·[hₜ₋₁,xₜ] + bᵢ)
LSTM CELL:     Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ
LSTM OUTPUT:   hₜ = oₜ⊙tanh(Cₜ)
```

## Key Definitions (1-liner):

- **NLP** = AI subfield for computers to understand/generate human language
- **Tokenization** = splitting text into tokens
- **Stemming** = chop suffix to get root (may not be real word)
- **Lemmatization** = get dictionary base form using vocabulary
- **N-gram** = sequence of N consecutive words
- **Perplexity** = inverse probability per word; lower = better model
- **WSD** = determining correct meaning of ambiguous word in context
- **HMM** = model with hidden states and observable outputs
- **Viterbi** = DP algorithm to find most likely hidden state sequence
- **CFG** = grammar rules with non-terminals producing sequences
- **CNF** = CFG where every rule is A→BC or A→a
- **CYK** = DP parsing algorithm for CNF grammars
- **PCFG** = CFG with probabilities on rules
- **RNN** = neural network with recurrent connections for sequences
- **LSTM** = RNN variant with gates to handle long-range dependencies
- **Lesk** = WSD algorithm using dictionary definition overlap

---

*End of NLP Complete Exam Notes*
*All topics from ETE Syllabus covered.*
