from __future__ import print_function, division
from matplotlib import pyplot as plt
import json
import random
import numpy as np

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions
from debiaswe.data import load_sports
from debiaswe.debias import debias

# Step 0: load google news wordvec
E = WordEmbedding('./embeddings/w2v_gnews_small.txt')

# Step 1: load professions
professions = load_professions()
profession_words = [p[0] for p in professions]
# Step 1: load sports
sports = load_sports()
sport_words = [s[0] for s in sports]

print("\n")

# Step 2: define gender direction
v_gender = E.diff('she', 'he')

print("\n")

# Step 3: generate analogies based on gender = 'man:x :: woman:y'
a_gender = E.best_analogies_dist_thresh(v_gender)

print("\n")

for (a,b,c) in a_gender:
	print('{:>20}'.format(a) + " - " + '{:>20}'.format(b) + " - " + '{:>10}'.format(str(c)))		
	# c = score of pair of words a and b
	# metric in paper page 4 equation 1

print("\n")

# Step 4: analyse gender bias in profession analogies. Structure: (projection score, profession word)
# sp = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])
sp = sorted([(E.v(w).dot(v_gender), w) for w in sport_words])
print(sp[0:20])
print("\n")
print(sp[-20:])

# Step 5: load gender related word lists to help us with debiasing
with open("./data/definitional_pairs.json", "r") as f:
	defs = json.load(f)

with open("./data/equalize_pairs.json", "r") as f:
	equalize_pairs = json.load(f)

with open("./data/gender_specific_seed.json", "r") as f:
	gender_specific_words = json.load(f)

# Step 6: Debias
print("!!! DEBIAS !!!")
debias(E, gender_specific_words, defs, equalize_pairs)

# analogies gender
a_gender_debiased = E.best_analogies_dist_thresh(v_gender)
print("\n")

for (a,b,c) in a_gender_debiased:
	print('{:>20}'.format(a) + " - " + '{:>20}'.format(b) + " - " + '{:>10}'.format(str(c)))

# profession analysis gender
sp_debiased = sorted([(E.v(w).dot(v_gender), w) for w in sport_words])
# sp_debiased = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])
print(sp_debiased[0:20])
print("\n")
print(sp_debiased[-20:])