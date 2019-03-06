from __future__ import print_function, division
from matplotlib import pyplot as plt
import json
import random
import numpy as np

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions
from debiaswe.debias import debias

# Step 0: load google news wordvec
E = WordEmbedding('./embeddings/w2v_gnews_small.txt')

# Step 1: load professions
professions = load_professions()
profession_words = [p[0] for p in professions]

# Step 2: define racial direction
names = ["Emily", "Aisha", "Anne", "Keisha", "Jill", "Tamika", "Allison", "Lakisha", "Laurie", "Tanisha", "Sarah",
         "Latoya", "Meredith", "Kenya", "Carrie", "Latonya", "Kristen", "Ebony", "Todd", "Rasheed", "Neil", "Tremayne",
         "Geoffrey", "Kareem", "Brett", "Darnell", "Brendan", "Tyrone", "Greg", "Hakim", "Matthew", "Jamal", "Jay",
         "Leroy", "Brad", "Jermaine"]
names_group1 = [names[2 * i] for i in range(len(names) // 2)]
names_group2 = [names[2 * i + 1] for i in range(len(names) // 2)]

vs = [sum(E.v(w) for w in names) for names in (names_group2, names_group1)]
vs = [v / np.linalg.norm(v) for v in vs]

v_racial = vs[1] - vs[0]
v_racial = v_racial / np.linalg.norm(v_racial)

# Step 3: generate racial biased analogies
a_racial = E.best_analogies_dist_thresh(v_racial)

for (a,b,c) in a_racial:
	print('{:>20}'.format(a) + " - " + '{:>20}'.format(b) + " - " + '{:>10}'.format(str(c)))

# Step 4: analyse racial bias in profession analogies
sp = sorted([(E.v(w).dot(v_racial), w) for w in profession_words])
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
sp_debiased = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])
print(sp_debiased[0:20])
print("\n")
print(sp_debiased[-20:])