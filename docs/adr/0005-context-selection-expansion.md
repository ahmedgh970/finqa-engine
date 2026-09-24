# ADR 0005 — Que mettre dans une fenêtre de 12K : grader + expansion aux voisins

## Statut

Accepté. La configuration retenue a été exécutée et jugée le 2026-09-24 : **74 good_job**
contre 68 pour le grader seul, en lisant 6 314 tokens. Voir « Validation ».

## Contexte

L'ADR 0001 a retenu `reranked(dense)` prefetch 50, doc-scoped. L'ADR 0004 a mesuré la
ligne `advanced` (k = 20, aucun nœud) et la ligne `grading` (grader par passage) sur la
grille ancrée dans l'evidence. Il en ressortait que la meilleure ligne, `advanced` à
`num_ctx` 24576, coûtait 187 s par question, et que le grader ne rendait pas ce qu'il
promettait.

Une analyse des échecs sur les 22 questions dont l'énoncé contient la formule a montré
que **le contexte manquait une grandeur dans 9 des 12 échecs**, même avec 20 passages.
La cause est structurelle : le `HybridChunker` coupe un état financier en chunks
consécutifs, et le reranker n'en remonte qu'un — souvent la moitié sans le chiffre utile.

Deux contraintes cadrent la décision :

- **La fenêtre de contexte est de la VRAM.** Elle dimensionne le cache KV, donc c'est un
  budget à dépenser, pas un nombre à augmenter. La cible est `num_ctx` **12288**.
- **Le dépassement est silencieux.** Vérifié : avec 745 tokens de prompt dans une fenêtre
  de 1024, le modèle a produit 300 tokens là où 279 places restaient. Ollama n'échoue pas,
  il **évince les tokens les plus anciens** — donc les instructions d'ancrage d'abord.
  D'où le trim maison (`_fit_context`), qui coupe par le bas et le comptabilise.

## Ce que les métriques de récupération ne voyaient pas

`recall@k` est défini **à la page** : un passage compte dès qu'il tombe sur une page de
l'evidence. Sur les trois corpus, à k = 20, il est plat — et trompeur :

| chunks | recall@20 | tokens / prompt | chiffres de l'evidence | toutes les grandeurs |
|---|---|---|---|---|
| 256 | 0,738 | 4 261 | 0,628 | 29 % |
| 512 | 0,744 | 7 874 | 0,705 | 38 % |
| 1024 | 0,751 | 14 437 | 0,803 | 48 % |

D'où une métrique ajoutée à l'éval retrieval (`src/evaluation/retrieval/coverage.py`) :
la part du **texte de l'evidence** réellement présente dans le contexte lu. Trois mesures,
et une leçon de méthode :

- `evidence_words` — recouvrement d'ensembles de mots. Généreux : « total », « net » et
  les millésimes se retrouvent partout.
- `evidence_overlap` — recouvrement des 5-grammes. **Inutilisable sur les tableaux** :
  FinanceBench stocke un tableau comme le dump brut du PDF, Docling le sérialise en
  `label, colonne = valeur`, donc aucune séquence ne correspond même quand tous les
  chiffres sont là. Les 29 questions qui y obtiennent ≥ 0,9 sont exactement celles dont
  l'evidence est narrative. Conservé, mais à lire sur la prose uniquement.
- `evidence_figures` / `all_evidence_figures` — les montants porteurs d'information,
  années et entiers à un ou deux chiffres exclus (≈ 46 gardés sur 57 par question).
  **C'est la mesure de référence pour les tableaux**, donc pour ce corpus.

## Méthode : balayer sans générer

Tout ce qui construit un contexte est déterministe — récupération rejouée depuis un
fichier figé, grader à température 0 dont les notes sont enregistrées par passage,
expansion arithmétique, trim fonction du texte. Une configuration s'évalue donc **sans
générer** (`scripts/context_sweep.py`).

Validation de la méthode : en rejouant ainsi deux lignes déjà exécutées (±3 et ±6 sur le
corpus 256), les contextes reconstruits sont **identiques à 150/150** à ceux que le
générateur avait lus. Le balayage est une prédiction exacte, pas une estimation.

## Résultats — `num_ctx` 12288

| chunks | note ≥ | fenêtre | passages lus | tokens | coupées | montants | **complètes** |
|---|---|---|---|---|---|---|---|
| 1024 | — | — | 10,2 | 7 593 | 147 | 0,648 | 43,9 % |
| 1024 | 2 | — | 5,6 | 4 048 | 10 | 0,656 | 44,6 % |
| **1024** | **2** | **±1** | 11,4 | **6 314** | 64 | **0,722** | **56,8 %** |
| 1024 | 2 | ±2 | 13,9 | 7 236 | 102 | 0,707 | 56,1 % |
| 1024 | 1 | ±1 | 11,5 | 6 395 | 67 | 0,720 | 56,8 % |
| 1024 | 3 | ±1 | 9,7 | 5 444 | 27 | 0,684 | 52,5 % |
| 512 | — | ±1 | 22,5 | 7 694 | 145 | 0,661 | 47,5 % |
| 256 | — | — | 19,6 | 4 261 | 0 | 0,568 | 36,0 % |
| 256 | 2 | — | 5,9 | 1 285 | 0 | 0,414 | 23,0 % |
| 256 | 2 | ±4 | 30,1 | 5 884 | 42 | 0,661 | 52,5 % |

Pour référence, hors budget : `advanced` k20 atteint 56,1 % à 24576 (14 340 tokens lus)
et 56,8 % à 32768. **La ligne retenue égale donc le 32K en lisant 44 % de son contexte.**

## Analyse

**Le grader et l'expansion sont un seul mécanisme.** Seul, le grader garde 5,6 passages
sur 20 et lit 4 048 tokens — un tiers de la fenêtre inutilisée, et pas plus d'evidence
qu'en lisant les vingt. Seule, l'expansion n'a rien à sélectionner : chaque passage
devient une ancre, la fenêtre sature et 147 questions sur 150 sont tronquées. Ensemble,
56,8 % en 6 314 tokens. Ce que le grader achète n'est pas de l'evidence, c'est de
**ne pas payer pour le reste** : à couverture égale (48,2 %), 5 120 tokens avec grader
contre 7 603 sans, soit un tiers de moins.

**Le seuil 2 est le bon.** Le seuil 3 laisse trop peu d'ancres et il faut des fenêtres
plus larges pour compenser, ce qui coûte plus cher à l'arrivée : 4 751 tokens pour 41,0 %
contre 4 187 pour 41,7 % au seuil 2. Le seuil 1 est indiscernable du 2 en lisant un peu
plus. Cohérent avec le barème : 2 signifie « contient une partie de ce sur quoi la
réponse se construit ».

**La fenêtre dépend de la taille du chunk, pas du budget.** ±1 sur des chunks de 1024,
±4 sur des chunks de 256 — et dans ce second cas le contexte compte 30 passages au lieu
de 11, pour 4 points de complétude en moins. Au-delà de l'optimum, la complétude
**baisse** : ±2 sur 1024 fait passer les coupes de 64 à 102, ±6 puis ±8 sur 256 régressent
à 53,2 % puis 51,1 %. Le trim est le mécanisme qui sanctionne l'excès.

**Le corpus 1024 domine.** Au-delà de ~4 000 tokens de budget, aucune configuration 256
ou 512 n'apparaît sur le front de Pareto. Une evidence fait 443 tokens en moyenne : elle
dépasse un chunk de 256 pour 63 % des questions, un chunk de 1024 pour 9 % seulement.

**La fragmentation a un coût propre.** Sur les lignes jugées du corpus 256, les refus
montent à 30–47 `dont_know` contre 21–26 sur le 1024, avec en miroir moins
d'hallucinations. Le modèle devient prudent quand le contexte est morcelé.

## Décision

1. **Le corpus de référence reste `docling_hybrid_1024_bge-m3`.** L'ablation 256 / 512 /
   1024 est tranchée par la couverture d'evidence, pas par `recall@k`, qui en est aveugle.
2. **Le nœud `expansion` est adopté** (`src/workflow/expansion.py`) : chaque passage
   retenu est lu avec les `window` chunks qui l'entourent dans son document, blocs
   contigus fusionnés, sans doublon, sans appel LLM.
3. **La configuration de référence sous 12288 est : corpus 1024, `keep_threshold: 2`,
   `min_chunks: 3`, `expansion.window: 1`.** 6 314 tokens lus, 56,8 % de questions dont
   toutes les grandeurs sont présentes.
4. **`estimated_context` reste à `len/3`.** Mesuré sur 36 prompts avec le tokenizer du
   générateur (`prompt_eval_count`) : médiane 3,90 caractères par token, **pire cas 3,22**.
   Le diviseur 3 ne garde que ~7 % de marge ; le calibrer sur la moyenne exposerait à
   l'éviction silencieuse des instructions.
5. **`max_tokens` reste à 1024.** Une réponse sur six dépasse 512 tokens ; la réserve
   n'est pas surdimensionnée.

## Validation

Exécution et jugement de la configuration retenue, juge `qwen3.5:9b` (ADR 0004) :

| ligne | appels / Q | tokens lus | latence | complètes | good_job | dk | halluc. | need_help |
|---|---|---|---|---|---|---|---|---|
| advanced k20, 24K | 1 | 14 340 | 187 s | 56,1 % | 77 | 21 | 16 | 23 |
| **grading ±1, 12K** | 21 | **6 314** | 136 s | **56,8 %** | **74** | 25 | 18 | 15 |
| advanced k20, 12K | 1 | 7 593 | 107 s | 43,9 % | 72 | 24 | 22 | 17 |
| grading ±0, 12K | 21 | 4 048 | 161 s | 44,6 % | 68 | 26 | 23 | 18 |
| grading 256 ±3, 12K | 21 | 5 120 | 117 s | 48,2 % | 72 | 32 | 16 | 15 |

**La prédiction du balayage s'est vérifiée au chiffre près** : 6 314 tokens lus, 64
questions tronquées, 56,8 % de complétude — les trois annoncés sans générer.

**L'expansion paie sur le grader : 68 → 74** (+12 gagnées, −6 perdues), et **9 des 12
gains viennent de `need_help`** : des questions dont l'evidence était déjà récupérée mais
que le générateur ratait, faute d'avoir le tableau entier. C'est le mécanisme visé.

**Contre les deux lignes sans sélection, rien n'est tranché** : +2 sur `advanced 12K`,
−3 sur `advanced 24K`, tous deux sous le seuil de ~5 good_job du juge. Ce qui est réel
est le contexte : **56 % de tokens en moins que la ligne 24K pour 3 good_job d'écart**.
Le coût reste 21 appels contre 1, et 136 s contre 107 s.

**Décision de lecture** : la ligne est retenue pour une cible sous contrainte de VRAM,
où le contexte lu est le poste dimensionnant. Elle ne remplace pas `advanced 12K` lorsque
le nombre d'appels prime.

## Ce qui n'est pas tranché

**Le corpus 512 n'a pas de run du grader**, donc il n'apparaît qu'au seuil 0. Vu la
domination du 1024, l'ablation n'a pas été jugée prioritaire.

**Le surcoût d'appels n'est pas amorti par le score** : à ±2 good_job près, `advanced 12K`
fait le même travail avec un seul appel. La ligne retenue se justifie par le contexte lu,
pas par la qualité mesurée.

## Conséquences

- Code : `src/workflow/expansion.py` (nœud + fonction), `expansion` dans la config du
  workflow, `name` pour nommer une cellule dont la sélection est rejouée,
  `src/evaluation/retrieval/coverage.py` (couverture d'evidence, intégrée à l'éval
  retrieval), `scripts/expand_context.py` et `scripts/context_sweep.py` (hors ligne),
  `scripts/evidence_overlap.py` (détail par question).
- Les passages transportent leur `chunk_id` du corpus de bout en bout (prompts
  matérialisés, replay, sorties du workflow) ; un fichier antérieur est rattrapé par
  correspondance de texte, les textes de chunks étant uniques dans un corpus.
- `make prompts` accepte `CHUNK_SIZE=`, `KS=` et `PREFETCH=`, le prefetch entrant dans le
  nom du fichier : une profondeur de présélection différente est une autre récupération,
  pas une tranche plus profonde de la même.
- Le README documente le tableau « What fits a 12K window ».
- **Sur les 22 questions dont l'énoncé porte la formule** : 12 ont désormais toutes les
  grandeurs de la formule dans le contexte, et **8 d'entre elles sont correctement
  répondues**. Les 10 autres échouent encore par contexte incomplet (6 refus, 3
  hallucinations, 1 `need_help`), dont trois par un piège d'extraction relevé à la
  lecture : valeurs du tableau de flux au lieu des soldes du bilan, revenus par segment
  au lieu des consolidés, montants trimestriels au lieu d'annuels.
- **À mesurer ensuite** : la voie calculatoire sur les 12 questions au contexte complet —
  4 y échouent encore alors que toutes les grandeurs sont présentes, ce qui en fait un
  problème d'arithmétique et non de récupération.
