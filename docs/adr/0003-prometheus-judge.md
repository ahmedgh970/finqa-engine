# ADR 0003 — Juge local reproductible : Prometheus-2 vs juge frontier (Claude)

> La reproductibilité annoncée ici a depuis été vérifiée, la validation étendue de
> 4 à 9 modèles, et la limite sur les refus requalifiée : voir
> « Amendement du 2026-09-21 » en fin de document.

## Statut

Accepté.

## Contexte

ADR 0002 a produit les résultats de génération avec un **juge frontier (Claude)**
sur la grille `correct` / `grounded` / `equivalent`. Ce juge est capable et
validé à la main (97,4 % d'accord humain), mais il **n'est pas reproductible
programmatiquement** : c'est un jugement en conversation, pas un modèle figé
qu'on peut re-lancer à l'identique sur un CI ou un nouveau run. Pour publier des
chiffres défendables et re-jauger à volonté, il faut un **juge local,
déterministe et re-runnable**.

Candidat : **Prometheus-2**, un modèle d'évaluation open fine-tuné pour la
notation, servi via Ollama (`ggozad/prometheus2:latest`, quantifié q4).

Réglage :

- **Protocole Absolute Grading `prometheus-eval` verbatim** (dévier du format
  dégrade le modèle) : system prompt dédié, `###Task Description`, l'instruction
  (= la question), la réponse à évaluer, la **Reference Answer (Score 5) = gold**,
  et un **score rubric 1-5**. Sortie `Feedback: … [RESULT] N`.
- **Rubric unique 1-5** « exactitude financière vs référence » (correctness et
  groundedness fondus dans le score — Prometheus n'émet qu'un score par appel).
- `temperature=0`, `num_ctx=8192` (le pire prompt réel mesuré = 1660 tokens),
  appel Ollama `/api/chat`.
- **Sous-ensemble de 8 cellules** : `qwen3.5:4b`, `qwen3.5:9b`, `granite4.1:3b`,
  `granite4.1:8b`, chacun à **k10 et k20** — les 4 modèles couvrent l'éventail de
  comportements (honest-but-timid vs confident) aux 2 profondeurs diagnostiques.
  150 QA par cellule. **0 réponse non parsée** sur 1200 (`[RESULT]` fiable à 100 %).

**Comparaison par classement, pas par question.** Les deux juges notent sur des
axes différents — Claude sort un booléen `equivalent` (→ `equivalent %` par
cellule), Prometheus un scalaire 1-5 (→ moyenne par cellule). On ne compare donc
pas verdict par verdict, mais le **classement des modèles** que chaque juge
induit, par **corrélation de rang** (Spearman ρ, Kendall τ). C'est le bon niveau
de comparaison quand les échelles diffèrent.

## Résultats

Note Prometheus moyenne (1-5) vs `equivalent %` de Claude, par modèle et profondeur :

| Modèle | Prom k10 | Prom k20 | Claude k10 | Claude k20 |
|---|---:|---:|---:|---:|
| granite4.1:8b | 4.13 | **4.17** | 62.0 % | **65.3 %** |
| qwen3.5:4b    | 3.93 | 4.11 | 53.3 % | 60.0 % |
| qwen3.5:9b    | 3.55 | 3.64 | 54.7 % | 58.0 % |
| granite4.1:3b | 3.28 | 3.37 | 39.3 % | 40.0 % |

Classement des 8 cellules par les deux juges et corrélation de rang
(`scripts/judge_ranking.py`) :

| Cellule | Claude equiv% (rang) | Prometheus moy (rang) | Δrang |
|---|---|---|---:|
| granite4.1:8b k20 | 65.3 (1) | 4.17 (1) | 0 |
| granite4.1:8b k10 | 62.0 (2) | 4.13 (2) | 0 |
| qwen3.5:4b k20 | 60.0 (3) | 4.11 (3) | 0 |
| qwen3.5:9b k20 | 58.0 (4) | 3.64 (5) | −1 |
| qwen3.5:9b k10 | 54.7 (5) | 3.55 (6) | −1 |
| qwen3.5:4b k10 | 53.3 (6) | 3.93 (4) | +2 |
| granite4.1:3b k20 | 40.0 (7) | 3.37 (7) | 0 |
| granite4.1:3b k10 | 39.3 (8) | 3.28 (8) | 0 |

**Spearman ρ = +0.929 · Kendall τ = +0.857.**

## Analyse

**Accord de rang quasi parfait → Prometheus est validé comme juge local
reproductible.** Un modèle 7B open, déterministe et re-runnable, retrouve le
classement du juge frontier (ρ = 0.93). Le haut de tableau est identique
(granite4.1:8b ≫ qwen3.5:4b ≫ … ≫ granite4.1:3b), et Prometheus confirme
**indépendamment** les deux résultats centraux d'ADR 0002 :

- **la profondeur aide** — k20 > k10 pour les 4 modèles sur la note Prometheus ;
- **le 4B n'est pas battu par le 9B** — Prometheus le classe même *au-dessus*.

**La seule divergence est interprétable, pas du bruit.** Prometheus remonte
`qwen3.5:4b k10` (rang 6 → 4) et fait glisser les deux cellules `qwen3.5:9b`
d'un cran. Cause **mesurée** : le rubric Prometheus **Score 1 = « faux OU
refus »** confond abstention et erreur. Le 9B « honest-but-timid » refuse
beaucoup — sa distribution est bimodale (à k20 : 32 notes de « 1 » *et* 66 de
« 5 »), et **26 de ces 32 « 1 » sont des refus explicites** (« The provided
context does not contain… »). Ces refus plombent sa moyenne. La grille à deux
axes de Claude, elle, **sépare** le refus honnête (`grounded=oui`, non
`equivalent`) de la fabrication, donc elle ne pénalise pas le 9B de la même
façon.

→ **Les deux juges s'accordent sur *qui* est le meilleur, mais divergent sur
*comment* traiter le refus honnête** : le scalaire unique de Prometheus fond
refus et erreur dans un même « 1 », là où la grille de Claude les distingue.
C'est la limite structurelle d'un juge à score unique, à garder en tête quand on
lit les **moyennes absolues** de Prometheus (le *classement*, lui, tient).

## Décision

1. **Adopter Prometheus-2 (`ggozad/prometheus2` q4, Absolute Grading) comme juge
   local reproductible** de la génération ; **Claude reste la référence frontier
   de validation**. On publie les deux — Prometheus pour la reproductibilité,
   Claude pour l'autorité.
2. **La comparaison inter-juges se fait par corrélation de rang** (ρ/τ), pas par
   accord question-par-question : les axes (binaire 2-axes vs scalaire 1-5) ne
   sont pas commensurables, seul le classement l'est.
3. **On lit Prometheus au niveau du classement, pas de la note absolue** : sa
   sévérité sur les refus biaise les moyennes absolues des modèles prudents, sans
   affecter l'ordre.

## Conséquences

- Code figé :
  - `src/evaluation/judge/protocols/prometheus.py` : protocole verbatim et
    parse `[RESULT]` ;
  - `make judge PROTOCOL=prometheus ANSWERS=...` (config
    `configs/evaluation/judge/prometheus.yaml`) : batch resumable ;
  - `scripts/judge_ranking.py` : ρ/τ, avec auto-découverte des cellules jugées
    par les deux juges.
- Verdicts dans `data/processed/judged/{stem}_judged_by_prometheus.jsonl`.
- **Limites assumées** :
  - Prometheus **conflate refus et erreur** (Score 1) → sous-estime les modèles
    honnêtes-timides en valeur absolue ; le ranking n'en souffre pas.
  - Sous-ensemble de **8 cellules** (4 modèles × k10/k20), pas les 23 cellules
    d'ADR 0002 — suffisant pour valider le classement, extensible au besoin.
  - Le juge Claude n'étant pas re-runnable programmatiquement, l'**accord
    par question (kappa)** n'a pas été calculé ; la corrélation de rang est le
    niveau adapté vu les axes différents.
  - GPU portable bridé (firmware) → ~70 s/question ; non bloquant (resumable).
- **Reste à mesurer** (ADR suivant) : Ragas (faithfulness, answer_relevancy) sur
  les meilleurs modèles à k20 — une lentille orthogonale (fidélité au contexte /
  pertinence), sans golden set côté génération.

## Amendement du 2026-09-21 : reproductibilité vérifiée, extension à 13 lignes

Le juge a servi à remplir la colonne Prometheus du tableau de génération du
README : nos 10 lignes (9 modèles locaux à leur meilleure profondeur, plus la
ligne grading du workflow) et les 3 jeux de réponses publiés par FinanceBench
en `singleStore`. Trois conclusions s'ajoutent à l'ADR.

### 1. Reproductibilité : vérifiée

Trois cellules déjà notées en juillet ont été re-jugées avec le même protocole,
le même modèle et `temperature=0`, à plusieurs semaines d'écart :

| Cellule | ADR 0003 | Re-run | Δ |
|---|---:|---:|---:|
| granite4.1:8b k20 | 4.17 | 4.153 | 0.017 |
| qwen3.5:4b k20 | 4.11 | 4.067 | 0.043 |
| qwen3.5:9b k20 | 3.64 | 3.700 | 0.060 |

Écart maximal **0,06 point sur 5**, classement inchangé. La quantification q4 et
l'échantillonnage résiduel d'Ollama expliquent ce bruit. C'est la propriété qui
motivait le choix de Prometheus face au juge frontier : le même run re-jugé
redonne le même chiffre.

### 2. L'accord de classement tient sur 10 lignes

L'ADR validait ρ = 0,929 sur 8 cellules et 4 modèles. Sur les 10 lignes du
tableau du README, qui couvrent 9 modèles de 3 à 12 milliards de paramètres et
une ligne de workflow :

- **ρ = 0,964** contre `equivalent` (correct et grounded) ;
- ρ = 0,939 contre `correct` seul.

L'accord ne se dégrade donc pas en élargissant l'éventail des modèles, il
s'améliore. Prometheus reste utilisable comme instrument de classement.

### 3. Les refus cassent la comparaison entre familles de lignes

La limite « Prometheus conflate refus et erreur » était assumée comme sans effet
sur l'ordre. Les réponses publiées par FinanceBench montrent qu'elle **cesse
d'être inoffensive dès que les taux de refus diffèrent fortement** entre les
lignes comparées. Les décomptes viennent du champ `label` publié avec chaque
réponse, ils ne sont pas estimés :

| Ligne publiée | Refus / 150 | Réponses fausses | correct (audité) | Prometheus |
|---|---:|---:|---:|---:|
| gpt-4-1106-preview | 58 | 17 | 48,0 % | 3.393 |
| gpt-4 | 71 | 16 | 41,3 % | 2.713 |
| llama-2-70b-chat | 7 | 81 | 37,3 % | **3.747** |

llama-2-70b-chat devance les deux lignes GPT-4 chez Prometheus tout en étant la
moins souvent correcte des trois : il répond faux 81 fois là où GPT-4 s'abstient,
et un refus vaut 1 quand une réponse fausse mais plausible obtient souvent 2 ou 3.
Sur les 13 lignes réunies, l'accord avec `correct` tombe à **ρ = 0,737**, contre
0,939 sur nos 10 lignes seules, dont les taux de refus sont comparables.

**Règle d'usage retenue** : comparer les notes Prometheus **à l'intérieur d'une
famille de lignes dont le taux de refus est comparable**, jamais entre familles.
Le tableau du README porte cette réserve sous les lignes publiées.

### Conséquences

- La colonne Prometheus du README est renseignée pour les 13 lignes, avec la
  réserve ci-dessus.
- Les limites de l'ADR restent valables ; la première (refus notés 1) est
  requalifiée : elle n'est neutre sur le classement que **si** les lignes
  comparées refusent à un taux voisin.
- Le protocole, le modèle et la configuration sont inchangés : cet amendement
  n'ajoute aucune décision, il étend la validation et précise une limite.
