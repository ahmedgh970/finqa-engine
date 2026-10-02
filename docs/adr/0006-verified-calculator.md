# ADR 0006 — Calculatrice vérifiée : le modèle désigne, le code calcule

## Statut

Accepté. La ligne `grade_exp1_calc` a été exécutée et jugée sur les 150 questions le
2026-09-28 : **75 good_job contre 74**, **11 calculs acceptés, 11 justes, 0 faux**. Une
règle de contrôle a été ajoutée le 2026-10-01 après un faux accepté trouvé en
démonstration avec un autre modèle ; elle ne change aucun calcul du run de référence.
Voir « Validation » et « Après l'adoption ».

## Contexte

L'ADR 0005 a fixé le contexte de référence sous 12 288 tokens : grader (note ≥ 2) puis
expansion aux chunks voisins (±1). Sur les **22 questions dont l'énoncé porte la
formule**, 12 ont désormais toutes les grandeurs dans le contexte, et **8 seulement sont
justes**. Sur ces 12, un échec ne vient plus de la récupération : la formule est donnée,
les chiffres sont là, c'est la lecture ou l'arithmétique qui casse.

**Cible** : 12/12 sur ces questions au contexte complet — 4 à gagner (10420, 04103, 03473,
04254), 8 à ne pas casser (07966, 04660, 03471, 04854, 10499, 04412, 03031, 04302).
Le good_job global n'est pas le critère : 4 questions sur 150 sont sous le seuil de
détection du juge (~5, ADR 0004).

**Deux contraintes** :

- **Zéro faux accepté.** Un chiffre présenté comme vérifié et faux est pire qu'une
  absence de chiffre : la calculatrice peut perdre une occasion, jamais introduire une
  erreur. C'est le critère décisif.
- **Agnosticité.** Aucun identifiant de question, nom d'entreprise ni formule en dur :
  l'outil doit marcher sur des questions qu'on n'a pas lues, et avec d'autres modèles.

## Design : choisir une ligne, ne pas recopier un nombre

**Routage** (`routes_to_calculator`) : une expression régulière sur la *forme* de
l'énoncé, jamais sur son sujet — une formule énoncée (« defined as », « Define X as »,
« using A and B », moyenne sur N ans, « as a % of »…), une demande de valeur, et pas une
question fermée. **30 déclenchements sur 150** : les 22 cibles et 8 autres questions
porteuses de formule. Le routage est un nœud, pour que la décision figure dans le
fichier de sortie.

**Un seul appel structuré** (`CalcSpec`). Le code découpe le contexte en lignes
numérotées `libellé = montant` (`rows_of`) et reconnaît pour chacune l'état financier
(titre ou postes caractéristiques), l'échelle (en-tête, héritée du document, ou note
repliée dans le libellé) et l'année. Le modèle ne renvoie que **l'expression** et, par
variable, **un terme et un numéro de ligne**. Il ne recopie aucun chiffre et ne calcule
jamais.

**Le code vérifie, sur le texte du rapport et non sur l'avis du modèle** :

- la ligne porte une année demandée ; une ligne sans année est refusée pour une question
  datée ; une colonne trimestrielle n'est pas une année ; une ligne de segment n'est pas
  le consolidé ;
- les qualificatifs absents de l'énoncé sont refusés (continuing operations,
  attributable to noncontrolling, per share, adjusted, non-GAAP, pro forma, excluding) ;
- la ligne porte le **même poste comptable** que le terme (lexique de 23 postes, du plus
  spécifique au plus général : EBITDAR avant EBITDA avant EBIT) et ce poste est cité par
  l'énoncé ; hors lexique, tous les mots du terme doivent figurer dans le libellé ;
- **chaque état cité par l'énoncé** (compte de résultat, bilan, flux) fournit au moins
  une valeur ;
- la **forme d'une moyenne** : ÷N en tête, somme de N ratios pour une moyenne de ratios,
  ÷2 pour une moyenne entre deux dates.

**Ce qui est fixé par le code, jamais par le modèle** : l'arrondi et le pourcentage (lus
dans l'énoncé), l'échelle commune, le signe (les coûts — capex, dividendes, intérêts,
COGS, SG&A, D&A — sont lus en valeur absolue ; un résultat négatif garde son signe).
Deux normalisations qui ne changent jamais la valeur : un `* 100` en tête d'une réponse
en pourcentage est retiré ; une ligne lue dans une note est rattachée à l'état cité s'il
imprime le même poste, le même montant, la même année.

**Calcul** en `Decimal`, par un parcours d'AST en liste blanche (`ast.parse`, aucun
`eval`). **Au moindre refus**, le générateur répond avec un prompt identique au bit près
à celui de la ligne sans calculatrice. **Sinon**, le chiffre vérifié est présenté au
générateur comme établi, et `computed_used` enregistre — comparé par valeur, pas par
chaîne — s'il a survécu dans la réponse.

## Mise au point : jeu séparé, puis hold-out

**Jeu de mise au point disjoint des 12 cibles** : 3 questions à contexte complet
(02608, 06272, 02981), 4 à contexte incomplet (04481, 04458, 06741, 10136), trois
extracteurs (granite4.1:8b, qwen3.5:9b, qwen3.5:4b). Gelé le 2026-09-25 : 21 extractions,
**0 faux accepté**.

**Leçon de méthode.** La première version faisait recopier les valeurs au modèle : il
recopiait mal (7 617 pour 7 616, année décalée, addition dans un champ numérique). La
refonte « désigner une ligne » a supprimé cette classe d'erreur. Et chaque règle ajoutée
au prompt pour corriger un cas déplaçait les choix du modèle sur les autres (mesuré) :
le prompt est tenu court, la vérification est dans le code.

**Hold-out sur les 23 questions routées jamais lues** :

| extracteur | calculs acceptés | justes | faux acceptés |
|---|---|---|---|
| granite4.1:8b | 9 | 8 | **1** (01911) |
| qwen3.5:9b | 10 | 8 | **2** (03473, 01911) |
| qwen3.5:4b | 7 | 6 | **1** (03473) |

Le critère n'était pas tenu. Cinq corrections **générales**, chacune justifiée par une
classe d'erreur et non par une question :

1. une note d'échelle repliée dans le libellé (« (in millions, except per share
   amounts) ») est lue comme l'échelle et retirée du libellé — elle faisait passer un
   résultat net pour un montant par action ; ce qu'elle disait des montants par action
   est conservé en marquant les lignes d'un bloc « per share » ;
2. une ligne sans année est refusée pour une question datée (le « Full Year » d'un
   tableau trimestriel) ;
3. EBITDAR, EBITDA et EBIT distingués ; hors lexique, tous les mots du terme requis ;
4. les coûts lus en valeur absolue dans tous les états (un intérêt entre parenthèses
   inversait un ratio de couverture) ;
5. le rattachement à l'état cité accepte le même poste, le même montant, la même année,
   y compris pour remplir un état cité resté vide.

**Revalidation sur les 30 questions routées** : 29 calculs acceptés (11, 10 et 8 selon
l'extracteur), **tous justes, 0 faux sur 90 extractions**.

**Réserve, à lire avec tous les résultats ci-dessous** : après ces corrections, les 30
questions routées ont toutes servi au réglage. La preuve indépendante est ailleurs : les
**120 questions non routées** ne doivent pas bouger, et des modèles non utilisés au réglage
doivent tenir le critère.

## Validation — run `grade_exp1_calc` sur les 150

Même contexte que la ligne `grade_exp1` (ADR 0005), granite4.1:8b, `num_ctx` 12 288,
juge `qwen3.5:9b` :

| ligne | good_job | unverified | need_help | halluc. | dont_know |
|---|---|---|---|---|---|
| `grade_exp1` (ADR 0005) | 74 | 18 | 15 | 18 | 25 |
| **`grade_exp1_calc`** | **75** | 18 | 14 | 18 | 25 |

- **30 routées, 11 calculs acceptés, 11 justes, 0 faux.** 19 refus.
- **Les 120 questions non routées ont des réponses identiques à l'octet.** C'est la
  preuve qu'il n'y a aucune régression hors du périmètre réglé.
- **Les 8 questions à ne pas casser restent good_job.**
- **Cible : 9/12.** 04254 est calculée juste (1 832), mais elle était déjà correcte. Les
  trois autres sont refusées, et le refus est le bon comportement : pour **10420** et
  **03473**, l'état cité par l'énoncé n'est pas dans le contexte (le chunk « Statements
  of Operations » de 10420 ne porte que son titre ; le 1 248 de 03473 vient d'un tableau
  trimestriel sans année) ; pour **04103**, une formule composée (DIO + DSO − DPO) que le
  modèle ne décompose pas.
- **`computed_used` : 10/11.** Sur 04302, le générateur reçoit 55,1 % (le gold) et écrit
  55,5 %.
- **Mouvements de la grille** : 02981 passe de need_help à good_job grâce au chiffre
  vérifié. Deux bascules sur des questions *refusées* se compensent et relèvent du bruit :
  06741 (même réponse finale, 6,1 %, jugée différemment) et 10136 (réponse changée).
  18 des 19 questions refusées ont un texte de réponse différent alors que leur prompt
  est identique : l'appel supplémentaire modifie vraisemblablement l'état d'Ollama
  (cache KV) et donc les calculs flottants. C'est une hypothèse, pas une mesure.
- **Coût** : un appel LLM de plus, sur les 30 questions routées seulement. La latence du
  run de référence n'est pas exploitable (GPU bridé par sa limite de puissance pendant
  le run). Mesure isolée en direct : nœud calculatrice **42 s** sur 04660, granite à
  60 % sur GPU.

## Après l'adoption

En démonstration, avec qwen3.5:4b, **un faux accepté** : sur « FY2020 revenue / (average
total assets between FY2019 and FY2020) », le modèle a désigné le chiffre d'affaires 2019,
et la vérification d'année, qui ne contrôlait que l'intervalle 2019–2020, l'a laissé
passer (1,22 au lieu de 1,33). Règle ajoutée : **dans la formule de l'énoncé, une année
écrite juste avant un poste fixe l'année de ce poste** — dans le titre, « FY2019 inventory
turnover ratio » nomme un indicateur et ne fixe rien. Rejouée sur les 90 extractions de
la revalidation : **aucun des 29 calculs acceptés ne change** ; sur le run de référence :
**les 11 restent acceptés**. En direct, la question est désormais refusée avec le motif
« L85 : [2019], but the formula asks for revenue of [2020] ».

C'est la limite de la validation sur 30 questions : un contrôle couvre ce qu'on a vu. Le
critère « zéro faux » se maintient en traitant chaque faux trouvé comme une classe
d'erreur, avec une règle générale rejouée sur tout l'historique.

## Décision

1. **La calculatrice est adoptée dans le workflow servi** (`serve_grade_exp1_calc_1024`),
   pour la **vérifiabilité** : un chiffre contrôlé par le code, avec les lignes et les
   pages d'où il vient, sans aucun faux mesuré.
2. **Elle n'est pas présentée comme un gain de qualité** : +1 good_job, sous le seuil du
   juge. granite calcule déjà juste quand il a les bonnes lignes.
3. **Les contrôles restent stricts.** Un état cité absent du contexte entraîne un refus,
   même quand un tableau de synthèse (« Selected Financial Data ») porte le bon chiffre :
   l'assouplir rouvrirait la classe d'erreur qui a motivé le contrôle (une ligne de
   pension tenant lieu de D&A).
4. **Le chiffre vérifié est proposé, pas imposé** au générateur. Forcer une ligne
   « résultat vérifié » en fin de réponse corrigerait 04302, mais produirait des réponses
   contradictoires et changerait le prompt mesuré.

## Ce qui n'est pas tranché

- **La couverture** : 11 calculs sur 30 routées. Les formules composées (04103) et les
  questions sans formule explicite (une phase 2 avec un glossaire de définitions
  standard, écrit à partir de sources publiques) restent hors périmètre.
- **La latence à froid**, sur un GPU non bridé, reste à mesurer.
- **La dépendance à l'extracteur** : qwen3.5:4b raisonne à voix haute et peut être coupé
  par la limite de sortie (désormais signalé par `truncated`) ; il ignore parfois le
  chiffre vérifié.

## Conséquences

- Code : `src/workflow/calculator.py` (lignes, contrôles, évaluateur, routage),
  `src/workflow/schemas.py` (`CalcSpec`, `CalcVariable`), `build_calc_prompt` dans
  `src/workflow/prompts.py`, nœuds `route` et `calculate` dans le graphe, bloc du chiffre
  vérifié dans `src/llm/prompts.py`. Instrumentation : `is_numeric`, `computed`,
  `calc_error`, `computed_used`, et le détail de chaque calcul pour l'inspection.
- Outils : `scripts/calc_tuning.py` (extraction seule, sans génération ni écriture, avec
  `--model`), `scripts/demo_calc_details.py` (rejoue le calcul d'un run enregistré et ne
  le garde que s'il reproduit le chiffre : 11/11).
- Configs : `grade_exp1_calc_1024_replayed.yaml` (la ligne mesurée),
  `serve_grade_exp1_calc_1024.yaml` (le workflow servi, recherche en direct).
- La démo montre, pour chaque calcul, la formule, les lignes désignées, ce que le code y
  a lu et le motif d'un refus.
