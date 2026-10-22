# Retrieval evaluation (2026-10-22 21:17)

Golden set: 20 questions, 17 answerable, 3 unanswerable. Small benchmark: differences of one or two questions are within noise.

C settings: reranker=BAAI/bge-reranker-v2-m3, candidates=30, max_length=512, quantize=int8; B/C: dense_k=30, bm25_k=30, rrf_k=60.

### All answerable questions

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 17 | 8/17 (47%) | 14/17 (82%) | 15/17 (88%) | 15/17 (88%) | 0.522 | 0.624 | 0.247 | 0.147 | 0.579 | 0.518 | 18/22 | 13/17 |
| B_hybrid | 17 | 10/17 (59%) | 14/17 (82%) | 16/17 (94%) | 14/17 (82%) | 0.424 | 0.484 | 0.188 | 0.112 | 0.687 | 0.471 | 18/22 | 13/17 |
| C_hybrid_rerank | 17 | 13/17 (76%) | 15/17 (88%) | 16/17 (94%) | 16/17 (94%) | 0.671 | 0.700 | 0.294 | 0.153 | 0.809 | 0.670 | 18/22 | 13/17 |

### By category: factual

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 7 | 2/7 (29%) | 5/7 (71%) | 5/7 (71%) | 5/7 (71%) | 0.452 | 0.500 | 0.200 | 0.114 | 0.386 | 0.385 | 5/7 | 5/7 |
| B_hybrid | 7 | 3/7 (43%) | 5/7 (71%) | 6/7 (86%) | 5/7 (71%) | 0.381 | 0.429 | 0.171 | 0.100 | 0.543 | 0.393 | 6/7 | 6/7 |
| C_hybrid_rerank | 7 | 5/7 (71%) | 6/7 (86%) | 6/7 (86%) | 6/7 (86%) | 0.714 | 0.714 | 0.286 | 0.143 | 0.762 | 0.671 | 6/7 | 6/7 |

### By category: identifier

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 3 | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 0.500 | 0.500 | 0.267 | 0.133 | 0.833 | 0.509 | 3/5 | 1/3 |
| B_hybrid | 3 | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 0.389 | 0.389 | 0.200 | 0.100 | 1.000 | 0.517 | 3/5 | 1/3 |
| C_hybrid_rerank | 3 | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 0.500 | 0.500 | 0.267 | 0.133 | 1.000 | 0.616 | 3/5 | 1/3 |

### By category: multi_chunk

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.600 | 0.800 | 0.400 | 0.300 | 0.625 | 0.683 | 5/5 | 2/2 |
| B_hybrid | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.433 | 0.533 | 0.300 | 0.200 | 0.625 | 0.516 | 4/5 | 1/2 |
| C_hybrid_rerank | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.700 | 0.700 | 0.500 | 0.250 | 0.625 | 0.622 | 4/5 | 1/2 |

### By category: paraphrase

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 3 | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 0.833 | 0.833 | 0.333 | 0.167 | 0.750 | 0.755 | 3/3 | 3/3 |
| B_hybrid | 3 | 2/3 (67%) | 2/3 (67%) | 3/3 (100%) | 2/3 (67%) | 0.333 | 0.500 | 0.133 | 0.100 | 0.708 | 0.473 | 3/3 | 3/3 |
| C_hybrid_rerank | 3 | 2/3 (67%) | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 0.667 | 0.833 | 0.267 | 0.167 | 0.722 | 0.663 | 3/3 | 3/3 |

### By category: section

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 2 | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 0.250 | 0.750 | 0.100 | 0.100 | 0.571 | 0.473 | 2/2 | 2/2 |
| B_hybrid | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.750 | 0.750 | 0.200 | 0.100 | 0.750 | 0.622 | 2/2 | 2/2 |
| C_hybrid_rerank | 2 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.750 | 0.750 | 0.200 | 0.100 | 1.000 | 0.807 | 2/2 | 2/2 |

### By language: en

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 10 | 2/10 (20%) | 7/10 (70%) | 8/10 (80%) | 8/10 (80%) | 0.337 | 0.477 | 0.160 | 0.110 | 0.359 | 0.314 | 10/13 | 7/10 |
| B_hybrid | 10 | 3/10 (30%) | 7/10 (70%) | 9/10 (90%) | 7/10 (70%) | 0.353 | 0.457 | 0.140 | 0.100 | 0.467 | 0.355 | 10/13 | 7/10 |
| C_hybrid_rerank | 10 | 6/10 (60%) | 8/10 (80%) | 9/10 (90%) | 9/10 (90%) | 0.557 | 0.607 | 0.220 | 0.120 | 0.675 | 0.560 | 10/13 | 7/10 |

### By language: mixed

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.750 | 0.750 | 0.400 | 0.200 | 0.625 | 0.632 | 3/3 | 2/2 |
| B_hybrid | 2 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.583 | 0.583 | 0.300 | 0.150 | 1.000 | 0.689 | 3/3 | 2/2 |
| C_hybrid_rerank | 2 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 0.750 | 0.750 | 0.400 | 0.200 | 1.000 | 0.790 | 3/3 | 2/2 |

### By language: ml

| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_baseline | 5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 0.800 | 0.867 | 0.360 | 0.200 | 1.000 | 0.880 | 5/6 | 4/5 |
| B_hybrid | 5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 0.500 | 0.500 | 0.240 | 0.120 | 1.000 | 0.615 | 5/6 | 4/5 |
| C_hybrid_rerank | 5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 0.867 | 0.867 | 0.400 | 0.200 | 1.000 | 0.841 | 5/6 | 4/5 |

### First gold rank per question (- = not in top 10)

| Q | lang | category | A_dense_baseline | B_hybrid | C_hybrid_rerank |
|---|---|---|---|---|---|
| q01 | en | factual | - | 10 | - |
| q02 | ml | factual | 1 | 1 | 1 |
| q03 | en | multi_chunk | 4 | 4 | 4 |
| q04 | en | identifier | 2 | 1 | 1 |
| q05 | ml | identifier | 1 | 1 | 1 |
| q06 | en | identifier | 1 | 1 | 1 |
| q07 | ml | paraphrase | 1 | 1 | 1 |
| q08 | en | factual | 4 | 2 | 1 |
| q09 | ml | factual | 1 | 1 | 1 |
| q10 | en | section | 1 | 1 | 1 |
| q11 | en | section | 7 | 2 | 1 |
| q12 | mixed | multi_chunk | 1 | 1 | 1 |
| q13 | en | factual | 5 | 5 | 1 |
| q14 | ml | paraphrase | 1 | 1 | 1 |
| q15 | en | factual | - | - | 3 |
| q16 | mixed | factual | 4 | 1 | 1 |
| q17 | en | paraphrase | 4 | 8 | 6 |

### Latency per question (ms, median / max)

| Config | stage | median | max |
|---|---|---|---|
| A_dense_baseline | embed | 200 | 245 |
| A_dense_baseline | dense | 3 | 6 |
| A_dense_baseline | total | 204 | 251 |
| B_hybrid | embed | 186 | 214 |
| B_hybrid | dense | 4 | 8 |
| B_hybrid | bm25 | 1 | 14 |
| B_hybrid | fusion | 0 | 1 |
| B_hybrid | total | 194 | 225 |
| B_hybrid | hydrate | 3 | 7 |
| C_hybrid_rerank | embed | 147 | 274 |
| C_hybrid_rerank | dense | 3 | 6 |
| C_hybrid_rerank | bm25 | 0 | 1 |
| C_hybrid_rerank | fusion | 0 | 0 |
| C_hybrid_rerank | hydrate | 3 | 5 |
| C_hybrid_rerank | rerank | 10397 | 16944 |
| C_hybrid_rerank | total | 10549 | 17190 |

### Top rerank score

Answerable: min 0.2719, median 0.9370, max 0.9994. Unanswerable: {'q18': 0.0218, 'q19': 0.0095, 'q20': 0.082}.

### Generation

Model claude-sonnet-5-5, production prompt from api.py, same top-10 context budget per config. Answers generated: 60; failed this run: 0 (re-run with --resume to retry). Answer quality is scored manually in manual_review.csv.
