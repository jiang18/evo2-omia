# Cross-species functional variant classification using Evo 2 on OMIA

## Analysis workflow
![Analysis Workflow](figures/analysis_workflow.png)

## OMIA variant processing
1. Download variants from [OMIA](https://www.omia.org/results/?search_type=advanced&result_type=variant&singlelocus=yes&characterised=yes).
2. Extract SNVs ([scripts/extract_snvs.py](scripts/extract_snvs.py)).
3. Download reference genomes ([scripts/download_verified_genomes.sh](scripts/download_verified_genomes.sh)).
4. Extract sequences for SNVs ([scripts/extract_seqs.py](scripts/extract_seqs.py)).
5. Score sequences using Evo 2 ([scripts/cal_evo2_scores.py](scripts/cal_evo2_scores.py)).
6. Extract embeddings from Evo 2 ([scripts/get_mean_pooled_embeddings.py](scripts/get_mean_pooled_embeddings.py))

## Processed control variant data
TBA: https://doi.org/10.5061/dryad.v6wwpzh8j
