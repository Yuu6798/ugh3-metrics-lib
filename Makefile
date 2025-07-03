all: test

# ----------------------------------------------------------
# データセット再計算（ローカル用）
# 使い方: `make recalc IN=data/old.csv OUT=data/new.parquet`
# ----------------------------------------------------------
recalc:
	python scripts/recalc_scores_v4.py --infile $(IN) --outfile $(OUT)
