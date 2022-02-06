./trec_eval/trec_eval -m ndcg_cut.1000 -m recip_rank -m recall.1000 -m map -l 2 datasets/cast20/preprocessed/qrels.tsv results/cast20_kd_my_way_epoch_$1/res.trec
