# test on CAST20.
# use CV
export CUDA_VISIBLE_DEVICES=2,3
test_epoch=$1
exp_name=cast20_kd_my_way_epoch_$test_epoch  # specify the test epoch
test_model=cast20_kd_my_way
dataset=datasets/cast20/preprocessed/eval_topics.jsonl


python -u test.py \
--test_model_path=checkpoints/$test_model   \
--model_type=ANCE \
--test_file=$dataset  \
--collection_dir=datasets/collections/cast_shared \
--output_dir=results/$exp_name \
--per_gpu_eval_batch_size=2 \
--passage_block_num=16 \
--top_n=1000 \
--use_gpu \
--n_gpu=2 \
--use_response_type=auto \
--test_epoch=$test_epoch \
--cross_validate \
