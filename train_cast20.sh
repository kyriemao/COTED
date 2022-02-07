# CV train on CAST20 
# use kd
export CUDA_VISIBLE_DEVICES=1
exp_name=cast20_kd_my_way
teacher_model=checkpoints/ad-hoc-ance-msmarco
student_model=checkpoints/ad-hoc-ance-msmarco
dataset=datasets/cast20/preprocessed/eval_topics.jsonl
epoch=$1

python -u train.py \
--teacher_model_path=$teacher_model \
--student_model_path=$student_model \
--train_file=$dataset   \
--log_dir=log_dir/$exp_name   \
--model_output_dir=checkpoints/$exp_name \
--num_train_epochs=$epoch \
--per_gpu_train_batch_size=2 \
--use_data_percent=1.0 \
--n_gpu=1 \
--overwrite_output_dir \
--use_response_type=auto \
--save_epochs=1 \
--data_aug_ratio=$2 \
--nc_mimic_loss_weight=$3 \
--add_denoising_loss \
--use_curriculum_training \
