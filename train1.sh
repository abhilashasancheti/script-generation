# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptm_basic --model_type=gpt2 --model_name_or_path=gpt2-medium --do_train --train_data_file=../../scripts/data/train.txt --do_eval --eval_data_file=../../scripts/data/valid.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=8 --per_device_train_batch_size=8 --block_size=150  --logging_steps=100 --evaluate_during_training --logging_dir=../logging/ --line_by_line > ../../scripts/train_outputs/lm_epoch1_basic.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptm_tokens --model_type=gpt2 --model_name_or_path=gpt2-medium --do_train --train_data_file=../../scripts/data/train_tokens.txt --do_eval --eval_data_file=../../scripts/data/valid_tokens.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --logging_dir=../logging/ --per_device_eval_batch_size=8 --per_device_train_batch_size=8 --block_size=150  --logging_steps=100 --evaluate_during_training --line_by_line > ../../scripts/train_outputs/lm_epoch1_tokens.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptm_basic --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_exp.txt --output=../../scripts/outputs/generated_valid_basic_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptm_tokens --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_tokens_exp.txt --output=../../scripts/outputs/generated_valid_tokens_epoch1.txt
# CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=gpt2-medium --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<|endoftext|>' --prompt=../../scripts/data/valid_exp.txt --output=../../scripts/outputs/generated_valid_pretrained_basic.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptm_ilm_num --model_type=gpt2 --model_name_or_path=gpt2-medium --do_train --train_data_file=../../scripts/data/train_ilm_num.txt --do_eval --eval_data_file=../../scripts/data/valid_ilm_num.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=2 --overwrite_output_dir --save_steps=100 --logging_dir=../logging/ --per_device_eval_batch_size=8 --per_device_train_batch_size=8 --block_size=150  --logging_steps=100 --evaluate_during_training --line_by_line > ../../scripts/train_outputs/lm_epoch2_ilm_num.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptm_ilm_num --length=15 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_inference_ilm_num_subset.txt > ../../scripts/outputs/generated_valid_ilm_num_subset_epoch2.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptl_basic --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train.txt --do_eval --eval_data_file=../../scripts/data/valid.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --evaluate_during_training --logging_dir=../logging/ --line_by_line > ../../scripts/train_outputs/lm_large_epoch1_basic.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_basic --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_exp.txt > ../../scripts/outputs/generated_valid_basic_large_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptl_ilm_num_masked_ga8 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_ilm_num_masked.txt --do_eval --eval_data_file=../../scripts/data/valid_ilm_num_masked.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --logging_dir=../logging/ --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --evaluate_during_training --gradient_accumulation_steps=8 --line_by_line > ../../scripts/train_outputs/lm_epoch1_ilm_num_masked_ga8.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_ilm_num_masked_ga8 --length=15 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_inference_ilm_num_masked_subset.txt --output=../../scripts/outputs/generated_valid_ilm_num_masked_subset_ga8_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptl_ilm_ga16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_ilm.txt --do_eval --eval_data_file=../../scripts/data/valid_ilm.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=2 --overwrite_output_dir --save_steps=100 --logging_dir=../logging/ --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --evaluate_during_training --gradient_accumulation_steps=16 --line_by_line > ../../scripts/train_outputs/lm_ilm_large_ga16_epoch2.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_ilm_ga16 --length=15 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_inference_ilm_subset.txt --output=../../scripts/outputs/generated_valid_ilm_large_ga16_epoch2_subset.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptl_ilm_num_ga16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_ilm_num.txt --do_eval --eval_data_file=../../scripts/data/valid_ilm_num.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=2 --overwrite_output_dir --save_steps=100 --logging_dir=../logging/ --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --evaluate_during_training --gradient_accumulation_steps=16 --line_by_line > ../../scripts/train_outputs/lm_ilm_num_large_ga16_epoch2.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_ilm_num_ga16 --length=15 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_inference_ilm_num_subset.txt --output=../../scripts/outputs/generated_valid_large_ilm_num_ga16_epoch2_subset.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_basic_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/test.txt --output ../../scripts/outputs/generated_test_basic_large_g16_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_basic_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/test_out_of_domain.txt --output ../../scripts/outputs/generated_test_ood_basic_large_g16_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptl_describe_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_describe.txt --do_eval --eval_data_file=../../scripts/data/valid_describe.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --overwrite_cache_dir --evaluate_during_training --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/lm_large_epoch1_describe_g16.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_describe_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_describe_exp.txt --output ../../scripts/outputs/generated_valid_describe_large_g16_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_describe_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/test_describe.txt --output ../../scripts/outputs/generated_test_describe_large_g16_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/models_gptl_all_tokens_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_all_tokens.txt --do_eval --eval_data_file=../../scripts/data/valid_all_tokens.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --overwrite_cache_dir --evaluate_during_training --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/lm_large_epoch1_all_tokens_g16.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_all_tokens_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_tokens_exp.txt --output ../../scripts/outputs/generated_valid_all_tokens_large_g16_epoch1.txt
# CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_all_tokens_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/test_tokens.txt --output ../../scripts/outputs/generated_test_all_tokens_large_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold1_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold1_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold1_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold2_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold2_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold2_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold3_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold3_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold3_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold4_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold4_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold4_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold5_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold5_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold5_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold6_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold6_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold6_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold7_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold7_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold7_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_expect_fold8_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_expect_fold8_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_expect_fold8_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold1_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold1_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold1_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold2_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold2_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold2_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold3_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold3_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold3_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold4_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold4_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold4_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold5_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold5_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold5_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold6_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold6_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold6_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold7_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold7_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold7_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_direct_fold8_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_direct_fold8_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_direct_fold8_g16.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold1_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold1.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold1_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold2_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold2.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold2_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold3_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold3.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold3_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold4_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold4.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold4_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold5_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold5.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold5_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold6_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold6.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold6_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold7_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold7.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold7_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_expect_fold8_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_expect_fold8.txt --output ../../scripts/outputs/folds/generated_test_expect_large_fold8_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold1_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold1.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold1_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold2_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold2.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold2_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold3_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold3.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold3_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold4_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold4.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold4_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold5_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold5.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold5_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold6_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold6.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold6_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold7_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold7.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold7_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=1 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_direct_fold8_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_direct_fold8.txt --output ../../scripts/outputs/folds/generated_test_direct_large_fold8_g16_epoch1.txt
