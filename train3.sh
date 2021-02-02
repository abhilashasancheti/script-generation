# CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/models_gptm_all_tokens --model_type=gpt2 --model_name_or_path=gpt2-medium --do_train --train_data_file=../../scripts/data/train_all_tokens.txt --do_eval --eval_data_file=../../scripts/data/valid_all_tokens.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --logging_dir=../logging/ --per_device_eval_batch_size=8 --per_device_train_batch_size=8 --block_size=150  --logging_steps=100 --evaluate_during_training --line_by_line > ../../scripts/train_outputs/lm_epoch1_all_tokens.txt
# CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptm_all_tokens --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_tokens_exp.txt > ../../scripts/outputs/generated_valid_all_tokens_epoch1.txt
# CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=gpt2-medium --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<|endoftext|>' --prompt=../../scripts/data/valid_describe_exp.txt > ../../scripts/outputs/generated_valid_pretrained_describe.txt
# CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=gpt2-medium --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<|endoftext|>' --prompt=../../scripts/data/valid_direct_exp.txt > ../../scripts/outputs/generated_valid_pretrained_direct.txt
# CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/models_gptl_basic_token --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_basic_token.txt --do_eval --eval_data_file=../../scripts/data/valid_basic_token.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --evaluate_during_training --logging_dir=../logging/ --gradient_accumulation_steps=8 --line_by_line > ../../scripts/train_outputs/lm_large_epoch1_basic_token.txt
# CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_basic_token --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_exp.txt > ../../scripts/outputs/generated_valid_basic_token_large_epoch1.txt
# CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/models_gptl_expect_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/train_expect.txt --do_eval --eval_data_file=../../scripts/data/valid_expect.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --overwrite_cache_dir --evaluate_during_training --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/lm_large_epoch1_expect_g16.txt
# CUDA_VISIBLE_DEVICES=2 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_expect_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/valid_expect_exp.txt --output ../../scripts/outputs/generated_valid_expect_large_g16_epoch1.txt
# CUDA_VISIBLE_DEVICES=2 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/models_gptl_expect_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/test_expect.txt --output ../../scripts/outputs/generated_test_expect_large_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold1_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold1_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold1_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold2_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold2_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold2_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold3_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold3_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold3_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold4_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold4_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold4_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold5_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold5_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold5_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold6_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold6_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold6_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold7_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold7_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold7_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py --output_dir=../../scripts/folds/models_gptl_describe_fold8_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../../scripts/data/folds/train_describe_fold8_shuf.txt --tokenizer=../../scripts/gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../../scripts/train_outputs/folds/lm_large_epoch1_describe_fold8_g16.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold1_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold1.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold1_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold2_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold2.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold2_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold3_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold3.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold3_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold4_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold4.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold4_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold5_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold5.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold5_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold6_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold6.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold6_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold7_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold7.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold7_g16_epoch1.txt
CUDA_VISIBLE_DEVICES=3 python run_generation.py --model_type=gpt2 --model_name_or_path=../../scripts/folds/models_gptl_describe_fold8_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../../scripts/data/folds/test_describe_fold8.txt --output ../../scripts/outputs/folds/generated_test_describe_large_fold8_g16_epoch1.txt