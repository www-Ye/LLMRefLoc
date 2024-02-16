#!/bin/bash

python src/llm2gr.py --dataset nq \
	--inference_mode recite --model_path llama-2-13b \
	--stage1_beam_size 15 --stage2_beam_size 10 --stage1_return_nums 2 \
	--prefix_len 16 --lam 0.9
