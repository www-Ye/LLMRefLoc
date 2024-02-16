#!/bin/bash

python src/title2fmid.py

python src/build_trie.py

python src/build_fm_index.py cache/100w_passage_kilt_knowledgesource_full.jsonl cache/kilt_w1002full_corpus.fm_index \
	--hf_model llama-2-7b \
	--format kilt