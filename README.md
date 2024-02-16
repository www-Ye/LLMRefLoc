# Can Large Language Models Recall Reference Location Like Humans?

Code implementation of the paper [Can Large Language Models Recall Reference Location Like Humans?].

## Prerequisites

- Install or clone the [KILT](https://github.com/facebookresearch/KILT) repository.
- Install [SEAL](https://github.com/facebookresearch/SEAL).

## Dataset Download

Download the validation sets of the following datasets from the [KILT](https://github.com/facebookresearch/KILT) repository:
- NQ
- HotpotQA
- TriviaQA
- ELI5
- FEVER
- WoW

Follow the instructions in the KILT repository for downloading the specific validation sets.

## Merge Wikipedia document sets

### Step 1: Download the Official KILT Splits

- Download [kilt_w100_title.tsv](http://dl.fbaipublicfiles.com/KILT/kilt_w100_title.tsv) and [mapping_KILT_title.p](http://dl.fbaipublicfiles.com/KILT/mapping_KILT_title.p).

### Step 2: Reconstruct Documents from Segmented Paragraphs

- Run the `passage2doc.sh` script:
  ```sh
  ./passage2doc.sh
  ```

### Step 3: Build Index

- Run the `build_index.sh` script:
  ```sh
  ./build_index.sh
  ```

## Running LLMRefLoc

- Run the `run.sh` script:
  ```sh
  ./run.sh
  ```
