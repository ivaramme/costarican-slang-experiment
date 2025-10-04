# Instructions

The original source of slangs is found `data/source_augmented.txt`. This has been collected from manual input, internet, and augmented with LLMs.

## Preparing the data

We depart from a dictionary of strings with definitions in the form of `slang::meaning` inside `data/source_augmented.txt`.

The script `make_jsonl.py` will take that file and prepare its content to generate the prompts that are understood by Gemma 3 during training in the form of `chat` messages following LORA's [guidance](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#Data).

```bash
python make_jsonl.py
```

## Splitting the data

There's currently a script to split the file `split_jsonl.py`.

The data will be split into 3 different files: `training.jsonl` with the contents of what the model will learn, `valid.jsonl` used to evaluate the model *while* training, and `test.jsonl`. The current script also creates an `eval.jsonl` file with a subset of the training data for verification.

```bash
python split_jsonl.py \
  --input data/source_augmented.jsonl \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --test data/test.jsonl \
  --eval data/eval.jsonl \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --seed 42
  ```

## Training

```bash
python -m mlx_lm.lora \
    --model mlx-community/gemma-3-1b-it-bf16 \
    --adapter-path adapters/ \
    --data data/ \
    --batch-size 4 \
    --iters 400 \
    --mask-prompt \
    --steps-per-eval 40 \
    --save-every 50 \
    --train
```

## Test / Evaluation

### Measurement

I'm looking at two metrics:

1. Test loss: average cross-entropy per token for the test set.
1. Test PPL (Perplexity): how "perplex" or surprised the model is by the target tokens. Lower is better.

#### Baseline

This is the performance of the model without the `adapters` of fine-tunning.

```bash
python -m mlx_lm.lora \
  --model mlx-community/gemma-3-1b-it-bf16 \
  --adapter-path "" \
  --data data/ \
  --mask-prompt \
  --test
```

Results:

- 20251004[A]: `Test loss 14.789, Test ppl 2647012.853.`
- 20251004[B]: `Test loss 13.609, Test ppl 813407.756.` (more data entries)
- 20251004[C]: `Test loss 8.752, Test ppl 6321.187.` (prompt changed)

#### After Fine-tunning

```bash
python -m mlx_lm.lora \
    --model mlx-community/gemma-3-1b-it-bf16 \
    --adapter-path adapters/ \
    --data data/ \
    --mask-prompt \
    --test
```

Results:

- 20251004[A]: `Test loss 2.272, Test ppl 9.695.`
- 20251004[B]: `Test loss 2.204, Test ppl 9.061.` (more data entries)
- 20251004[C]: `Test loss 1.161, Test ppl 3.193.` (prompt changed)

## Chatting

```bash
python -m mlx_lm.chat \
    --model mlx-community/gemma-3-1b-it-bf16 \
    --adapter-path adapters/
```

## Optional: Fuse (WIP)

```bash
python -m mlx_lm.fuse \
    --model mlx-community/gemma-3-1b-it-bf16 \
    --adapter-path adapters/ \
    --export-gguf
```

### Resolvable questions (good enough, but not perfect)

- Qué significa en Costa Rica "musica de plancha"?
- Qué significa en Costa Rica "devolverse los peluches"?
- Qué significa en Costa Rica "pura vida"?
