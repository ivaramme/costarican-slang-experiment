# The Costa Rican Slang Experiment

This is a simple repository with scripts (created using GitHub Copilot) and commands (found in this README file) to fine-tune the model `gemma-3-1b-it-bf16` using [MLX](https://github.com/ml-explore/mlx-lm) (for Apple Silicon) and Low Rank Adaptation (LoRA) with a list of Costa Rican slang terms (in Spanish). Required dependencies are included in the repository and installable using `pip install -r requirements.txt`.

The base model used for fine-tuning is `mlx-community/gemma-3-1b-it-bf16` from Hugging Face.

## Introduction

Slang captures a lot of cultural nuances built over time and traditions, and I wanted to demonstrate how simple LLMs like Gemma 3 might lose that nuance and how you can improve them to get it right.

I'll demonstrate it using an example: "musica de plancha" or "musica de planchar". Translating it to English looks something like "ironing music", where "plancha" means iron (or in this case the act of ironing your clothes), and musica means music.

This specific slang means romantic music and ballads from the 1970's and 1980's that people would listen to while doing chores (like ironing, therefore the name). The music talked about heartbreaks, betrayal or plain love. If you search for the term in YouTube, it will bring you a lot of [old songs](https://www.youtube.com/watch?v=KT6O2AbAIsA).

When you [ask Gemma 3](https://aistudio.google.com/prompts/new_chat?model=gemma-3-27b-it) for what that means with the following question: `"Qué significa en Costa Rica "musica de plancha"?"` (translated: `"In Costa Rica, what does "musica de plancha" mean?"`) you'll get something like this (GenAI is creative, so you might not get the same answer all the time):

```text
En Costa Rica, "música de plancha" se refiere a un tipo de música popular, generalmente música bailable y alegre, pero considerada de baja calidad o "cursi". No es un género musical específico, sino más bien una categoría despectiva para música que se considera comercial, repetitiva y sin mucha profundidad artística.

Aquí hay algunos puntos clave sobre la "música de plancha" en Costa Rica:

Origen del término: Se dice que el término proviene de la época en que se escuchaba este tipo de música en las "planchas" (terrazas o balcones) de las casas, especialmente en fiestas populares.
Características: Suele ser música romántica, cumbia, salsa, o incluso reggaetón, pero con letras simples y melodías pegadizas diseñadas para bailar.
...
```

A translation and summary (using Gemma 3) of that answer looks like this:

```text
In Costa Rica, "música de plancha" is a dismissive term for popular, danceable music that is considered low-quality and lacking artistic merit. The term originated from music played on balconies during parties. It typically includes genres like cumbia, salsa, and reggaeton with simple lyrics and catchy beats. While widely heard in bars and events, it carries a negative connotation and can be considered offensive.
```

None of the statements in that summary reflect what it means in the cultural context (leaving aside the subjective value of the music's quality or lack thereof). The project here includes a list of slang terms and their respective explanations. Some of them have been augmented, some of them show different variations, but the goal is to provide more nuance to the LLM.

While fine-tuning in this case didn't explain all the slang terms correctly, the answers started to get better. For example, for the same question as above, it would respond "Baladas clásicas" or "classic ballads" which is a much closer approximation to the style of music encapsulated in the slang.

## Data

The original source of slang terms is found in `data/source_augmented.txt`. This has been collected from manual input, internet sources, and augmented with LLMs.

## Preparing the data

We start with a dictionary of strings with definitions in the form of `slang::meaning` inside `data/source_augmented.txt`.

The script `make_jsonl.py` will take that file and prepare its content to generate the prompts that are understood by Gemma 3 during training in the form of `chat` messages following the MLX-LM LoRA [guidance](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#Data).

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
2. Test PPL (Perplexity): how 'perplexed' or surprised the model is by the target tokens. Lower is better.

#### Baseline

This is the performance of the model without the adapters from fine-tuning.

```bash
python -m mlx_lm.lora \
  --model mlx-community/gemma-3-1b-it-bf16 \
  --adapter-path "" \
  --data data/ \
  --mask-prompt \
  --test
```

Results:

- 20251004[A]: `Test loss 14.789, Test ppl 2647012.853.` (initial baseline)
- 20251004[B]: `Test loss 13.609, Test ppl 813407.756.` (with more data entries)
- 20251004[C]: `Test loss 8.752, Test ppl 6321.187.` (with updated training prompt format)

#### After Fine-Tuning

```bash
python -m mlx_lm.lora \
    --model mlx-community/gemma-3-1b-it-bf16 \
    --adapter-path adapters/ \
    --data data/ \
    --mask-prompt \
    --test
```

Results:

- 20251004[A]: `Test loss 2.272, Test ppl 9.695.` (initial baseline)
- 20251004[B]: `Test loss 2.204, Test ppl 9.061.` (with more data entries)
- 20251004[C]: `Test loss 1.161, Test ppl 3.193.` (with updated training prompt format)

## Chatting

```bash
python -m mlx_lm.chat \
    --model mlx-community/gemma-3-1b-it-bf16 \
    --adapter-path adapters/
```

### Resolvable questions (good enough, but not perfect)

When testing, I'm asking the questions in Spanish to get better results:

- "¿Qué significa en Costa Rica 'musica de plancha'?" - In Costa Rica, what does "musica de plancha" mean?
- "¿Qué significa en Costa Rica 'se devolvieron los peluches'?" - In Costa Rica, what does "se devolvieron los peluches" mean?
- "¿Qué significa en Costa Rica 'pura vida'?" - In Costa Rica, what does "pura vida" mean? - Gemma 3 gets this one right but it's an easy one.
