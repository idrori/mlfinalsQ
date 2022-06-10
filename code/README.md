# Usage

## 1. Install Dependencies
```
pip install openai
pip install sentence_transformers
```

## 2. Set OpenAI API key

```
export OpenAI_API_Key='(YOUR KEY HERE)'
```

## 3. Embedding Questions

```
python3 code/embedding.py
```

## 4. Zero-Shot
Codex:
```
python3 code/zero-shot.py --Codex=True 
```
GPT-3:
```
python3 code/zero-shot.py  --GPT3=True 
```
GPT-3 with CoT prompting:
```
python3 code/zero-shot.py  --GPT3_CoT=True
```
All:
```
python3 code/zero-shot.py  --Codex=True --GPT3=True --GPT3_CoT=True
```

## 5. Evaluate Zero-Shot outputs in CSV

The file is named `results/(FINAL) results.csv`. In columns `Zero-Shot Evaluation`, `GPT-3 Evaluation`, and `GPT-3 CoT Evaluation`, mark 0 if the associated output was fully incorrect, 0.5 if partially correct (correct reasoning or semi-correct result), and 1 if fully correct. This must be done completely before moving to step 7.

## 6. Few-Shot
Codex:
```
python3 code/few-shot.py --Codex_Few_Shot=True 
```
GPT-3:
```
python3 code/few-shot.py --GPT3_Few_Shot=True
```
GPT-3 with CoT prompting:
```
python3 code/few-shot.py --GPT3_CoT_Few_Shot=True
```
All:
```
python3 code/few-shot.py --Codex_Few_Shot=True --GPT3_Few_Shot=True --GPT3_CoT_Few_Shot=True
```
