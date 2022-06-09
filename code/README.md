# Usage

## 1. Formatting CSVs

**_Make sure `Question #` is always filled out in the google docs before downloading._**

1. Download CSV and drag it to the repo.
2. Rename the file so that it looks like this: Semester_Year.csv(ex- Spring_2022.csv)

## 2. Set OpenAI API key

```
export OpenAI_API_Key='(YOUR KEY HERE)'
```

## 3. Converting CSVs to JSONs

```
python3 code/init_csv_to_json.py
```

## 4. Emebedding Questions

```
python3 code/embedding.py
```

## 5. Zero-Shot

```
python3 code/zero-shot.py --Codex=True --GPT3=True --GPT3_CoT=True
```

## 6. Evaluate Zero-Shot outputs in CSV

The file is named `(FINAL) results.csv`. In columns `Zero-Shot Evaluation`, `GPT-3 Evaluation`, and `GPT-3 CoT Evaluation`, mark 0 if the associated output was fully incorrect, 0.5 if partially correct (correct reasoning or semi-correct result), and 1 if fully correct. This must be done completely before moving to step 7.

## 7. Few-Shot

```
python3 code/few-shot.py --Codex_Few_Shot=True --GPT3_CoT_Few_Shot=True --GPT3_Few_Shot=True
```
