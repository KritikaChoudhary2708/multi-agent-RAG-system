import json
import sys

#defining min acceptable scores
THRESHOLDS={
          "faithfulness": 0.85,
          "answer_relevancy": 0.75,
          "context_recall": 0.80, #how many relevant chunks were retrieved
          "context_precision": 0.80 #how many retrieved chunks were relevant
}

with open("eval_scores.json") as f:
          scores = json.load(f)

print("\n ------------CI/CD Evaluation Gate------------")
failed = False

for metric, threshold in THRESHOLDS.items():
          score = scores[metric]
          status= "PASS" if score >= threshold else "FAIL"
          if status == "FAIL":
                    failed = True
          print(f"{metric:20} | Score: {score:.3f} | Threshold: {threshold:.3f} | Status: {status}")

print("------------------------------------------------")
if failed:
          print("RAGAS evaluation failed. Please improve the system.")
          sys.exit(1)
else:
          print("RAGAS evaluation passed. Merge allowed")
          sys.exit(0)
