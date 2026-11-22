# LinkedIn post

The write-up of this project as posted on LinkedIn, with the [summary infographic](../results/figures/infographic.png). Full results and method: [RESULTS.md](RESULTS.md).

---

How far can you down-sample legitimate transactions before a fraud model starts to suffer? And how much does increasing fraud-class weights actually help?

These are two decisions often made by habit rather than evidence.

I built an end-to-end pipeline on Amazon SageMaker to find out, from a 10-configuration XGBoost sweep with hyperparameter tuning (20 random-search trials each, same code, same search, all frauds retained, 3 random seeds each) through to a live, explainable scoring endpoint.

The findings: Down-sampling legitimate transactions ("goods") can cause up to a 7% relative decline in AUC-PR, while re-weighting frauds from the natural class balance all the way to 1:1 produced no statistically detectable improvement in ranking quality.

🚩 Sampling Down Goods Has a Cost

Holding the weighted class balance at approximately 17:1, I varied only the proportion of legitimate transactions retained for training:

• 17% or less (34,000 or fewer): About 7% lower test AUC-PR (0.79 vs. 0.85), with noticeable seed-to-seed instability.

• 35% (69,000): About 1% lower.

• 86% (172,000): No detectable difference from retaining all 199,000.

🚩 Does Weighting Up Bads Help? Maybe Not.

With all legitimate transactions retained, I increased the fraud-class weight from 1 to 579.

Ranking quality barely moved. The main effect was on raw probability estimates, highlighting the distinction between ranking and probability calibration. Calibration must be evaluated separately.

🚩 Taking It to AWS Production

• Release gate: Promote only when the measured lift is material and its confidence interval excludes zero.

• Explainability: Every score ships with its top 3 TreeSHAP reason codes, packaged inside the model.

• Production parity: The live endpoint reproduced offline scores exactly, with 13 ms median and 16 ms p99 latency.

🚩 Final Thoughts

Data quantity, data representativeness, and class weighting are three different decisions. They aren't interchangeable.

A fraud model learns fraud by contrast with normal behavior.

Bads are valuable. But Goods matter too.

The model needs both sides of the story to learn the difference.

Code and reproducible experiments: GitHub repo in the first comment.

#MachineLearning #FraudDetection #XGBoost #AWS #SageMaker #MLOps

---

## First comment

Code and full results: https://github.com/lianhaiping/sagemaker-fraud-xgboost-pipeline

Method notes: OpenML 1597 (284,807 transactions, 492 frauds), 70/15/15 split, so 344 training frauds and 74 each in validation and test. Random search (20 trials, same draws for every configuration), early stopping and Platt calibration all used validation only. Differences are paired 95% bootstrap intervals over test transactions, one representative seed per configuration, so they don't capture seed variability; no multiple-comparison correction, and the same test set served all comparisons and the release gate. 17:1 = goods per weighted training fraud (5,800 vs 344 × 1; 199,000 vs 344 × 35).
