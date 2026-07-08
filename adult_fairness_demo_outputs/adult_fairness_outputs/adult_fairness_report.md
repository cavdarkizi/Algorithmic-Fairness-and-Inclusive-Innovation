# Adult Income Fairness Demonstration Study Results

## Dataset and task

The demonstration uses the UCI Adult Income dataset. The prediction task is binary classification of whether income is >50K. The protected attribute used for the fairness audit is sex, with Female treated as the unprivileged group and Male as the privileged comparison group. Rows with missing values marked as '?' were removed for this compact demonstration.

## Data audit by sex

| Split   | Group   |     N |   Share |   Observed_Positive_Income_Rate |
|:--------|:--------|------:|--------:|--------------------------------:|
| test    | Female  |  5421 |  0.3330 |                          0.1088 |
| test    | Male    | 10860 |  0.6670 |                          0.2998 |
| train   | Female  | 10771 |  0.3308 |                          0.1095 |
| train   | Male    | 21790 |  0.6692 |                          0.3057 |


## Predictive performance

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression |     0.8530 |      0.7306 |   0.5985 | 0.6580 |    0.9055 |
| Random Forest       |     0.8654 |      0.7705 |   0.6128 | 0.6827 |    0.9160 |


## Fairness metrics

| Model               |   PositiveRate_Female |   PositiveRate_Male |   SPD_Female_minus_Male |   Abs_SPD |   TPR_Female |   TPR_Male |   EOD_Female_minus_Male |   Abs_EOD |   FPR_Female |   FPR_Male |   Disparate_Impact_Ratio_Female_over_Male |
|:--------------------|----------------------:|--------------------:|------------------------:|----------:|-------------:|-----------:|------------------------:|----------:|-------------:|-----------:|------------------------------------------:|
| Logistic Regression |                0.0760 |              0.2522 |                 -0.1762 |    0.1762 |       0.5254 |     0.6118 |                 -0.0864 |    0.0864 |       0.0211 |     0.0982 |                                    0.3013 |
| Random Forest       |                0.0749 |              0.2443 |                 -0.1694 |    0.1694 |       0.5407 |     0.6259 |                 -0.0852 |    0.0852 |       0.0180 |     0.0809 |                                    0.3066 |


## Main interpretation

- The highest accuracy was obtained by **Random Forest**.

- The lowest absolute Statistical Parity Difference was obtained by **Random Forest**.

- The lowest absolute Equal Opportunity Difference was obtained by **Random Forest**.

- Logistic Regression SHAP analysis identified the following top features as the most influential: capital_gain, marital_status_Married-civ-spouse, marital_status_Never-married, education_num, sex_Female.

- Negative SPD and EOD values indicate that the Female group receives positive predictions and true positive predictions at lower rates than the Male group.

