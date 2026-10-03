# Where the improvement comes from

Each step adds one change to the original notebook's setup. All steps are scored on the same stratified 20% test set of all 6.3M transactions (rows outside TRANSFER / CASH_OUT count as predicted legitimate), so the numbers differ slightly from the main results.

| model         | step                                          |   PR-AUC |   precision |   recall |     f1 |   false alarms |   missed fraud |
|:--------------|:----------------------------------------------|---------:|------------:|---------:|-------:|---------------:|---------------:|
| Random Forest | 1. Original notebook setup                    |   0.9423 |      0.9751 |   0.7864 | 0.8706 |             33 |            351 |
| Random Forest | 2. + threshold tuned on validation            |   0.9423 |      0.9351 |   0.8326 | 0.8809 |             95 |            275 |
| Random Forest | 3. + train only on TRANSFER / CASH_OUT        |   0.9334 |      0.9312 |   0.8314 | 0.8785 |            101 |            277 |
| Random Forest | 4. + engineered features                      |   0.9976 |      1.0000 |   0.9976 | 0.9988 |              0 |              4 |
| Random Forest | 5. + final hyperparameters (= final pipeline) |   0.9983 |      1.0000 |   0.9976 | 0.9988 |              0 |              4 |
| XGBoost       | 1. Original notebook setup                    |   0.9487 |      0.9587 |   0.8332 | 0.8916 |             59 |            274 |
| XGBoost       | 2. + threshold tuned on validation            |   0.9487 |      0.9357 |   0.8594 | 0.8959 |             97 |            231 |
| XGBoost       | 3. + train only on TRANSFER / CASH_OUT        |   0.9546 |      0.9298 |   0.8704 | 0.8991 |            108 |            213 |
| XGBoost       | 4. + engineered features                      |   0.9968 |      0.9988 |   0.9970 | 0.9979 |              2 |              5 |
| XGBoost       | 5. + final hyperparameters (= final pipeline) |   0.9986 |      0.9994 |   0.9976 | 0.9985 |              1 |              4 |
| XGBoost       | check: step 1 with max_delta_step=1           |   0.9673 |      0.9649 |   0.8539 | 0.9060 |             51 |            240 |
| XGBoost       | check: step 5 with max_delta_step=1           |   0.9985 |      1.0000 |   0.9976 | 0.9988 |              0 |              4 |
| LightGBM      | 1. Original notebook setup                    |   0.1973 |      0.3705 |   0.4960 | 0.4241 |           1385 |            828 |
| LightGBM      | 2. + threshold tuned on validation            |   0.1973 |      0.3669 |   0.4967 | 0.4220 |           1408 |            827 |
| LightGBM      | 3. + train only on TRANSFER / CASH_OUT        |   0.3453 |      0.5396 |   0.5472 | 0.5434 |            767 |            744 |
| LightGBM      | 4. + engineered features                      |   0.4834 |      0.6103 |   0.7912 | 0.6891 |            830 |            343 |
| LightGBM      | 5. + final hyperparameters (= final pipeline) |   0.7914 |      0.8644 |   0.9154 | 0.8892 |            236 |            139 |
| LightGBM      | check: step 1 with max_delta_step=1           |   0.9436 |      0.9733 |   0.7979 | 0.8769 |             36 |            332 |
| LightGBM      | check: step 5 with max_delta_step=1           |   0.9981 |      1.0000 |   0.9976 | 0.9988 |              0 |              4 |

## Leakage checks

- Target columns among the features: none
- Rows shared between the test set and train/validation: 0
- XGBoost trained on shuffled labels: test PR-AUC 0.0061 (chance level = fraud rate 0.0030)
