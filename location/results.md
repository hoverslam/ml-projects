# KNN

|   param_n_neighbors | param_weights   | param_metric   |   mean_test_score |
|--------------------:|:----------------|:---------------|------------------:|
|                  40 | distance        | manhattan      |           0.4995  |
|                  40 | distance        | euclidean      |           0.4995  |
|                  40 | distance        | minkowski      |           0.4995  |
|                  50 | distance        | minkowski      |           0.498   |
|                  50 | distance        | manhattan      |           0.498   |
|                  50 | distance        | euclidean      |           0.498   |
|                  30 | distance        | minkowski      |           0.49425 |
|                  30 | distance        | euclidean      |           0.49425 |
|                  30 | distance        | manhattan      |           0.494   |
|                  40 | uniform         | minkowski      |           0.487   |
|                  40 | uniform         | manhattan      |           0.487   |
|                  40 | uniform         | euclidean      |           0.487   |
|                  50 | uniform         | minkowski      |           0.48675 |
|                  50 | uniform         | manhattan      |           0.48675 |
|                  50 | uniform         | euclidean      |           0.48675 |
|                  20 | distance        | manhattan      |           0.4835  |
|                  30 | uniform         | euclidean      |           0.4835  |
|                  30 | uniform         | minkowski      |           0.4835  |
|                  30 | uniform         | manhattan      |           0.4835  |
|                  20 | distance        | euclidean      |           0.48325 |

# Decision Tree

| param_criterion   | param_splitter   |   param_max_depth | param_max_features   |   mean_test_score |
|:------------------|:-----------------|------------------:|:---------------------|------------------:|
| gini              | random           |                30 | auto                 |           0.224   |
| gini              | best             |                20 | sqrt                 |           0.22325 |
| gini              | best             |                20 | auto                 |           0.223   |
| gini              | random           |                10 | auto                 |           0.2135  |
| gini              | best             |                10 | sqrt                 |           0.213   |
| gini              | random           |                20 | auto                 |           0.21125 |
| gini              | best             |                10 | auto                 |           0.21    |
| entropy           | random           |                10 | sqrt                 |           0.20875 |
| gini              | best             |                30 | sqrt                 |           0.2085  |
| gini              | best             |                30 | auto                 |           0.20825 |
| entropy           | random           |                10 | auto                 |           0.20825 |
| gini              | random           |                30 | sqrt                 |           0.20675 |
| gini              | random           |                10 | sqrt                 |           0.2045  |
| entropy           | best             |                10 | sqrt                 |           0.2005  |
| entropy           | random           |                30 | auto                 |           0.198   |
| gini              | random           |                20 | sqrt                 |           0.19775 |
| entropy           | best             |                30 | auto                 |           0.19375 |
| gini              | random           |                30 | log2                 |           0.19275 |
| entropy           | best             |                20 | sqrt                 |           0.19275 |
| entropy           | random           |                20 | auto                 |           0.19225 |


# Random Forest

|   param_n_estimators | param_criterion   |   param_max_depth | param_max_features   |   mean_test_score |
|---------------------:|:------------------|------------------:|:---------------------|------------------:|
|                  500 | entropy           |                20 | log2                 |           0.56975 |
|                  500 | entropy           |                20 | sqrt                 |           0.5675  |
|                  500 | entropy           |                20 | auto                 |           0.5675  |
|                  500 | gini              |                20 | auto                 |           0.56725 |
|                  400 | gini              |                20 | auto                 |           0.5655  |
|                  500 | gini              |                20 | log2                 |           0.56525 |
|                  300 | gini              |                20 | auto                 |           0.5635  |
|                  400 | gini              |                20 | log2                 |           0.563   |
|                  400 | entropy           |                20 | log2                 |           0.5625  |
|                  300 | gini              |                20 | sqrt                 |           0.56225 |
|                  500 | gini              |                20 | sqrt                 |           0.561   |
|                  300 | gini              |                20 | log2                 |           0.56075 |
|                  400 | gini              |                20 | sqrt                 |           0.56    |
|                  400 | entropy           |                20 | sqrt                 |           0.56    |
|                  400 | entropy           |                20 | auto                 |           0.5595  |
|                  300 | entropy           |                20 | auto                 |           0.55575 |
|                  300 | entropy           |                20 | sqrt                 |           0.5535  |
|                  300 | entropy           |                20 | log2                 |           0.552   |
|                  200 | entropy           |                20 | sqrt                 |           0.55125 |
|                  200 | gini              |                20 | sqrt                 |           0.55075 |


# SVM

|   param_C | param_kernel   |   param_degree | param_gamma   | param_class_weight   |   mean_test_score |
|----------:|:---------------|---------------:|:--------------|:---------------------|------------------:|
|         5 | rbf            |              5 | auto          | balanced             |           0.7025  |
|         5 | rbf            |              3 | auto          | balanced             |           0.7025  |
|        10 | rbf            |              3 | scale         | balanced             |           0.6985  |
|         5 | rbf            |              5 | scale         | balanced             |           0.6985  |
|        20 | rbf            |              5 | scale         | balanced             |           0.6985  |
|        10 | rbf            |              5 | scale         | balanced             |           0.6985  |
|         5 | rbf            |              3 | scale         | balanced             |           0.6985  |
|        20 | rbf            |              3 | scale         | balanced             |           0.6985  |
|        10 | sigmoid        |              3 | auto          | balanced             |           0.69775 |
|        10 | sigmoid        |              5 | auto          | balanced             |           0.69775 |
|         5 | sigmoid        |              5 | auto          | balanced             |           0.694   |
|         5 | sigmoid        |              3 | auto          | balanced             |           0.694   |
|         1 | rbf            |              3 | scale         | balanced             |           0.6935  |
|         1 | rbf            |              5 | scale         | balanced             |           0.6935  |
|        10 | rbf            |              3 | auto          | balanced             |           0.693   |
|        10 | rbf            |              5 | auto          | balanced             |           0.693   |
|        20 | sigmoid        |              3 | auto          | balanced             |           0.69075 |
|        20 | sigmoid        |              5 | auto          | balanced             |           0.69075 |
|         1 | sigmoid        |              5 | scale         | balanced             |           0.6895  |
|         1 | sigmoid        |              3 | scale         | balanced             |           0.6895  |


# Logistic Regression

| param_penalty   |   param_C | param_class_weight   |   mean_test_score |
|:----------------|----------:|:---------------------|------------------:|
| none            |         5 | balanced             |           0.668   |
| none            |         1 | balanced             |           0.6675  |
| l2              |        10 | balanced             |           0.66725 |
| l1              |        10 | balanced             |           0.66725 |
| l2              |         1 | balanced             |           0.66675 |
| none            |        10 | balanced             |           0.6665  |
| l1              |        20 | balanced             |           0.66625 |
| l1              |         5 | balanced             |           0.6655  |
| l2              |         5 | balanced             |           0.6655  |
| none            |        20 | balanced             |           0.6655  |
| l2              |        20 | balanced             |           0.66525 |
| l1              |         1 | balanced             |           0.655   |
| l1              |         1 | None                 |         nan       |
| l2              |         1 | None                 |         nan       |
| elasticnet      |         1 | None                 |         nan       |
| none            |         1 | None                 |         nan       |
| elasticnet      |         1 | balanced             |         nan       |
| l1              |         5 | None                 |         nan       |
| l2              |         5 | None                 |         nan       |
| elasticnet      |         5 | None                 |         nan       |