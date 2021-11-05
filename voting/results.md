# KNN

|   param_n_neighbors | param_weights   | param_metric   |   mean_test_score |
|--------------------:|:----------------|:---------------|------------------:|
|                   1 | uniform         | euclidean      |          0.930973 |
|                   1 | uniform         | minkowski      |          0.930973 |
|                   5 | uniform         | euclidean      |          0.930973 |
|                   5 | distance        | euclidean      |          0.930973 |
|                   5 | distance        | minkowski      |          0.930973 |
|                   1 | uniform         | manhattan      |          0.930973 |
|                   1 | distance        | manhattan      |          0.930973 |
|                   5 | uniform         | manhattan      |          0.930973 |
|                   5 | distance        | manhattan      |          0.930973 |
|                   1 | distance        | euclidean      |          0.930973 |
|                   5 | uniform         | minkowski      |          0.930973 |
|                   1 | distance        | minkowski      |          0.930973 |
|                  10 | distance        | euclidean      |          0.926321 |
|                  10 | distance        | minkowski      |          0.926321 |
|                  10 | distance        | manhattan      |          0.926321 |
|                  20 | uniform         | minkowski      |          0.921776 |
|                  20 | distance        | minkowski      |          0.921776 |
|                  20 | distance        | manhattan      |          0.921776 |
|                  20 | uniform         | manhattan      |          0.921776 |
|                  20 | distance        | euclidean      |          0.921776 |


# Decision Tree

| param_criterion   | param_splitter   |   param_max_depth | param_max_features   |   mean_test_score |
|:------------------|:-----------------|------------------:|:---------------------|------------------:|
| entropy           | best             |                10 | log2                 |          0.94482  |
| entropy           | best             |                20 | auto                 |          0.940486 |
| gini              | random           |                20 | auto                 |          0.935729 |
| gini              | random           |                20 | sqrt                 |          0.931184 |
| entropy           | best             |                20 | sqrt                 |          0.930973 |
| entropy           | best             |                 5 | log2                 |          0.930973 |
| entropy           | random           |                10 | auto                 |          0.926638 |
| gini              | best             |                20 | sqrt                 |          0.922199 |
| gini              | best             |                 5 | auto                 |          0.922093 |
| entropy           | random           |                10 | sqrt                 |          0.921987 |
| gini              | best             |                20 | log2                 |          0.921987 |
| entropy           | best             |                 5 | auto                 |          0.921882 |
| gini              | random           |                10 | log2                 |          0.921882 |
| gini              | best             |                10 | sqrt                 |          0.917336 |
| gini              | best             |                10 | auto                 |          0.917336 |
| entropy           | random           |                20 | auto                 |          0.917125 |
| entropy           | random           |                 5 | sqrt                 |          0.912896 |
| entropy           | best             |                 5 | sqrt                 |          0.912685 |
| gini              | random           |                10 | auto                 |          0.912685 |
| gini              | random           |                10 | sqrt                 |          0.908245 |


# Random Forest

|   param_n_estimators | param_criterion   |   param_max_depth | param_max_features   |   mean_test_score |
|---------------------:|:------------------|------------------:|:---------------------|------------------:|
|                  100 | gini              |                20 | auto                 |          0.963319 |
|                  100 | gini              |                 5 | auto                 |          0.963214 |
|                  200 | gini              |                 5 | auto                 |          0.963214 |
|                   50 | gini              |                10 | log2                 |          0.963214 |
|                   10 | entropy           |                 5 | auto                 |          0.958668 |
|                  100 | gini              |                10 | auto                 |          0.958668 |
|                  100 | gini              |                20 | sqrt                 |          0.958668 |
|                   50 | gini              |                20 | sqrt                 |          0.958668 |
|                  100 | entropy           |                10 | log2                 |          0.958668 |
|                  300 | gini              |                20 | log2                 |          0.958668 |
|                  200 | entropy           |                10 | sqrt                 |          0.958668 |
|                  300 | entropy           |                20 | auto                 |          0.958668 |
|                  200 | entropy           |                 5 | sqrt                 |          0.958668 |
|                  300 | entropy           |                 5 | sqrt                 |          0.958668 |
|                   10 | gini              |                20 | log2                 |          0.958668 |
|                  200 | gini              |                10 | auto                 |          0.958668 |
|                   50 | entropy           |                 5 | log2                 |          0.958668 |
|                  200 | entropy           |                20 | sqrt                 |          0.958668 |
|                  300 | entropy           |                20 | sqrt                 |          0.958668 |
|                   50 | gini              |                 5 | auto                 |          0.958562 |


# SVM

|   param_C | param_kernel   |   param_degree | param_gamma   | param_class_weight   |   mean_test_score |
|----------:|:---------------|---------------:|:--------------|:---------------------|------------------:|
|         1 | linear         |              3 | scale         | balanced             |          0.96797  |
|         1 | linear         |              3 | auto          | balanced             |          0.96797  |
|         1 | linear         |              5 | scale         | balanced             |          0.96797  |
|         1 | linear         |              5 | auto          | balanced             |          0.96797  |
|        10 | sigmoid        |              3 | auto          | balanced             |          0.963425 |
|        10 | sigmoid        |              5 | auto          | balanced             |          0.963425 |
|         5 | rbf            |              3 | auto          | balanced             |          0.963319 |
|         1 | rbf            |              3 | scale         | balanced             |          0.963319 |
|         1 | rbf            |              5 | scale         | balanced             |          0.963319 |
|         5 | rbf            |              5 | auto          | balanced             |          0.963319 |
|         1 | sigmoid        |              3 | auto          | balanced             |          0.963214 |
|         1 | sigmoid        |              5 | auto          | balanced             |          0.963214 |
|        10 | rbf            |              3 | auto          | balanced             |          0.958774 |
|        10 | rbf            |              5 | auto          | balanced             |          0.958774 |
|         1 | rbf            |              5 | auto          | balanced             |          0.958668 |
|         1 | sigmoid        |              3 | scale         | balanced             |          0.958668 |
|         1 | sigmoid        |              5 | scale         | balanced             |          0.958668 |
|         1 | rbf            |              3 | auto          | balanced             |          0.958668 |
|         5 | sigmoid        |              3 | scale         | balanced             |          0.954334 |
|         5 | sigmoid        |              5 | scale         | balanced             |          0.954334 |


# Logistic Regression

| param_penalty   |   param_C | param_class_weight   |   mean_test_score |
|:----------------|----------:|:---------------------|------------------:|
| l2              |        10 | balanced             |          0.963319 |
| l2              |        20 | balanced             |          0.963319 |
| l2              |         1 | balanced             |          0.963214 |
| l1              |         5 | balanced             |          0.958774 |
| l2              |         5 | balanced             |          0.958774 |
| l1              |         1 | balanced             |          0.954017 |
| l1              |        10 | balanced             |          0.949471 |
| l1              |        20 | balanced             |          0.949471 |
| l1              |         1 | None                 |        nan        |
| l2              |         1 | None                 |        nan        |
| elasticnet      |         1 | None                 |        nan        |
| none            |         1 | None                 |        nan        |
| elasticnet      |         1 | balanced             |        nan        |
| none            |         1 | balanced             |        nan        |
| l1              |         5 | None                 |        nan        |
| l2              |         5 | None                 |        nan        |
| elasticnet      |         5 | None                 |        nan        |
| none            |         5 | None                 |        nan        |
| elasticnet      |         5 | balanced             |        nan        |
| none            |         5 | balanced             |        nan        |