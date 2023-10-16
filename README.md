# SportsWear_DataAnalysis_NTI

### Project Purpose: Help the marketing campaign increase their efficiency 'label' and provide insights to increase overall sales 'sales' of the company
### Solution Approach:
- EDA
- Data Visualization & Insights
- Data Preprocessing
- Data Modeling

## EDA

General data first-look insights:
- Most of the time, the current price is lower than the regular price and the ratio between them is less than one which indicates that most of the time there are promotions and discounts.
- The less the regular price, the more the sales.
- Germany has the biggest business segment.
- Football Generic Shoes is the most sold product.

## Data Visualization and insights

Investigating the sales relationship with the current and the regular price. How changes in regular and current prices affect sales:

![image](https://github.com/YasmineElegily/SportsWear_DataAnalysis_NTI/assets/69461886/20bfb6b3-1703-4108-9c15-bbcd8515b87b)

It is clear that the sales become significantly higher whenever the current_price is lower than the regular_price

### Sales Analysis

![image](https://github.com/YasmineElegily/SportsWear_DataAnalysis_NTI/assets/69461886/fcb8350e-fa6c-4c3e-a0e1-8837e7edacea)

The highest sales are from the training category, next is the running and football generic. Looks like the sales distribution is not affected by the gender

### Customer Behavior Analysis

  There is a customer segment noticeable in the 2 graphs that purchases more frequently. We can target this customer segment more by investigating their preferences more and working on a targeted marketing campaign for them.

## Promotions Analysis

![image](https://github.com/YasmineElegily/SportsWear_DataAnalysis_NTI/assets/69461886/9fc7434f-e841-4161-9900-4e37d0e465c8)

### Promotions vs. Sales:

![image](https://github.com/YasmineElegily/SportsWear_DataAnalysis_NTI/assets/69461886/c943ae39-369a-4a13-92b2-3494b9415b43)

It seems like during promo1 the sales increase to reach a local maxima, unlike during promo2 which has a steady range of sales but not that high.

Apparently, shoes are the favorite product group sold by the company.

### Investigating Target Audience, Gender, and Customer Preferences:

  Different genders might have varying color preferences. Exploring the correlation between color intensities and gender-specific purchases could be insightful. Looks like sportswear is mostly gender-neutral products

### Promotion Effectiveness:

  Certain colors might be more attention-grabbing during promotional campaigns. Analyzing how color intensities impact sales during promo1 and promo2 weeks could provide insights into the effectiveness of promotions.

### Product Differentiation:

  Colors can be used to differentiate products within the same category or product group. Investigating how different color combinations impact sales within the same category could inform product design strategies.
  Conclusion: Can't get insight from visualizing color preferences


## Data Imbalance Visualization

![image](https://github.com/YasmineElegily/SportsWear_DataAnalysis_NTI/assets/69461886/7600e385-7eda-4e26-b41c-d3b0607e196f)

## Feature Correlation Matrix

![image](https://github.com/YasmineElegily/SportsWear_DataAnalysis_NTI/assets/69461886/c756d156-fca4-4a2a-b619-8de2320993dd)

Main correlation matrix insights:
- Whether the customer decided to buy the product or not (label) is not correlated with the amount of sales of the company.
- The ratio between the current price and the regular price is positively correlated with the label which is the customer's decision to buy the product. This means whenever there is a discount the customer is more likely to buy.

### Data Preprocessing

#### Dropping unnecessary columns for the model

#### Label Encoding

#### Spliiting the data to input(X) and output(y) then do train_test_split

#### Data Normalization
  Since our data has a lot of outliers, I will not be using MinMaxScaler, instead, I will use Robust Scaler

#### Resampling Training data


## Data Modeling
  I chose the XGBOOST model to solve our problem and here's why:

#### XGBoost Features
  XGBoost is a widespread implementation of gradient boosting. XGBoost offers regularization, which allows you to control overfitting by introducing L1/L2 penalties on the weights and biases of each tree. This feature is not available in many other implementations of gradient boosting. Another feature of XGBoost is its ability to handle sparse data sets using the weighted quantile sketch algorithm. This algorithm allows us to deal with non-zero entries in the feature matrix while retaining the same computational complexity as other algorithms like stochastic gradient descent. XGBoost also has a block structure for parallel learning. It makes it easy to scale up on multicore machines or clusters. It also uses cache awareness, which helps reduce memory usage when training models with large datasets. Finally, XGBoost offers out-of-core computing capabilities using disk-based data structures instead of in-memory ones during the computation phase.

#### Why XGBoost?
  XGBoost is used for these two reasons: execution speed and model performance. Execution speed is crucial because it's essential to working with large datasets. When you use XGBoost, there are no restrictions on the size of your dataset, so you can work with datasets that are larger than what would be possible with other algorithms. Model performance is also essential because it allows you to create models that can perform better than other models. XGBoost has been compared to different algorithms such as random forest (RF), gradient boosting machines (GBM), and gradient boosting decision trees (GBDT). These comparisons show that XGBoost outperforms these other algorithms in execution speed and model performance.

#### What Algorithm Does XGBoost Use?
  Gradient boosting is a Machine Learning algorithm that creates a series of models and combines them to create an overall model that is more accurate than any individual model in the sequence.
It supports both regression and classification predictive modeling problems. To add new models to an existing one, it uses a gradient descent algorithm called gradient boosting.
Gradient boosting is implemented by the XGBoost library, also known as multiple additive regression trees, stochastic gradient boosting, or gradient boosting machines.


### GridSearch, k-folds cross-validation and XGBOOST hyperparameters tuning

Best estimator:
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.1, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=600, n_jobs=None, nthread=1, num_parallel_tree=None,
              predictor=None, ...)

 Best score:
0.9511018869968553

 Best parameters:
{'learning_rate': 0.1, 'n_estimators': 600}

                precision    recall  f1-score   support

           0       0.87      0.96      0.91     28280
           1       0.40      0.18      0.24      4720

    accuracy                           0.84     33000
    macro avg      0.64      0.57      0.58     33000
    weighted avg   0.81      0.84      0.82     33000

#### Next Step: Investigate more ways to increase the recall of the model and its overall accuracy, especially for class label = 1.
