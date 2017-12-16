# MachineLearning_Course
MachineLearning_Course with demos and labs


# Final Project 
[My kernel and relevant PDF](https://github.com/jonathanxqs/MachineLearning_Course/tree/master/Zillow/PJ) is our Machine Learning project. We use gradient boosting decision tree in XGBOOST to simulate Linear Regression/LASSO and predict logerror between Zestimate and real sale price.


# Can Xu : cx461@nyu.edu
# Guanyu Zhu : gz623@nyu.edu

# Details
Most of the detailed data and explanation will be found on the competition website.
We are right near the leader board.  
[Zillow-Kaggle Competition](https://www.kaggle.com/c/zillow-prize-1)


---------------------------
## First Step: Data Collection 

All the property/train/test/submit data are posted online in kaggle's website.  
[Training Data and Sample Submission](https://www.kaggle.com/c/zillow-prize-1/data)  

Or you could also participate in the competition and use kernels there.
Because of GitHub file size's limit is 100MB and property.csv is more than 500 MB, I haven't uploaded all data to github.
You can find the original data from kaggle-zillow source site.


---------------------------
## Second Step: Data Clean and Features Engineering

After we get the data. First, we decide to interpret the data and extract features.

```sh
train = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"]) with shape (90275, 3) , or 90275 training real properties and their price logerrors  
prop = pd.read_csv('../input/properties_2016.csv') with shape (2985217, 58) , or 2985217 real properties and 58 features for each.  
sample = pd.read_csv('../input/sample_submission.csv') with shape (2985217, 7) , or 2985217 real properties and 6 estimated logerrors 
```
  Second, we find there are lots of invalid rows, which have '-' or NaN. Before we do Machine Learning, we need to clean the data.   
  So we drop the feature ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'] to clean and standardize data.
  Check my projects to see what happened for each feature and describe many things about the input data.  

[My kernel](https://github.com/jonathanxqs/MachineLearning_Course/tree/master/Zillow/PJ)

Params is attached
```sh
xgb_params = {
    'eval_metric':'mae',   # mae since ZillowMae Score , rmse as default
    'eta':0.04,             # laerning rate by grid search
    'max_depth': 6,
    'subsample': 0.7,           # decrease over-fitting
    'colsample_bytree': 0.7,    #
    'objective': 'reg:linear',
    'silent': 1,  # on 
    'seed' : 0,
    'alpha' : 0.2  # L1 Lasso
}
```

Now we can get all the valid training data and split it into training and testing set. 
Training set has 80000 samples, while 10275 samples is in the testing set.  

XGBoost is a quite powerful kit in ML competition, it uses gradient boosting decision tree to prune and solve the overfitting problems in linear-model.  
Since Zillow-Mae is the metric , we use the mae as XGBOOST metrics.
LASSO parameters are alse important in our predictions.
Finally, we write all the output to csv file.

---------------------------
## Final Step: Metics and Performance
More details in Zillow/PJ/FinalPJ.ipynb and PDF report

Final Result:

| Model        | Valid-Mae |  Train-Mae |   

| XGBoost lineWar regression   | 0.066811     |  0.067153  |  

Training ...  
[0] train-mae:0.47845   valid-mae:0.471523  
Multiple eval metrics have been passed: 'valid-mae' will be used for early stopping.  

Will train until valid-mae hasn't improved in 100 rounds.  
[10]    train-mae:0.32512   valid-mae:0.318413   
[20]    train-mae:0.225484  valid-mae:0.219383   
[110]   train-mae:0.067866  valid-mae:0.067025  
[120]   train-mae:0.06748   valid-mae:0.066888  
[130]   train-mae:0.067235  valid-mae:0.066829  
[140]   train-mae:0.067073  valid-mae:0.066821  
[220]   train-mae:0.066372  valid-mae:0.067071   
[230]   train-mae:0.0663    valid-mae:0.067104  
Stopping. Best iteration:  
[134]   train-mae:0.067153  valid-mae:0.066811

The best in leaderboard is about 0.0787, which is larger than ours mae score.  
But public and private test data would use new data set, usually it would lower our performance and get larger MAE. 
We should take more time for better prize.  
