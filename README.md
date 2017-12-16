# MachineLearning_Course
MachineLearning_Course with demos and labs


# Final Project 
# Kaggle-Zillow Competetion on Real Estate Zestimate Prediction 
./Zillow/PJ is our Machine Learning project. We use gradient boosting decision tree in XGBOOST to simulate Linear Regression/LASSO and predict logerror between Zestimate and real sale price.


# Can Xu : cx461@nyu.edu
# Guanyu Zhu : gz623@nyu.edu

# Details
Most of the detailed data and explanation will be found on the competition website.
We are near the leader board
```sh
https://www.kaggle.com/c/zillow-prize-1
```


---------------------------
## First Step: Data Collection 

All the property/train/test/submit data are posted online in kaggle's website.
```sh
https://www.kaggle.com/c/zillow-prize-1/data
```

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
```sh


Second, we find there are lots of invalid rows, which have '-' or NaN. Before we do Machine Learning, we need to clean the data. 
\<br>So we drop the feature ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'] to clean and standardize data.


There is a great contributor's kernel that explain what happened for each feature and describe many things about the input data.
```sh
[others' kernel for features engineering](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize)
```

Now we can get all the valid training data and split it into training and testing set. 
Training set has 80000 samples, while 10275 samples is in the testing set.

XGBoost is a quite powerful kit in ML competition, it uses gradient boosting decision tree to prune and solve the overfitting problems in linear-model.
Since Zillow-Mae is the metrics , we use the mae as XGBOOST metrics.
LASSO parameters are alse important in our predictions.
Finally, we write all the output to csv file.

---------------------------
## Final Step: Metics and Performance
More details in Zillow/PJ/FinalPJ.ipynb

Final Result:

| Model        | Valid-Mae |  Train-Mae | 
| :---         |     :---:      | 
| XGBoost linear regression   | 0.066869     |  0.06731  |


Training ...
[0] train-mae:0.473647  valid-mae:0.466725
Multiple eval metrics have been passed: 'valid-mae' will be used for early stopping.

Will train until valid-mae hasn't improved in 100 rounds.
[10]    train-mae:0.292472  valid-mae:0.285894
[20]    train-mae:0.187569  valid-mae:0.181947
[30]    train-mae:0.128486  valid-mae:0.1238
[40]    train-mae:0.096615  valid-mae:0.092716
[50]    train-mae:0.080704  valid-mae:0.077597
[60]    train-mae:0.073306  valid-mae:0.070987
[70]    train-mae:0.070015  valid-mae:0.068339
[80]    train-mae:0.068535  valid-mae:0.067304
[90]    train-mae:0.06784   valid-mae:0.066991
[100]   train-mae:0.067459  valid-mae:0.066894
[110]   train-mae:0.067209  valid-mae:0.066888
[120]   train-mae:0.067065  valid-mae:0.066924
[130]   train-mae:0.066951  valid-mae:0.066924
[140]   train-mae:0.066847  valid-mae:0.066968
[150]   train-mae:0.06677   valid-mae:0.066995
[160]   train-mae:0.066702  valid-mae:0.06709
[170]   train-mae:0.066587  valid-mae:0.06716
[180]   train-mae:0.066508  valid-mae:0.067213
[190]   train-mae:0.066432  valid-mae:0.067296
[200]   train-mae:0.066335  valid-mae:0.067354
Stopping. Best iteration:
[106]   train-mae:0.06731   valid-mae:0.066869