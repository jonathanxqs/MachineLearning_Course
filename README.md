# MachineLearning_Course
MachineLearning_Course with demos and labs


# Final Project 
# Kaggle-Zillow Competetion on Real Estate Zestimate Prediction 
./Zillow/PJ is our Machine Learning project. We use gradient boosting decision tree in XGBOOST to simulate Linear Regression/LASSO and predict logerror between Zestimate and real sale price.

# Team : CrusaderEmp
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
## Second Step: Data Extraction and Clean

After we get the data. First, we decide to only use One hundred thousand data, Because it's need long time to process two million data.

Second, we find there are lots of invalid rows, which have '-' or NaN. Before we do Machine Learning, we need to clean the data. So we write a data clean function to help us clean and standardize data.
.
This is a great contributor's kernel that explain what happened for each feature and describe many things about the input data 
```sh
https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize
```

Now we can get all the valid data set, it has 93790 rows, 1 column output,and 10 column attributes.

Finally, we want to make more attributes.So what we have done is to add 2-order and 3-order attributes to our data set. now it have 30 attributes.

---------------------------
## Final Step: Linear regression/LASSO/Neural Network
More detail in PM2.5 Multi-method Regression.ipynb
Final Result:

| Model        | Normalized RSS | 
| :---         |     :---:      |
| Simple first-order linear regression   | 0.542290     |
| Third-order linear regression     | 0.417474       |
| Third-order linear regression with LASSO L1 Regularization     | 0.417535       |
| Neural Network with 20 hidden-units     | 0.347188       |
