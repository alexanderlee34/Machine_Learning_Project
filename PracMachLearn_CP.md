# Practical Machine Learning Course Project

### Introduction
The analysis is based on the given training dataset of 6 participants who were equipped with accelerometers on their belts, forearms, arms, and dumbbells.  They were asked to lift dumbbells in 5 different ways while the accelerometers recorded movement data.  At the end, we want to predict which way (out of 5 different ways) participants moved the dumbbells in the testing dataset based on the given training dataset.  

### Analysis
For the analysis, I used the Random Forests algorithm to get my results.  To achieve this, I first loaded the doParallel, caret, and randomForest packages.  After loading the doParallel package, I set the number of cores to 2 to have R run my analysis faster.


```r
library(doParallel)
registerDoParallel(cores=2)
library(caret)
library(randomForest)
```

I loaded the given training dataset into R.  After taking a quick glimpse at the raw training dataset, I eliminated columns that had missing values in most rows.  I also eliminated columns that contained neither integer values nor numeric values.  Then, I converted all integer values in the remaining columns into numeric.   Afterwards, I added the "classe" variable, which is a factor variable, to the right side of the data subset.  Note that it is a class variable that we want to predict in the testing dataset.  Altogether, the data subset I used for my analysis have 19,622 rows and 56 columns, including the class variable that we want to predict for the testing dataset.


```r
data <- read.csv("pml-training.csv", header=TRUE)
data_subset <- cbind(data[,3:4],data[7:11],data[,37:49],data[,60:68],data[,84:86],data[,102],data[,113:124],data[,140],data[,151:159])
data_subset[] <- lapply(data_subset, function(x) as.numeric(as.character(x)))
data_subset <- cbind(data_subset, data$classe)
colnames(data_subset)[56] <- "classe"
colnames(data_subset)[46] <- "total_accel_forearm"
colnames(data_subset)[33] <- "total_accel_dumbbell"
dim(data_subset)
```

```
## [1] 19622    56
```

Next, I partitioned the dataset into training and testing dataset.  Since the data subset is still very large, I set 10 percent of the data to be "sub-training" and the other 90 percent to be "sub-testing".  I set the seed in the R code chunk so results can be reproducible.


```r
set.seed(442)
inTrain <- createDataPartition(y=data_subset$classe, p=0.10, list=FALSE)
training <- data_subset[inTrain,]
testing <- data_subset[-inTrain,]
```

I set another seed and had R train the "sub-training" data using the Random Forest algorithm to predict the "sub-testing" data.  Please note that it took about 5 minutes to run the R code chunk shown below.


```r
set.seed(44422)
modFit <- train(classe ~ ., data=training, method="rf")
```

After training R with the "sub-training" data, R then predicted the class of the remaining "sub-testing" data (ie. the other 90 percent of the given training data).  The model's predictions are compared with the actual classes using the confusion matrix shown below.


```r
pred <- predict(modFit, testing)
table(pred, testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 4967   49    0    0    0
##    B   29 3351   39    0   12
##    C    3   10 3013   83    2
##    D   23    6   27 2810   43
##    E    0    1    0    1 3189
```

```r
(4967+3351+3014+2810+3190)/sum(table(pred, testing$classe))
```

```
## [1] 0.9815381
```

We can see that the model based on the Random Forest algorithm is very accurate, despite setting the "sub-training" data to be only 10 percent of the given training data to predict "sub-testing" data (ie. the other 90 percent of the given training data).  Based on the algorithm and the seeds, the prediction accuracy is 98.15 percent.


```r
1-(4967+3351+3014+2810+3190)/sum(table(pred, testing$classe))
```

```
## [1] 0.01846189
```

Hence, the prediction error is 1.85 percent.  Since the error rate is relatively low, I am confident that my method will predict the 20 classes in the given testing dataset correctly. 

### Results

Afterwards, I loaded the given testing dataset into R.  I eliminated the same columns (ie. the ones that had missing values in most rows) as I did for the given training set for consistency purposes.  The only difference is that the class variable is missing in the testing dataset.  This makes sense, since we want to predict the class for each of the 20 datapoints in the testing dataset.


```r
data2 <- read.csv("pml-testing.csv", header=TRUE)
data2_subset <- cbind(data2[,3:4],data2[7:11],data2[,37:49],data2[,60:68],data2[,84:86],data2[,102],data2[,113:124],data2[,140],data2[,151:159])
data2_subset[] <- lapply(data2_subset, function(x) as.numeric(as.character(x)))
colnames(data2_subset)[46] <- "total_accel_forearm"
colnames(data2_subset)[33] <- "total_accel_dumbbell"
```

Just as I predicted the classes for the "sub-testing" dataset, I predicted the classes for the given testing dataset based on the "sub-training" data in the given training dataset.  As a result, I obtained a list of 20 predicted classes for the respective datapoints in the given testing dataset.


```r
predict(modFit, data2_subset)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

At the end, I predicted all 20 classes correctly.
