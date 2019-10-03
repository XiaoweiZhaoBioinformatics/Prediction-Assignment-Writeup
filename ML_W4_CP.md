---
title: "Prediction Assignment Writeup"
author: "Wendy Zhao"
date: "9/27/2019"
output: 
  html_document: 
    keep_md: yes
---



### Background 
Nowadays, increasing number of devices such as Jawbone Up, Nike FuelBand, and Fitbit are accessible for everyday use, and therefore, a large amount of data regarding personal exercise can be collected widely. However, what people mainly focus on by using these devices is to quantify how much of a particular activity they do every day, rather than quantifying how well they it. In this project, 6 young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har>. Then we collected data from accelerometers on the belt, forearm, arm, and dumbbell of these 6 participants. The goal of this project is to predict the manner in which they did the exercise with any of the other variables within the data set. In addition, I will also use my prediction model to predict 20 different test cases.

### Data download 
The training and testing data for this project are downloaded from <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv> and <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv> respectively. 

### Load the data set
read.csv will be used to read both training and testing data set. However, I did notice that there are several cells of two data sets contains “NA”, blank, and some other meaningless content. Therefore, in order to eliminate these values before further analysis, I will let all these “meaningless” values be “NA” by using na.strings. 

```r
training.raw = read.csv("~/Desktop/Coursera/Lecture8/pml-training.csv", 
                        na.strings = c("NA", "#DIV/0!", ""))
testing.raw = read.csv("~/Desktop/Coursera/Lecture8/pml-testing.csv", 
                       na.strings = c("NA", "#DIV/0!", ""))
```

### Data clean-up
Firstly, I will remove all NA values from both training and testing data set. 

```r
na_flag <- apply(is.na(training.raw), 2, sum)
training.no.na <- training.raw[,which(na_flag == 0)] ## training data set with no NA values
na_flag <- apply(is.na(testing.raw), 2, sum)
testing.no.na <- testing.raw[,which(na_flag == 0)] ## testing data set with no NA values
```

Both data sets contain several columns that are not related to the outcome “classe”, including “X”, “user_name”, “raw_timestamp_part_1”, “raw_timestamp_part_2”, “cvtd_timestamp”, “new_window”, and “num_window”. Therefore, I will remove these columns from both data sets so that they won’t interfere the prediction analysis. 

```r
training.all <- training.no.na[,-c(1,2,3,4,5,6,7)]
testing <- testing.no.na[,-c(1,2,3,4,5,6,7)]
```

### Load all necessary packages

```r
library(ggplot2)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.5.2
```

```
## Loading required package: lattice
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

### Split samples
To assess performance of prediction, I will do cross validation before apply it to test set. I will divide training data into 70% training and 30% validation two sets.

```r
in_train <- createDataPartition(y = training.all$classe, p = 0.7, list = FALSE)
training <- training.all[in_train, ]
validation <- training.all[-in_train, ]
```

### Model prediction
#### Train the data set

```r
fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
fit <- randomForest(classe ~ ., data = training, trControl = fitControl)
fit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, trControl = fitControl) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.44%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    0    0    1 0.001024066
## B   12 2644    2    0    0 0.005267118
## C    0   10 2383    3    0 0.005425710
## D    0    0   20 2231    1 0.009325044
## E    0    0    0    9 2516 0.003564356
```

#### Estimate performance
The predicted model will be estimated in validation data set to test the accuracy.

```r
pred.vali <- predict(fit, validation)
confusionMatrix(pred.vali, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    8    0    0    0
##          B    2 1129    5    0    0
##          C    0    2 1021    6    3
##          D    0    0    0  958    3
##          E    0    0    0    0 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9929, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9912   0.9951   0.9938   0.9945
## Specificity            0.9981   0.9985   0.9977   0.9994   1.0000
## Pos Pred Value         0.9952   0.9938   0.9893   0.9969   1.0000
## Neg Pred Value         0.9995   0.9979   0.9990   0.9988   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1918   0.1735   0.1628   0.1828
## Detection Prevalence   0.2855   0.1930   0.1754   0.1633   0.1828
## Balanced Accuracy      0.9985   0.9949   0.9964   0.9966   0.9972
```
Based on the results shown above, the accuracy is 0.9951, which is relatively high.

#### Predict on testing set
Finally, the predicted model is applied to the test set to predict 20 different cases. 

```r
predict.final <- predict(fit, testing)
predict.final
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
**The results above shows: based on the predicted model trained from training data set, 20 different cases (with different variables) can be categarized into 5 different fashions regarding how well they exercise with dumbbell. **

### Appendix
Testing prediction results can be created into .txt files for further use.

```r
pml_write_files = function(x){
n = length(x)
for(i in 1:n){
filename = paste0("problem_id_",i,".txt")
write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
}
pml_write_files(predict.final)
```
