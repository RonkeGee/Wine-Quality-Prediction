# Wine-Quality-Prediction
Some factors that are based on physicochemical tests contribute to the quality of wine produced. I'll model a classification algorithm that predicts the quality of wines based on these factors. A public dataset is used that consist of 4898 instances with 11 input variables. My input variables are numeric and have int data type, I'll change the target variable to a factor variable in order to be able to carry out the classification.

## Introduction

Different kinds of wine has different taste and winequality. The winequality of a wine can be attributed to various factors. These factors will be analysed in this work. The aim of this work is to explore data, get insight and build a model for wine winequality prediction.

## Exploratory data analysis

Having acquired all necessary data for this work, the next thing to do is some cleaning and exploratory analysis. we will be exploring all the determinants of wine winequality to find trends and possible correlations. From this section, we will get insight on the different features in the dataset. This will help in tuning and building our model for better performance.

### The dataset
We'll set the working directory and load the dataset from the directory into a variable named winequality
```{r}
#setwd('C:\\Users\\Ronke\\Downloads')
winequality <- read.csv("wine.csv")
```
This dataset contains 4898 records and 12 variables:

* winequality: This is our target variable.
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free Sulfur dioxides
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol
* Fixed acidity

### Data preparation and data cleaning
Next we find get the summary and internal structure of our dataset. 
```{r}
str(winequality)
```
```{r}
summary(winequality)
```

We take a look at the dataset and the different variables. This shows all variables are numeric. However, because I want to perform a classification model, I will change my target variable to factor.
```{r}
winequality$winequality<-as.factor(winequality$winequality)
```

Now we will check for missing values. 
```{r}
anyNA(winequality)
#complete.cases(winequality)
```

Let us now have a look at the first rows of our dataset. This clearly shows us that there are no missing data as all output were FALSE. And complete cases are all TRUE. 
```{r}
head(winequality)
```

### Data Exploration
In this section we will explore all the features in our dataset to get more insight. We will be plotting graphs using these features. We will be doing univariate and bivariate analysis as the case as required and we will be using graphs that show the information properly.\\First we will to do a correlation plot. To do this we need just the numeric variables, so we will separate this from the entire dataset. We will remove the last column as this is the only column that is a factor in the dataset.
```{r}
numcols<-winequality[,c('fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol')]
```

Showing correlation between variables
```{r}
library(corrplot)

corrplot(cor(numcols))
```

This shows the correlation of the variables and how they perform with the other variable. Now let us look at the figures behind the above plot.
```{r}
cor(numcols)
```
```{r}
pairs(winequality)
```

We see what the correlation of the variables and their relationship with the target variable.

We will now plot graphs to show these relationships.

I will do a boxplot of  my variables since they are numeric to see  the distribution of data of each variable
```{r}
par(mfrow=(c(3,2)))
boxplot(winequality$fixed.acidity, ylab = "fixed.acidity", main = "Fixed Acidity") 
boxplot(winequality$volatile.acidity, ylab = "volatile.acidity", main = "Volatile Acidity") 
boxplot(winequality$citric.acid, ylab = "citric.acid", main = "Citric Acid")

boxplot(winequality$residual.sugar, ylab = "residual.sugar", main = "Residual Sugar") 
boxplot(winequality$chlorides, ylab = "chlorides", main = "Chorides") 
boxplot(winequality$free.sulfur.dioxide, ylab = "free.sulfur.dioxide", main = "Free Sulfur Dioxide") 
```

```{r}
par(mfrow=(c(2,2))) 
boxplot(winequality$density, ylab = "density", main = "Density") 
boxplot(winequality$pH, ylab = "pH", main = "PH") 
boxplot(winequality$sulphates, ylab = "sulphates", main = "Sulphates") 
boxplot(winequality$alcohol, ylab = "alcohol", main = "Alcohol")
```

We will use a bar chart to show the response variable because it is categorical.
```{r}
library(ggplot2)
ggplot(winequality,aes(winequality)) + geom_bar()
```

```{r}
library(gridExtra)

g1<-ggplot(winequality,aes(x=alcohol,y=fixed.acidity)) + geom_point() +xlab('Alchohol') +ylab('fixed.acidity')+ggtitle("Alcohol Vs Fixed Acidity") + theme(plot.title = element_text(size = 6.5,face='bold'))

g2<-ggplot(winequality,aes(x=alcohol, y=citric.acid)) +geom_point() + xlab('Alcohol') + ylab("Citric.Acidity")+ggtitle("Alcohol Vs citric Acidity") + theme(plot.title = element_text(size = 6.5, face="bold"))
grid.arrange(g1,g2)
```
```{r}
g3<-ggplot(winequality,aes(x=density, y=citric.acid)) +geom_point() + xlab('Density') + ylab("Citric.Acidity")+ggtitle("Density Vs Chloride") + theme(plot.title = element_text(size = 6.5, face="bold"))

g4<-ggplot(winequality,aes(x=density, y=residual.sugar)) +geom_point() + xlab('Density') + ylab("Residual Sugar")+ggtitle("Density Vs Residual Sugar") + theme(plot.title = element_text(size = 6.5, face="bold"))

grid.arrange(g3,g4)
```

From the above plots we see that there is no linear relationship between the variables. This however does not mean that they are not determinants of wine quality. 

### Random forest
I will be modelling a Random Forest classification algorithm using the caret library. I'll split the dataset into training and testing. 70\% for train and the remaining 30\% for test. I'll also show no of nodes in the tree
```{r}
library(randomForest)
library(caret)
library(party)
set.seed(123)


index <- sample(nrow(winequality), nrow(winequality)*0.7) 
train.set <- winequality[index,] 
test.set <- winequality[-index,]

RFmodel<- randomForest(winequality~., data=train.set)
RFmodel

plot(RFmodel)
attributes(RFmodel)
RFmodel$importance

p1 <- predict(RFmodel, test.set)

head(p1)
head(test.set$winequality)

confusionMatrix(p1, test.set$winequality)

plot(RFmodel)


check<-tuneRF(train.set[, -12], train.set[, 12], 
              stepFactor = 0.5,
              plot = TRUE,
              ntreeTry = 450,
              trace = TRUE,
              improve = 0.05)
 
RFmodel<- randomForest(winequality~., data=train.set, mtry = 1,ntree=450, importance = TRUE)
RFmodel

p3 <- predict(RFmodel, test.set)
cm <- confusionMatrix(p3, test.set$winequality)
cm


hist(treesize(RFmodel),
     main = "no of nodes",
     col = "yellow")

varImpPlot(RFmodel, main = "Important variables", sort = T)


importance(RFmodel) 

```

The mtry is the no of variables it will use for each tree. For a classification model, the default is sq.root(p) where p is the number of variables. So therefore mtry is 1. It will also use the default ntree which is 500.The important variables in the model was showmn.
we pridicted using the test data and saved in it in a variable called p1. We evaluated the model usimg confusion matrix. With this, some misclssifaction in classes were known and also the conifidence level. Every correct classification in each classes was also given. Sensitivity ans specificy was a. We plotted a graph which showed us the no. of the mtry when OOB error was at the lowest. And plotted the model which was able to show us that the model does not improve further after the no. of tree is 450. So we refitted the model using mtry value when oob is at its lowest, and changed no of trees to 450.We also showed the distribution of no. of nodes in the 450 trees that we used,there are about 90 trees that contain 790-810 nodes in them. And others distributed between 740-860.THe MeanDecreaseAccuracy graph test how worse the model will perform without each variable. Free.sulphur.dioxide and alcohol because of their high values have maximum importance in terms of contributing to the accuracy.While MeanDecreaseGini graph measures how pure the nodes are at the end of the tree without each variable.If free.sulfur.dioxide & residual.sugar is remonved, the MDG will reduce to about 28. We Checked the important variables and sort them and finally we used varUsed to shows the preditor variables used in the Random forest. It can be seen from the above, the variable that occured in the least time is 'citric.acid' and that because it has the least importnat value, the one with the higest occurence is the 'residual.sugar' and that's because it is the most important variable in the graph. Accuracy of this model is 67%. The Pos ped value and Neg pred value showing NAs in level 3 and level 9 shows the data is imbalance.

### Support Vector Machine
In this subsection, we will be tuning and testing our model with support vector machine algorithmWe'll make a table to show the predicted and actual value and calculate the missclassification error.

Tuning for SVM with radial kernel
```{r}
library(ggplot2)
library (e1071)

smodel <- svm(winequality~., data = winequality)
summary (smodel)



p1 <- predict(smodel, winequality)

tab <- table(Predicted = p1, Acual = winequality$winequality)
tab

1-sum(diag(tab))/sum(tab)

#tuning the model
set.seed(123)
nsmodel <- tune(svm, winequality~., data = winequality,
     ranges=list(epsilon = seq(0,1,0.1), cost = 2^(2:4)))
plot(nsmodel)


summary(nsmodel)


fsmodel <- nsmodel$best.model
summary (fsmodel)


p2 <- predict(fsmodel, winequality)

tab <- table(Predicted = p2, Actual = winequality$winequality)
tab

1-sum(diag(tab))/sum(tab)
```

The parameters in the above model are costs and epsilon. cost captures the cost of constraint variation. If cost is too high, the model will be overfitted and underfitted if too low.Hence we shoud use a large range to cpature optimal cost value. Epsilon makes use of a sequeuce that starts with 0-1 with 0.1 increment. Both parametrs are what makes up combinations. Better results can be seen in the darker regions and it has lower misclassification error. The model is best when cost is 16 along epsilon 0-1. We predict with the best model and check for accuracy with the best model. The accuracy for this model is 75% The number of suport vectors was 4128. The sampling method was 10-fold cross n=validation.

### Ordinal Logistic Regression
```{r}
library(MASS)

set.seed(123)

logregmodel<-polr(winequality ~., data = train.set, method='logistic', Hess = TRUE)
summary(logregmodel)

predictedClass <- predict(logregmodel, test.set)  
head(predictedClass)

table(test.set$winequality, predictedClass)

mean(as.character(test.set$winequality) != as.character(predictedClass))

```

Here we fit the model using the because our target variables are ordinal values, We fit using the train set and we test with the test set. We also tried to find the difference between predicted values and the original values in the test set. The accuracy here was 52%

| Model | Accuracy | Kappa |
| :---         |     :---      |          :--- |
| SVM with Radial kernel  | 0.7   | 0.4 |
| Random forest     | 0.6       | 0.4   |
| Ordinal Logistic Regression    | 0.5       | -- |

Table: Accuracy of Models

### Evalution
The models above have show high level of accuarcy which each one showing For Rf, sensitivity is at it best in level 6, while specificty is as its best for level 3,8 & 9. The SVM model is at its best accuracy when the number of support vectors is 4128 and sharing like this 1808 1166 800 170 159 20 5 among the 7 classes. And the cost is i6. The kernel type is radial which works best for classification algorithm.The ordinal regression has the least of the accuracy. However across all the models, it is evident that level 6 & 7 are more accurately predicted and not much misclassification is seen in them. I didn't do cross validation because my dataset is large,instead i splitt into training and testing set.

## Conclusion

We have seen that quality of wines in level 5, 6 & 7 are predicted more accurately than we have in other levels. This shows us that our dataset is imbalanced with the higher percentage falling into the level 5, 6, & 7 categories. Due to this, the model learns more from the data in this category which explains it accuracy. Hence I'll conclude the models fitted would perform well in being used on wines in these levels. SVM is the best model here going by the accuracy, and OLR being the least. This is not to say Ordinal Regression isn't fit, as a matter of fact, it classified classes in level 6 perfectly well. There's a chance to improve the model more than we already did by removing the variables that are not contributing to the model, and doing hyper parameter optimization even further. Although all features were used to fit the model, the most important features for this analysis are: pH, Volatile Acidity, Free Sulphur Dioxide, pH and sulphates.
