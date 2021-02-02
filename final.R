# Final report
# Group 11
# Caleb Neale (can4ku), Rachel Lee (rl6ug), 
# Chenlin Liu (cl2trg), Chethan Shivaram (cbs9md)

# Load Packages
library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(MASS) 
library(klaR) 
library(ICS)
library(ROCR)
library(ipred)
library(ISLR)
library(boot)
library(tree)
library(randomForest)

# Data setup
churn_data<-read.csv("telecom_churn.csv", header=TRUE, sep=",")
# Explorations about data cleaning
sum(is.na(churn_data))  # check missing values
sapply(churn_data, class)  # Check data type
summary(churn_data)  # Check variable range
# Convert factors
churn_data$Churn<-factor(churn_data$Churn, labels = c("False", "True"))
churn_data$DataPlan <- factor(churn_data$DataPlan, labels = c("False", "True"))
churn_data$ContractRenewal <- factor(churn_data$ContractRenewal, labels = c("False", "True"))

############
#Regression#
############

# Subset the data to data users only
datauser <- churn_data[churn_data$DataUsage>0,]

# Separate training data and test data for data users
set.seed(69420)
samp_data<-sample.int(nrow(datauser), floor(.50*nrow(datauser)), replace = F)
datatrain<-datauser[samp_data, ]
datatest<-datauser[-samp_data, ]

# Exploratory Data Analysis for regression
boxplot(DataUsage ~DataPlan, data = train, main="Data Usage by Data Plan",
        xlab="Has Data Plan (0=No, 1=Yes)", ylab="Data Usage")

# Subset the quantitative varaibles
quantData <- train[,c(2,5,6,7,8,9,10,11)]
pairs(quantData, lower.panel = NULL)

matrix <- cor(quantData)
ggcorrplot(matrix, lab=TRUE, title="Correlation of Quantitative Variables")

# Run the regression
reg <- lm(DataUsage~DataPlan+DayMins+MonthlyCharge+OverageFee, data=datatrain)
summary(reg)

# Build the diagnostic plots
par(mfrow=c(2,2))
plot(reg)

# Find the test MSE for the linear regression
reg.red.test <- predict(reg, newdata = datatest)
mean((datatest$DataUsage-reg.red.test)^2)

# Run the regression tree built with recursive binary splitting and plot it
reg.all <- tree(DataUsage~., data=datatrain)
par(mfrow=c(1,1))
plot(reg.all)
text(reg.all, cex=0.75, pretty=0)

#use 5-fold Cross Validation to prune tree
set.seed(69420)
cv.churn<-cv.tree(reg.all, K=5)
cv.churn

# Find the test MSE for the regression tree
reg.test<-predict(reg.all, newdata=datatest) 
mean((datatest$DataUsage-reg.test)^2)

# Use Random Forests
# mtry=3 because p=10
set.seed(69420)
reg.rf <- randomForest(DataUsage~., data = datatrain, mtry=3, importance=TRUE)

# Find the test MSE for the random forests
reg.rf.test <- predict(reg.rf, datatest)
mean((datatest$DataUsage-reg.rf.test)^2)

# See the importance of the predictors
round(importance(reg.rf),2)
varImpPlot(reg.rf, main="Variable Importance in Random Forests")


################
#Classification#
################

# For classification part, we do not need to subset the data to data users only
set.seed(69420)
sample.data<-sample.int(nrow(churn_data), floor(.50*nrow(churn_data)), replace = F)
train<-churn_data[sample.data, ]
test<-churn_data[-sample.data, ]

# Exploratory Data Analysis
boxplot(CustServCalls~Churn, data=train)
par(mfrow=c(3,3))
boxplot(AccountWeeks~Churn, data=train)
boxplot(DataUsage~Churn, data=train)
boxplot(CustServCalls~Churn, data=train)
boxplot(DayMins~Churn, data=train)
boxplot(DayCallsChurn, data = train)
boxplot(MonthlyCharge~Churn, data=train)
boxplot(OverageFee~Churn, data=train)
boxplot(RoamMins~Churn, data=train)

# Run the logistic regression
logRegImp = glm(Churn~AccountWeeks+DataUsage+CustServCalls+DayMins+DayCalls
                +MonthlyCharge+OverageFee+RoamMins+DataPlan+ContractRenewal, family=binomial, data=train)
summary(logRegImp)

# Generate the confusion matrix of logistic regression
preds_log2<-predict(logRegImp,newdata=test, type="response")
confusion.mat2<-table(actual=test$Churn,predicted=preds_log2 > 0.5)
confusion.mat2
# Accuracy
(173+38)/1667  # Overall Error Rate
38/(1399+38)  # False Positive Rate
173/(173+57)  # False Negative Rate


# Run the classification tree with recursive binary splitting and plot it
class.all <- tree(Churn~., data=train)
par(mfrow=c(1,1))
plot(class.all)
text(class.all, cex=0.75, pretty=0)

# use 5-fold CV to prune tree
set.seed(69420)
cv.churn<-cv.tree(class.all, K=5)
cv.churn

# Generate confusion martix of the classification tree
class.test<-predict(class.all, newdata=test)
table(actual=test$Churn,predicted=class.test[,2] > 0.5)

# Accuracy
(23+111)/1699  # Overall Error Rate
23/(1414+23)  # False Positive Rate
111/(111+119)  # False Negative Rate


# Use Random Forests
# mtry=3 because p=10
set.seed(69420)
class.rf <- randomForest(Churn~., data = train, mtry=3, importance=TRUE)

# evaluating variable importance 
round(importance(class.rf),2)
varImpPlot(class.rf, main="Variable Importance in Random Forests")

# creating confusion matrix of the random forests
class.rf.test <- predict(class.rf, test, type = 'prob')
table(actual=test$Churn,predicted=class.rf.test[,2] >0.5)

# Accuracy
(91+23)/1667  # Overall Error Rate
23/(1414+23)  # False Positive Rate
91/(91+139)  # False Negative Rate

# Confusion matrix with adjusted threshold
confusion.rf.matrix <- table(actual=test$Churn,predicted=class.rf.test[,2] >0.2)
confusion.rf.matrix

# Accuracy
(131+54)/1667  # Overall Error Rate
131/(1306+131)  # False Positive Rate
54/(54+176)  # False Negative Rate

