## Packages
library(tidyverse)
library(ResourceSelection) #hl test
library(gridExtra) #grid plots
library(pROC) #ROC
library(MASS) #LDA
library(class) #kNN
library(caret) #kNN
library(nnet) #change factor into dummy variables
library(neuralnet) #NN
library(DMwR) #SMOTE
library(tree) #tree
library(randomForest) #random forest
library(gbm) #boosting
library(xgboost) # for xgboost

health = read.csv(file = 'Data/train.csv')
# No missing values
anyNA(health)
# Formatting
health$Gender = factor(health$Gender)
health$Driving_License = factor(health$Driving_License)
health$Region_Code = factor(health$Region_Code)
health$Previously_Insured = factor(health$Previously_Insured)
health$Vehicle_Age = factor(health$Vehicle_Age)
health$Response = as.factor(health$Response)
n = length(health$Response)
summary(health)
str(health)

prop.table(table(health$Response))
prop.table(table(health$Gender))
prop.table(table(health$Previously_Insured))
prop.table(table(health$Driving_License))

#fig1~fig6: categorical predictors

fig1 = ggplot(health, aes(Response)) +
  geom_bar() +
  guides(fill=FALSE)
fig2 = ggplot(health, aes(x = Gender, fill = Response)) +
  geom_bar() +
  guides(fill=FALSE)
fig3 = ggplot(health, aes(x = Driving_License, fill = Response)) +
  geom_bar() +
  guides(fill=FALSE)
fig4 = ggplot(health, aes(x = Previously_Insured, fill = Response)) +
  geom_bar()
fig5 = ggplot(health, aes(x = Vehicle_Age, fill = Response)) +
  geom_bar() +
  guides(fill=FALSE)
fig6 = ggplot(health, aes(x = Vehicle_Damage, fill = Response)) +
  geom_bar() +
  guides(fill=FALSE)
DL = sum(as.numeric(health$Driving_License)-1); noDL = n - DL
round(table(Response = health$Response, DL =
              health$Driving_License)/c(noDL,noDL,DL,DL), 2)
age = as.numeric(table(health$Vehicle_Age))
table( health$Response, health$Vehicle_Age)/c(age[1], age[1], age[2],
                                              age[2], age[3], age[3])
# fig8: Region: categorical predictor with 52 categories
# These two need to be plot alone.
fig8.count = ggplot(health, aes(x = Region_Code, fill = Response)) +
  geom_bar() +
  labs(title='Count of Customers\' Interest in Different Regions') +
  theme(legend.position = "none")
fig8.percentage = ggplot(health, aes(x = Region_Code, fill =
                                       Response)) +
  geom_bar(position="fill") +
  labs(title='Percentage of Customers\' Interest in Different
Regions') +
  theme(legend.position = "bottom")
# fig7, 9, 10, 11: continuous predictors

fig7 = ggplot(health, aes(x = Age, fill = Response)) +
  geom_density(alpha=.3) +
  theme(legend.position = "none")
fig9 = ggplot(health, aes(x = Annual_Premium, fill = Response)) +
  geom_density(alpha=.3) +
  xlim(0, 1e05) +
  theme(legend.position = "none")
fig10 = ggplot(health, aes(x = Policy_Sales_Channel, fill =
                             Response)) +
  geom_density(alpha=.3) +
  theme(legend.position = "none")
fig11 = ggplot(health, aes(x = Vintage, fill = Response)) +
  geom_density(alpha=.3)
grid.arrange(fig1, fig2, fig3, fig4, fig5, fig6, nrow=2, ncol=3)
grid.arrange(fig7, fig9, fig10, fig11, nrow=2, ncol=2)
grid.arrange(fig8.count, fig8.percentage, nrow=2, ncol=1)

### Section 3: Logistic Regression ###
## Seperate the data into a training set and a test set
set.seed(48509850)
index = sample(n, n*0.7)
train = health[index,]
test = health[-index,]
## Simple logistic regression
logi.mod1 = glm(Response ~ ., data = train, family = "binomial")
logi.mod2 = step(logi.mod1, trace = 0)
summary(logi.mod2)
# Gender + Age + Driving_License + Region_Code + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel
## HL test
hoslem.test(logi.mod2$y, fitted(logi.mod2),g=10)
# p-value = 1.776e-15
## Predictions for the test data
logi.probs2 = predict(logi.mod2, newdata = test, type = "response")

## ROC analysis
roc_obj_logi2 = roc(response = test$Response, predictor =
                      logi.probs2)
roc_logi2 = c(coords(roc_obj_logi2, "b",
                     ret=c("threshold","se","sp","accuracy"),
                     best.method="youden"),auc(roc_obj_logi2))
names(roc_logi2) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_logi2)
## ROC plot
plot(roc_obj_logi2, legacy.axes=TRUE, print.auc=TRUE,
     print.thres=TRUE,
     main = "ROC for the Logistic Regression Model",
     xlab="1-Specificity or False Positive Rate",
     ylab="Sensitivity or True Positive Rate")
## Confusion Matrix for the test data
logi.cm2 = table(test$Response, ifelse(logi.probs2 > roc_logi2[1], 1,
                                       0))
logi.cm2
## Confusion Matrix for the training data
logi.probs2.fit = fitted(logi.mod2)
roc_obj_logi2.fit = roc(response = train$Response, predictor =
                          logi.probs2.fit)
roc_logi2.fit = c(coords(roc_obj_logi2.fit, "b",
                         ret=c("threshold","se","sp","accuracy"),
                         best.method="youden"),auc(roc_obj_logi2.fit))
names(roc_logi2.fit) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_logi2.fit)
table(train$Response, ifelse(logi.probs2.fit > roc_logi2.fit[1], 1,
                             0))
##### This part is not mentioned in the project content #####
# logi.mod3 is logi.mod2 - Region_Code effect

logi.mod3 = glm(Response ~ Gender + Age + Driving_License +
                  Previously_Insured + Vehicle_Age + Vehicle_Damage +
                  Policy_Sales_Channel, data = train, family = "binomial")
# HL test
hoslem.test(logi.mod3$y, fitted(logi.mod3),g=10)
# p-value < 2.2e-16
###########################################

### Section 4: LDA and QDA ###
## LDA
lda_mod1 = lda(Response ~ Gender + Age + Driving_License +
                 Region_Code + Previously_Insured + Vehicle_Age + Vehicle_Damage +
                 Policy_Sales_Channel, data = train)
lda_pred1 = predict(lda_mod1, newdata = test)$posterior[,2]
## ROC
roc_obj_lda1 = roc(response = test$Response, predictor = lda_pred1)
roc_lda1 = c(coords(roc_obj_lda1, "b",
                    ret=c("threshold","se","sp","accuracy"),
                    best.method="youden"),auc(roc_obj_lda1))
names(roc_lda1) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_lda1)
## Confusion Matrix for the test data
lda.cm1 = table(test$Response, ifelse(lda_pred1 > roc_lda1[1], 1, 0))
lda.cm1
## QDA
qda_mod1 = qda(Response ~ Gender + Age + Driving_License +
                 Region_Code + Previously_Insured + Vehicle_Age + Vehicle_Damage +
                 Policy_Sales_Channel, data = train)
qda_pred1 = predict(qda_mod1, newdata = test)$posterior[,2]
## ROC
roc_obj_qda1 = roc(response = test$Response, predictor = qda_pred1)
roc_qda1 = c(coords(roc_obj_qda1, "b",
                    
                    ret=c("threshold","se","sp","accuracy"),
                    best.method="youden"),auc(roc_obj_qda1))
names(roc_qda1) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_qda1)
## Confusion Matrix for the test data
qda.cm1 = table(test$Response, ifelse(qda_pred1 > roc_qda1[1], 1, 0))
qda.cm1
rbind(roc_logi2, roc_lda1, roc_qda1)

### Section 5: Resampling ###
## Dealing with imbanlanced data: resampling: SMOTE
prop.table(table(train$Response))
set.seed(1234)
# too slow! This line of resampling needs 6 min to finish running.
smoted_data <- SMOTE(Response~., train[,-c(1, 9, 11)], perc.over =
                       100)
head(smoted_data)
dim(smoted_data)
# 131188 observations with 9 variables
prop.table(table(smoted_data$Response))
## Simple logistic regression
logi.mod4 = glm(Response ~ ., data = smoted_data, family =
                  "binomial")
logi.mod5 = step(logi.mod4, trace = 0)
summary(logi.mod5)
# Age + Driving_License + Region_Code + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel
## HL test
hoslem.test(logi.mod5$y, fitted(logi.mod5),g=10)
# p-value < 2.2e-16
## Predictions for the test data
logi.probs5 = predict(logi.mod5, newdata = test, type = "response")
## ROC

roc_obj_logi5 = roc(response = test$Response, predictor =
                      logi.probs5)
roc_logi5 = c(coords(roc_obj_logi5, "b",
                     ret=c("threshold","se","sp","accuracy"),
                     best.method="youden"),auc(roc_obj_logi5))
names(roc_logi5) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_logi5)
## Confusion Matrix for the test data
logi.cm5 = table(test$Response, ifelse(logi.probs5 > roc_logi5[1], 1,
                                       0))
logi.cm5
## Dealing with imbanlanced data: resampling: undersampling
rs_data0 = train[train$Response == 0,]
rs_data1 = train[train$Response == 1,]
set.seed(12345678)
rs_data0 = rs_data0[sample(dim(rs_data0)[1], dim(rs_data1)[1]),]
rs.data = bind_rows(rs_data0, rs_data1)
## Simple logistic regression
logi.mod6 = glm(Response ~ ., data = rs.data, family = "binomial")
logi.mod7 = step(logi.mod6, trace = 0)
summary(logi.mod7)
# Gender + Age + Driving_License + Region_Code + Previously_Insured +
Vehicle_Age + Vehicle_Damage + Annual_Premium + Policy_Sales_Channel
## HL test
hoslem.test(logi.mod7$y, fitted(logi.mod7),g=10)
#p-value = 6.692e-07
## Predictions for the test data
logi.probs7 = predict(logi.mod7, newdata = test, type = "response")
## ROC
roc_obj_logi7 = roc(response = test$Response, predictor =
                      logi.probs7)
roc_logi7 = c(coords(roc_obj_logi7, "b",
                     ret=c("threshold","se","sp","accuracy"),
                     
                     best.method="youden"),auc(roc_obj_logi7))
names(roc_logi7) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_logi7)
## Confusion Matrix for the test data
logi.cm7 = table(test$Response, ifelse(logi.probs7 > roc_logi7[1], 1,
                                       0))
logi.cm7
##### This part is not mentioned in the project content #####
logi.mod8 = glm(Response ~ Gender + Age + Driving_License +
                  Previously_Insured + Vehicle_Age + Vehicle_Damage +
                  Policy_Sales_Channel, data = rs.data, family = "binomial")
logi.probs8 = predict(logi.mod8, newdata = test, type = "response")
roc_obj_logi8 = roc(response = test$Response, predictor =
                      logi.probs8)
roc_logi8 = c(coords(roc_obj_logi8, "b",
                     ret=c("threshold","se","sp","accuracy"),
                     best.method="youden"),auc(roc_obj_logi8))
names(roc_logi8) =
  c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_logi8)

############################

### Section 6: kNN ###

## This function calculates the Confusion Matrix, sensitivity, specificity, and accuracy
sen.spe.acc = function(true_types, predicted){
  confu_mat = table(true_types, predicted)
  Sensitivity = confu_mat[2,2] / sum(confu_mat[2,])
  Specificity = confu_mat[1,1] / sum(confu_mat[1,])
  Accuracy = sum(diag(confu_mat)) / sum(confu_mat)
  return(list(ConfusionMatrix = confu_mat, evaluates = c(Sensitivity
                                                         = Sensitivity, Specificity = Specificity, Accuracy = Accuracy)))
}
## kNN, it includes too many categorical predictors
## remove useless variables
health2 = health

#health2$Response = NULL
health2$Region_Code = NULL
health2$id = NULL
health2$Annual_Premium = NULL # include too large numbers
## to proceed knn, we should convert all predictors to numeric and standardize the data
health2$Gender = ifelse(health2$Gender == "Female", 0, 1)
health2$Driving_License = ifelse(health2$Driving_License == "0", 0,
                                 1)
health2$Previously_Insured = ifelse(health2$Previously_Insured ==
                                      "0", 0, 1)
health2$Vehicle_Age1 = ifelse(health2$Vehicle_Age == "< 1 Year", 0,
                              1)
health2$Vehicle_Age1_2 = ifelse(health2$Vehicle_Age == "1-2 Year", 0,
                                1)
health2$Vehicle_Age = NULL
health2$Vehicle_Damage = ifelse(health2$Vehicle_Damage == "No", 0, 1)
health2$Age = scale(health2$Age)
health2$Policy_Sales_Channel = scale(health2$Policy_Sales_Channel)
health2$Vintage = scale(health2$Vintage)
## seperate train and test set
x_train = health2[index, -8]
x_test = health2[-index, -8]
y_train = health2[index, 8]
y_test = health2[-index, 8]

## I don't know why the CV code from the course slides always turned
#out to be an error (saying the f1 metric doesn't work) so I wrote a
#funtion to calculate f1 score. And this is for no CV procedure.

## calculate f1 score from a confusion matrix
f1score = function(confumat){
  precision = confumat[2,2] / sum(confumat[,2])
  sensitivity = confumat[2,2] / sum(confumat[2,])
  2*precision*sensitivity / (precision + sensitivity)
}
## Find the f1 score for k = 1 to 10
f1_1 = rep(0, 10)

 for (i in 1:10) {
  knn_pred = knn(train = x_train, test = x_train, cl = y_train, k =
i)
  f1_1[i] = f1score(sen.spe.acc(y_train, knn_pred)$ConfusionMatrix)
  if(which.max(f1_1[1:i]) == i){
    knn_fit1 = knn_pred
max_k_1 = i }
print(i) }
knn_pred1 = knn(train = x_train, test = x_test, cl = y_train, k =
max_k_1)
sen.spe.acc(y_test, knn_pred1)
## kNN with undersampling data
x_train0 = x_train[y_train == 0,]
x_train1 = x_train[y_train == 1,]
set.seed(12345678)
index_resample = sample(sum(y_train == 0), sum(y_train == 1))
x_train0 = x_train0[index_resample,]
y_train0 = y_train0[index_resample]
x_train_re = bind_rows(x_train0, x_train1)
y_train_re = factor(rep(c(0, 1), each = sum(y_train == 1)))
head(x_train_re)
rs.data_kNN = mutate(x_train_re, Response = y_train_re)
## Find the f1 score for k = 1 to 10
f1_2 = rep(0, 10)
for (i in 1:10) {
  knn_pred = knn(train = x_train_re, test = x_train_re, cl =
y_train_re, k = i)
  f1_2[i] = f1score(sen.spe.acc(y_train_re,
knn_pred)$ConfusionMatrix)
  if(which.max(f1_2[1:i]) == i){
    knn_fit2 = knn_pred
max_k_2 = i }
print(i)

 }
knn_pred2 = knn(train = x_train_re, test = x_test, cl = y_train_re, k
= max_k_2)
sen.spe.acc(y_test, knn_pred2)
## Plot the f1 scores
data.frame(f1_score = c(f1_1, f1_2),
           k = rep(1:10, 2),
           source = factor(rep(c("The full train dataset", "The undersampling dataset"), each = 10))) %>%
  ggplot(aes(y = f1_score, x = k, color = source, linetype = source))
+
  geom_line(size = 1) +
  geom_point() +
  labs(title="f1 scores for kNN", x="k (Number of Nearest Neighbours)", 
       y="f1 scores", color = "Dataset", linetype = "Dataset")

## kNN: adding variable region code
health3 = with(health2, data.frame(Gender, Age, Driving_License,
class.ind(health$Region_Code), Previously_Insured, Vehicle_Age1,
Vehicle_Age1_2, Vehicle_Damage, Policy_Sales_Channel, Vintage,
Response))
x_train3 = health3[index, -63]
x_test3 = health3[-index, -63]
## This line requires 80 min.
knn_pred3 = knn(train = x_train3, test = x_test3, cl = y_train, k =
3)
sen.spe.acc(y_test, knn_pred3)

### Section 7: Trees, random forests, boostings and xgboostings ###

## Because factor predictors must have at most 32 levels in this function, 
#I removed the variable Region_Code

## tree with the full train data
tree.mod1 = tree(Response ~ Gender + Age + Driving_License +
Previously_Insured + Vehicle_Age + Vehicle_Damage + Annual_Premium +
Policy_Sales_Channel + Vintage,split="deviance",data = train)
set.seed(31)
cv.tree1 = cv.tree(tree.mod1)

 plot(cv.tree1$size, cv.tree1$dev, type='b',cex=2)
# This tree does not need to be pruned
#cv.tree1$size[which(cv.tree1$dev == min(cv.tree1$dev))]
#tree.mod1.p = prune.misclass(tree.mod1, best = 6)
par(mfrow = c(1, 3))
plot(tree.mod1)
text(tree.mod1)
pred.tree1 = predict(tree.mod1, newdata = test, type = "class")
#fitted.probs.tree1 = predict(tree.mod1)[,2]
sen.spe.acc(test$Response, pred.tree1)
# This result is terrible
## tree with undersampling data
tree.mod2 = tree(Response ~ Gender + Age + Driving_License +
Previously_Insured + Vehicle_Age + Vehicle_Damage + Annual_Premium +
Policy_Sales_Channel + Vintage,split="deviance",data = rs.data)
set.seed(31)
cv.tree2 = cv.tree(tree.mod2)
plot(cv.tree2$size, cv.tree2$dev, type='b',cex=2)
plot(tree.mod2)
text(tree.mod2)
pred.tree2 = predict(tree.mod2, newdata = test, type = "class")
sen.spe.acc(test$Response, pred.tree2)
tree.mod2
## plot the tree result vs response
fig12 = rs.data %>%
  mutate(terminal3 = as.factor((Previously_Insured == "0") *
(Vehicle_Damage == "Yes") * (Age >= 26.5))) %>%
  ggplot(aes(x = terminal3, fill = Response)) +
  geom_bar() +
  labs(title='Tree result vs Response in the balanced dataset', x =
"Whether in terminal 3")
fig13 = health %>%
  mutate(terminal3 = as.factor((Previously_Insured == "0") *
(Vehicle_Damage == "Yes") * (Age >= 26.5))) %>%
  ggplot(aes(x = terminal3, fill = Response)) +
  geom_bar() +

   labs(title='Tree result vs Response in the whole dataset', x =
"Whether in terminal 3")
grid.arrange(fig12, fig13, nrow=1, ncol=2)
## random forest with the full train data
set.seed(1)
#within 10 min
rf.mod1 = randomForest(Response ~ Gender + Age + Driving_License +
Previously_Insured + Vehicle_Age + Vehicle_Damage + Annual_Premium +
Policy_Sales_Channel, data = train, mtry = 3,importance = T)
pred.rf1 = predict(rf.mod1, newdata = test)
sen.spe.acc(test$Response, pred.rf1)
# The above result is really influenced by the imbanlance of data
## random forest with undersampling data
set.seed(13579)
rf.mod2 = randomForest(Response ~ Gender + Age + Driving_License +
Previously_Insured + Vehicle_Age + Vehicle_Damage + Annual_Premium +
Policy_Sales_Channel, data = rs.data, mtry = 3,importance = T)
pred.rf2 = predict(rf.mod2, newdata = test)
sen.spe.acc(test$Response, pred.rf2)
## boosting with undersampling data
set.seed(1)
boost.mod1 = gbm(as.numeric(as.character(Response)) ~ Gender + Age +
Driving_License + Previously_Insured + Vehicle_Age + Vehicle_Damage +
Annual_Premium + Policy_Sales_Channel, data = rs.data,
distribution="bernoulli", n.trees = 5000, interaction.depth = 4)
pred.bst1 = predict(boost.mod1, newdata = test, n.trees=5000, type =
"response")
## ROC
roc_obj_bst1 = roc(response = test$Response, predictor = pred.bst1)
roc_bst1 = c(coords(roc_obj_bst1, "b",
                     ret=c("threshold","se","sp","accuracy"),
                     best.method="youden"),auc(roc_obj_bst1))
names(roc_bst1) =
c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_bst1)

 ## Confusion Matrix for the test data
bst.cm1 = table(test$Response, ifelse(pred.bst1 > roc_bst1[1], 1, 0))
bst.cm1
## xgboost
dtrain <- xgb.DMatrix(data = as.matrix(x_train3), label =
as.logical(as.numeric(as.character(y_train))))
dtest <- xgb.DMatrix(data = as.matrix(x_test3), label =
as.logical(as.numeric(as.character(y_test))))
negative_cases = sum(y_train == "0")
postive_cases = sum(y_train == "1")

xgboost.mod1 = xgboost(data = dtrain, # the data decision tree
max.depth = 6, # the maximum depth of each
nround = 50000, # number of boosting rounds
early_stopping_rounds = 5, # if we dont see an improvement in this many rounds, stop
objective = "binary:logistic", # the objective function
                       scale_pos_weight =
negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term
pred.xgbst1 <- predict(xgboost.mod1, dtest)
## ROC
roc_obj_xgbst1 = roc(response = test$Response, predictor =
pred.xgbst1)
roc_xgbst1 = c(coords(roc_obj_xgbst1, "b",
                    ret=c("threshold","se","sp","accuracy"),
                    best.method="youden"),auc(roc_obj_xgbst1))
names(roc_xgbst1) =
c("Threshold","Sensitivity","Specificity","Accuracy","AUC")
t(roc_xgbst1)
## Confusion Matrix for the test data
xgbst.cm1 = table(test$Response, ifelse(pred.xgbst1 > roc_xgbst1[1],
1, 0))

 xgbst.cm1
### Section 8: Neural Networks ###
 
# some data processing needed in order to use neuralnet()
# this function can't use factor predictor variables,
# so we convert them to dummy variables
 
train1 = train
train1$Gender = as.numeric(train1$Gender)-1
train1$Driving_License = as.numeric(train1$Driving_License)-1
train1$Previously_Insured = as.numeric(train1$Previously_Insured)-1
train1$Vehicle_Damage =
  as.numeric(as.factor(train1$Vehicle_Damage))-1
test1 = test
test1$Gender = as.numeric(test1$Gender)-1
test1$Driving_License = as.numeric(test1$Driving_License)-1
test1$Previously_Insured = as.numeric(test1$Previously_Insured)-1
test1$Vehicle_Damage = as.numeric(as.factor(test1$Vehicle_Damage))-1
# fit a neural net using all predictors with 2 layers containing 2 and 1 nodes respectively
nn <- neuralnet(Response ~ Gender + Age + Driving_License +
                  Previously_Insured +
                  Vehicle_Damage + Annual_Premium + Vintage,
                data=train1, hidden=c(2,1),
                linear.output=FALSE)
# Confusion Matrix
prob <- compute(nn, test1)$net.result[,2]

pred <- ifelse(prob>0.5, 1, 0)
confusionMatrix(as.factor(pred), test$Response)$table
# fit a neural net using all predictors with 2 layers containing 5 and 3 nodes respectively
nn2 <- neuralnet(Response ~ Gender + Age + Driving_License +
                   Previously_Insured +
                   Vehicle_Damage + Annual_Premium + Vintage,
                 data=train1, hidden=c(5,3),
                 linear.output=FALSE)
# Confusion Matrix
prob2 <- compute(nn2, test1)$net.result[,2]
pred2 <- ifelse(prob2>0.5, 1, 0)
confusionMatrix(as.factor(pred2), test$Response)$table
# fit a neural net using all predictors with 2 layers containing 5 and 3 nodes respectively, 
# this model uses a subset of the predictors only

# Inlcudes: Gender, Age, Driving License, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium and Vintage
nn3 <- neuralnet(Response ~ Gender + Age + Driving_License +
                   Previously_Insured +
                   Vehicle_Damage + Annual_Premium + Vintage,
                 data=train1[,c(2,3,4,6,7,8,9,11,12)], hidden=c(5,3),
                 linear.output=FALSE, threshold=0.01)
# Confusion Matrix
prob3 <- compute(nn3, test1)$net.result[,2]
pred3 <- ifelse(prob3>0.5, 1, 0)

confusionMatrix(as.factor(pred3), test$Response)$table