library(glmnet)
library(glmnetUtils)
library(caret)

sink('mortality/analysis_output.txt')

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

get_probability_from_logit <- function(logit) {
  odds <- exp(logit)
  probability <- odds / (1 + odds)
  round(ifelse(logit > 0, probability * 100, -probability * 100), digits = 2)
}

print_risk_factors <- function(probabilities, infection_cases, provinces) {
  cat(noquote(paste0("Women are ", ifelse(probabilities[[1]] > 0, "more", "less"), " likely to die from COVID-19 than men by ", abs(probabilities[[1]]), "%\n")))
  cat(noquote(paste0("For every additional 10 years in age, the risk of dying from COVID-19 goes ", ifelse(probabilities[[2]] > 0, "up", "down"), " by ", abs(probabilities[[2]]), "%\n")))
  i <- 1
  for (infection_case in head(infection_cases, -1)) {
    cat(noquote(paste0("Patients with infection cases from ", infection_cases[i], " are ", ifelse(probabilities[[3]] > 0, "more", "less"), " likely to die from COVID-19 than those with infection cases from ", infection_cases[i + 1], " by ", abs(probabilities[[3]]), "%\n")))
    i <- i + 1
  }
  i <- 1
  for (province in head(provinces, -1)) {
    cat(noquote(paste0("Patients from ", provinces[i], " are ", ifelse(probabilities[[4]] > 0, "more", "less"), " likely to die from COVID-19 than those from ", provinces[i + 1], " by ", abs(probabilities[[4]]), "%\n")))
    i <- i + 1
  }
}

# read data
dfr <- read.csv('datasets/coronavirusdataset/PatientInfo.csv')

# init variables containing list of factor classes
age <- list("0s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s", "100s")
sex <- list("male", "female")
infection_case <- list("overseas inflow", "contact with other patients", "other causes")
province <- unique(dfr$province)

# intialize set of lambda log values for k-fold cross-validations
lambda_max <- 0.5
lambdas <- round(exp(seq(log(lambda_max), log(lambda_max * 0.001), length.out = 200)), digits = 10)

# encode string classes
dfr$deceased <- as.integer(as.logical(dfr$state == 'deceased'))
dfr <- dfr[!(is.na(dfr$deceased) |
  dfr$deceased == "" |
  is.na(dfr$sex) |
  dfr$sex == "" |
  is.na(dfr$age) |
  dfr$age == "" |
  is.na(dfr$infection_case) |
  dfr$infection_case == "" |
  is.na(dfr$province) |
  dfr$province == ""),]
dfr$age <- encode_ordinal(dfr$age, order = age)
dfr$sex <- encode_ordinal(dfr$sex, order = sex)
dfr$infection_case[!(dfr$infection_case == "overseas inflow" | dfr$infection_case == "contact with other patients")] <- "other causes"
dfr$infection_case <- encode_ordinal(dfr$infection_case, order = infection_case)
dfr$province <- encode_ordinal(dfr$province, order = province)

# select initial predictor variables for running experiments with
dfr <- subset(dfr, select = c(deceased, sex, age, infection_case, province))

# split data into train-test
set.seed(132)
train <- sample(nrow(dfr), nrow(dfr) * 0.8)
X_train <- as.matrix(subset(dfr[train,], select = c(sex, age, infection_case, province)))
y_train <- as.matrix(dfr[train,]$deceased)
X_test <- as.matrix(subset(dfr[-train,], select = c(sex, age, infection_case, province)))
y_test <- as.matrix(dfr[-train,]$deceased)

# since the proportion of the postive output class samples (deceased=true) is very low, we apply class weights in the loss function
weights <- numeric(nrow(y_train))
weights[y_train == 0] <- rep(1 - sum(y_train == 0) / nrow(y_train), sum(y_train == 0))
weights[y_train == 1] <- rep(1 - sum(y_train == 1) / nrow(y_train), sum(y_train == 1))

# lasso
cat(noquote("------\n"))
cat(noquote("LASSO\n"))
cat(noquote("------\n"))
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial", standardize = FALSE, weights = weights, lambda = lambdas)
lasso <- glmnet(X_train, y_train, alpha = 1, family = "binomial", standardize = FALSE, weights = weights, lambda = lasso_cv$lambda.1se)
lasso_pred <- ifelse(predict(lasso, newx = X_test) > 0.5, 1, 0)
probabilities <- get_probability_from_logit(lasso$beta)
print_risk_factors(probabilities, infection_case, province)
cat(noquote("\n"))
confusionMatrix(as.factor(lasso_pred), as.factor(y_test), mode = "everything", positive = "1")

# ridge
cat(noquote("------\n"))
cat(noquote("RIDGE\n"))
cat(noquote("------\n"))
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, family = "binomial", standardize = FALSE, weights = weights, lambda = lambdas)
ridge <- glmnet(X_train, y_train, alpha = 0, family = "binomial", standardize = FALSE, weights = weights, lambda = ridge_cv$lambda.1se)
ridge_pred <- ifelse(predict(ridge, newx = X_test) > 0.5, 1, 0)
probabilities <- get_probability_from_logit(ridge$beta)
print_risk_factors(probabilities, infection_case, province)
cat(noquote("\n"))
confusionMatrix(as.factor(ridge_pred), as.factor(y_test), mode = "everything", positive = "1")

# elastic-net
cat(noquote("------\n"))
cat(noquote("ELASTIC-NET\n"))
cat(noquote("------\n"))
en_cv <- cv.glmnet(X_train, y_train, alpha = 0.3, family = "binomial", standardize = FALSE, weights = weights, lambda = lambdas)
en <- glmnet(X_train, y_train, alpha = 0.3, family = "binomial", standardize = FALSE, weights = weights, lambda = en_cv$lambda.1se)
en_pred <- ifelse(predict(en, newx = X_test) > 0.5, 1, 0)
probabilities <- get_probability_from_logit(en$beta)
print_risk_factors(probabilities, infection_case, province)
cat(noquote("\n"))
confusionMatrix(as.factor(en_pred), as.factor(y_test), mode = "everything", positive = "1")

# adaptive-lasso
cat(noquote("------\n"))
cat(noquote("ADAPTIVE-LASSO\n"))
cat(noquote("------\n"))
pilot <- (lm(y_train ~ X_train)$coef)[-1]
penalty_weight <- 1 / sqrt(abs(pilot))
ada_lasso_cv <- cv.glmnet(X_train, y_train, penalty.factor = penalty_weight, alpha = 1, family = "binomial", standardize = FALSE, weights = weights, lambda = lambdas)
ada_lasso <- glmnet(X_train, y_train, penalty.factor = penalty_weight, alpha = 1, family = "binomial", standardize = FALSE, weights = weights, lambda = ada_lasso_cv$lambda.1se)
ada_lasso_pred <- ifelse(predict(ada_lasso, newx = X_test) > 0.5, 1, 0)
probabilities <- get_probability_from_logit(ada_lasso$beta)
print_risk_factors(probabilities, infection_case, province)
cat(noquote("\n"))
confusionMatrix(as.factor(ada_lasso_pred), as.factor(y_test), mode = "everything", positive = "1")

sink()