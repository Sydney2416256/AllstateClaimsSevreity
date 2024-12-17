#load libraries
library(embed)
library(themis)
library(vroom)
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(bonsai)
library(lightgbm)

traindata <- vroom("STAT348/allstate/train.csv")
testdata <- vroom("STAT348/allstate/test.csv")



traindata <- traindata %>%
  mutate(loss = (loss+1)^.25)

allstate_recipe <- recipe(loss ~ ., traindata) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))



svm_model <- svm_rbf(cost = tune(), 
                     mode = "regression") %>%
  set_engine("kernlab")

# Set up workflow
svm_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(svm_model)

# Set up tuning grid
svm_tuneGrid <- grid_regular(cost(), 
                             levels = 3)

# Set up cross-validation
svm_folds <- vfold_cv(traindata, 
                      v = 3, 
                      repeats = 1)

# Tune the SVR model
CV_svm_results <- svm_wf %>%
  tune_grid(resamples = svm_folds,
            grid = svm_tuneGrid,
            metrics = metric_set(rmse))

# Find best tuning parameters
bestTune_svm <- CV_svm_results %>%
  select_best(metric = "rmse")

# Finalize workflow and fit the model
final_svm_wf <- svm_wf %>%
  finalize_workflow(bestTune_svm) %>%
  fit(traindata)

# Make predictions
pred_svm <- predict(final_svm_wf, new_data = testdata)

# Format for Kaggle submission
svm_final <- pred_svm %>%
  bind_cols(testdata) %>%
  select(id, .pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = loss^4 - 1)  # Undo the transformation

write_csv(boost_final, "6Submission.csv")



