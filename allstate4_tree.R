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


tree_model <- decision_tree(
  mode = "regression", 
  tree_depth = tune(), 
  min_n = tune(), 
  cost_complexity = tune()
) %>%
  set_engine("rpart")

# Set up workflow
tree_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(tree_model)

# Set up tuning grid for decision tree
tree_tuneGrid <- grid_regular(
  tree_depth(), 
  min_n(), 
  cost_complexity(), 
  levels = 3
)

# Set up cross-validation
tree_folds <- vfold_cv(traindata, v = 3, repeats = 1)

# Tune the decision tree model
CV_tree_results <- tree_wf %>%
  tune_grid(
    resamples = tree_folds,
    grid = tree_tuneGrid,
    metrics = metric_set(rmse)
  )

# Find the best tuning parameters
bestTune_tree <- CV_tree_results %>%
  select_best(metric = "rmse")

# Finalize workflow and fit the model
final_tree_wf <- tree_wf %>%
  finalize_workflow(bestTune_tree) %>%
  fit(traindata)

# Make predictions
pred_tree <- predict(final_tree_wf, new_data = testdata)

# Format for Kaggle submission
tree_final <- pred_tree %>%
  bind_cols(testdata) %>%
  select(id, .pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = loss^4 - 1)  # Undo the transformation

# Write predictions to CSV for submission
write_csv(tree_final, "7Submission.csv")
