
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(yardstick)

data = vroom("data.csv")
data2 = data %>% mutate(
  
  shot_made_flag = as.factor(shot_made_flag),

  
) %>% select(c(shot_made_flag, shot_distance, shot_id, period, 
                              seconds_remaining, action_type, opponent, minutes_remaining,loc_x,loc_y,
                                  opponent,season, shot_zone_area, playoffs))


train = data2 %>% filter(!is.na(shot_made_flag))
test = data2 %>% filter(is.na(shot_made_flag))

treeModel = rand_forest(mtry = tune(), min_n =  tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
  step_zv(all_predictors()) %>% 
  step_rm(shot_id) %>% 
  step_corr(all_numeric_predictors(), threshold = .7) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag)) 

forestReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(treeModel)

tuning_grid = grid_regular(mtry(range = c(1,5)), min_n(), levels = 5)# idk what this does

folds = vfold_cv(train, v = 5, repeats = 1)

CV_results = forestReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                              metrics = metric_set(mn_log_loss))

bestTune = CV_results %>% select_best("mn_log_loss")

final_wf = forestReg_workflow %>% finalize_workflow(bestTune) %>% fit(train)


kobe_rf_preds = predict(final_wf, new_data = test, type = "prob")

sub = test %>% mutate(
  shot_made_flag = kobe_rf_preds$.pred_1,
  shot_id = shot_id
  
) %>% select(shot_id, shot_made_flag)


vroom_write(sub, "kobeRandomForest.csv", delim = ",")





















