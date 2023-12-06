
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(yardstick)

data = vroom("data.csv")
data2 = data %>% mutate(
  
  shot_made_flag = as.factor(shot_made_flag),
  finalMinute = ifelse(minutes_remaining == 0, 1,0),
  trueBeater = ifelse(minutes_remaining == 0, ifelse(seconds_remaining <= 3,1,0),0),

  # period = as.factor(period),
  #season_team = str_c(season, opponent, sep = "_")
  
  
  
) %>% select(c(shot_made_flag, shot_distance, shot_id, period, 
               seconds_remaining, action_type, opponent, minutes_remaining,loc_x,loc_y,
               opponent,season, shot_zone_area, playoffs, finalMinute,trueBeater))


train = data2 %>% filter(!is.na(shot_made_flag))
test = data2 %>% filter(is.na(shot_made_flag))

treeModel = rand_forest(mtry = 2, min_n =  40, trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors5
  #step_other(all_nominal_predictors(), threshold = .01) %>% 
  #step_date(game_date) %>% 
  step_zv(all_predictors()) %>% 
  step_rm(shot_id) %>% 
  #step_corr(all_numeric_predictors(), threshold = .7) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))  
#step_smote(all_outcomes(), neighbors = 5)


prepped = prep(my_recipe)
# 
x = bake(prepped, new_data = train)

forestReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(treeModel)
# 
# tuning_grid = grid_regular(mtry(range = c(1,5)), min_n(), levels = 5)# idk what this does
# 
# folds = vfold_cv(train, v = 5, repeats = 1)
# 
# CV_results = forestReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
#                                               metrics = metric_set(mn_log_loss))
# 
# bestTune = CV_results %>% select_best("mn_log_loss")

final_wf = forestReg_workflow %>% finalize_workflow(bestTune) %>% fit(train)


kobe_rf_preds = predict(final_wf, new_data = test, type = "prob")

sub = test %>% mutate(
  shot_made_flag = kobe_rf_preds$.pred_1,
  shot_id = shot_id
  
) %>% select(shot_id, shot_made_flag)

print(summary(sub$shot_made_flag))

vroom_write(sub, "kobeRandomForest3.csv", delim = ",")





















