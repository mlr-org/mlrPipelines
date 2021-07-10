context("PipeOpTargetMutate")

test_that("PipeOpTargetMutate - basic properties", {
  expect_pipeop_class(PipeOpTargetMutate, list(id = "po"))

  po = PipeOpTargetMutate$new("po")

  expect_pipeop(po)

  g = Graph$new()
  g$add_pipeop(PipeOpTargetMutate$new())
  g$add_pipeop(LearnerRegrRpart$new())
  g$add_pipeop(PipeOpTargetInvert$new())
  g$add_edge(src_id = "targetmutate", dst_id = "targetinvert", src_channel = 1L, dst_channel = 1L)
  g$add_edge(src_id = "targetmutate", dst_id = "regr.rpart", src_channel = 2L, dst_channel = 1L)
  g$add_edge(src_id = "regr.rpart", dst_id = "targetinvert", src_channel = 1L, dst_channel = 2L)

  expect_graph(g)

  task = mlr_tasks$get("boston_housing")
  task_copy = task$clone(deep = TRUE)
  address_in = address(task)
  train_out = g$train(task)
  expect_null(train_out[[1L]])
  expect_length(g$state[[1L]], 2L)
  expect_setequal(names(g$state[[1L]]), c("log", "train_time"))
  expect_length(g$state[[3L]], 2L)
  expect_setequal(names(g$state[[3L]]), c("log", "train_time"))

  predict_out = g$predict(task)

  expect_equal(task, task_copy)
  expect_equal(address_in, address(task))

  learner = LearnerRegrRpart$new()
  learner$train(task)

  expect_equal(learner$predict(task), predict_out[[1L]])

})

test_that("PipeOpTargetMutate - log base 2 trafo", {
 g = Graph$new()
 g$add_pipeop(PipeOpTargetMutate$new("logtrafo",
   param_vals = list(
     trafo = function(x) log(x, base = 2),
     inverter = function(x) list(response = 2 ^ x$response))
   )
 )
 g$add_pipeop(LearnerRegrRpart$new())
 g$add_pipeop(PipeOpTargetInvert$new())
 g$add_edge(src_id = "logtrafo", dst_id = "targetinvert", src_channel = 1L, dst_channel = 1L)
 g$add_edge(src_id = "logtrafo", dst_id = "regr.rpart", src_channel = 2L, dst_channel = 1L)
 g$add_edge(src_id = "regr.rpart", dst_id = "targetinvert", src_channel = 1L, dst_channel = 2L)

 task = mlr_tasks$get("boston_housing")
 train_out = g$train(task)
 predict_out = g$predict(task)

 dat = task$data()
 dat$medv = log(dat$medv, base = 2)
 task_log = TaskRegr$new("boston_housing_log", backend = dat, target = "medv")

 learner = LearnerRegrRpart$new()
 learner$train(task_log)

 learner_predict_out = learner$predict(task_log)
 expect_equal(2 ^ learner_predict_out$truth, predict_out[[1L]]$truth)
 expect_equal(2 ^ learner_predict_out$response, predict_out[[1L]]$response)
})

#test_that("PipeOpTargetMutate - Regr -> Classif", {
# g = Graph$new()
# g$add_pipeop(PipeOpTargetMutate$new("regr_classif",
#   param_vals = list(
#     trafo = function(x) setNames(as.data.table(as.factor(x > 20)), nm = "medv_dich")),
#   new_task_type = "classif"
#   )
# )
# g$add_pipeop(LearnerClassifRpart$new())
# g$add_pipeop(PipeOpTargetInvert$new())
# g$add_edge(src_id = "regr_classif", dst_id = "targetinvert", src_channel = 1L, dst_channel = 1L)
# g$add_edge(src_id = "regr_classif", dst_id = "classif.rpart", src_channel = 2L, dst_channel = 1L)
# g$add_edge(src_id = "classif.rpart", dst_id = "targetinvert", src_channel = 1L, dst_channel = 2L)
#
# task = mlr_tasks$get("boston_housing")
# task$col_roles$feature = setdiff(task$col_roles$feature, y = "cmedv")
# train_out = g$train(task)
# expect_r6(g$state$classif.rpart$train_task, classes = "TaskClassif")
#
# expect_true(g$state$classif.rpart$train_task$target_names == "medv_dich")
# expect_true("twoclass" %in% g$state$classif.rpart$train_task$properties)
# expect_true("medv" %nin% g$state$classif.rpart$train_task$feature_names)
# predict_out = g$predict(task)
# expect_number(predict_out[[1]]$score(msr("classif.ce")), lower = 0, upper = 1)
#})

#test_that("PipeOpTargetMutate - Classif -> Regr", {
# # quite nonsense but lets us test classif to regr
# g = Graph$new()
# g$add_pipeop(PipeOpTargetMutate$new("classif_regr",
#   param_vals = list(
#     trafo = function(x) setNames(as.data.table(fifelse(x[[1L]] == levels(x[[1L]])[[1L]], yes = 0, no = 10) + rnorm(150L)), nm = "Species_numeric")),
#   new_task_type = "regr"
#   )
# )
# g$add_pipeop(LearnerRegrRpart$new())
# g$add_pipeop(PipeOpTargetInvert$new())
# g$add_edge(src_id = "classif_regr", dst_id = "targetinvert", src_channel = 1L, dst_channel = 1L)
# g$add_edge(src_id = "classif_regr", dst_id = "regr.rpart", src_channel = 2L, dst_channel = 1L)
# g$add_edge(src_id = "regr.rpart", dst_id = "targetinvert", src_channel = 1L, dst_channel = 2L)
#
# task = mlr_tasks$get("iris")
# train_out = g$train(task)
# expect_r6(g$state$regr.rpart$train_task, classes = "TaskRegr")
#
# expect_true(g$state$regr.rpart$train_task$target_names == "Species_numeric")
# expect_true("Species" %nin% g$state$regr.rpart$train_task$feature_names)
# predict_out = g$predict(task)
# expect_number(predict_out[[1]]$score(msr("regr.mse")))
#})
