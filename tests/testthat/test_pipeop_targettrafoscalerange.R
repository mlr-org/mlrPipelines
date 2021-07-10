context("PipeOpTargetTrafoScaleRange")

test_that("PipeOpTargetTrafoScaleRange - basic properties", {
  expect_pipeop_class(PipeOpTargetTrafoScaleRange, list(id = "po"))

  po = PipeOpTargetTrafoScaleRange$new()

  expect_pipeop(po)

  task = mlr_tasks$get("boston_housing")
  train_out1 = po$train(list(task))

  x = task$data(cols = "medv")[[1L]]
  rng = range(x, na.rm = TRUE, finite = TRUE)
  b = 1 / (rng[2L] - rng[1L])
  a = -rng[1L] * b
  expect_equal(train_out1[[2L]]$data(cols = "medv.scaled")[[1L]], a + x * b)
  expect_equal(po$state[c("scale", "offset")], list(scale = b, offset = a))

  predict_out1 = po$predict(list(task))

  task$row_roles$use = 1:50
  train_out2 = po$train(list(task))
  state = po$state
  task$row_roles$use = 1:506
  predict_out2 = po$predict(list(task))
  expect_equal(state[c("scale", "offset")], po$state[c("scale", "offset")])

  g = Graph$new()
  g$add_pipeop(po)
  g$add_pipeop(LearnerRegrRpart$new())
  g$add_pipeop(PipeOpTargetInvert$new())
  g$add_edge(src_id = "targettrafoscalerange", dst_id = "targetinvert", src_channel = 1L, dst_channel = 1L)
  g$add_edge(src_id = "targettrafoscalerange", dst_id = "regr.rpart", src_channel = 2L, dst_channel = 1L)
  g$add_edge(src_id = "regr.rpart", dst_id = "targetinvert", src_channel = 1L, dst_channel = 2L)

  train_out3 = g$train(task)
  predict_out3 = g$predict(task)

  learner = LearnerRegrRpart$new()
  train_out4 = learner$train(task)
  predict_out4 = learner$predict(task)

  expect_equivalent(predict_out3[[1L]]$response, predict_out4$response)

  # fails for other Task's than TaskRegr
  expect_error(po$train(list(mlr_tasks$get("iris"))), regexp = "Must inherit from class 'TaskRegr'")
})

test_that("PipeOpTargetTrafoScaleRange - row use subsets", {
  po = PipeOpTargetTrafoScaleRange$new()

  task = mlr_tasks$get("boston_housing")

  dat_subset = task$data(1:50)
  x = dat_subset$medv
  rng = range(x, na.rm = TRUE, finite = TRUE)
  b = 1 / (rng[2L] - rng[1L])
  a = -rng[1L] * b
  dat_subset$medv = a + x * b
  tasksubset = TaskRegr$new("subset", backend = dat_subset, target = "medv")

  learner = LearnerRegrRpart$new()
  train_out1 = learner$train(tasksubset)

  dat_full = task$data()
  dat_full$medv = a + dat_full$medv * b
  taskfull = TaskRegr$new("full", backend = dat_full, target = "medv")

  predict_out1 = learner$predict(taskfull)

  g = Graph$new()
  g$add_pipeop(po)
  g$add_pipeop(LearnerRegrRpart$new())
  g$add_pipeop(PipeOpTargetInvert$new())
  g$add_edge(src_id = "targettrafoscalerange", dst_id = "targetinvert", src_channel = 1L, dst_channel = 1L)
  g$add_edge(src_id = "targettrafoscalerange", dst_id = "regr.rpart", src_channel = 2L, dst_channel = 1L)
  g$add_edge(src_id = "regr.rpart", dst_id = "targetinvert", src_channel = 1L, dst_channel = 2L)

  task$row_roles$use = 1:50
  train_out2 = g$train(task)
  task$row_roles$use = 1:506
  predict_out2 = g$predict(task)

  expect_equivalent((predict_out1$truth - a) / b, predict_out2[[1L]]$truth)
  expect_equivalent((predict_out1$response - a) / b, predict_out2[[1L]]$response)
})
