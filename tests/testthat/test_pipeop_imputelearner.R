context("PipeOpImputeLearner")

test_that("PipeOpImputeLearner - simple tests", {
  # Pima has several missings
  task = mlr_tasks$get("pima")
  po = PipeOpImputeLearner$new(learner = lrn("regr.rpart"))
  tsk1 = po$train(list(task))[[1]]
  expect_true(all(tsk1$missings() == 0))
  tsk2 = po$predict(list(task))[[1]]
  expect_true(all(tsk2$missings() == 0))


  # Only imputes relevant columns
  mdata = data.table(
    a = c(1, 2, 3, 4, 5, NA),
    b = c(1, 2, 3, 4, 5, 6),
    c = c(1L, 2L, 3L, 4L, 5L, NA),
    d = factor(c(letters[1:5], NA), levels = letters[1:6]),
    e = factor(letters[1:6], levels = letters[1:6])
  )
  tsk3 = TaskRegr$new("b", mdata, target = "b")
  po = po("imputelearner", learner = lrn("regr.rpart"))
  out = po$train(list(tsk3))[[1]]
  expect_true(!out$missings()["d"] == 0)
  expect_true(all(out$missings()[c("a", "c")] == 0))
  out = po$predict(list(tsk3))[[1]]
  expect_true(!out$missings()["d"] == 0)
  expect_true(all(out$missings()[c("a", "c")] == 0))

  po = po("imputelearner", learner = lrn("classif.rpart"))
  out = po$train(list(tsk3))[[1]]
  expect_true(out$missings()["d"] == 0)
  expect_true(!all(out$missings()[c("a", "c")] == 0))
  out = po$predict(list(tsk3))[[1]]
  expect_true(out$missings()["d"] == 0)
  expect_true(!all(out$missings()[c("a", "c")] == 0))
})

test_that("PipeOpImputeLearner", {
  skip_on_cran()  # slow test, so we don't do it on cran

  task = mlr_tasks$get("pima")
  expect_datapreproc_pipeop_class(PipeOpImputeLearner,
    constargs = list("learner" = lrn("regr.rpart")),
    task = task,
    predict_rows_independent = FALSE,
    affect_context_independent = FALSE,
    check_ps_default_values = FALSE)

  mdata = data.table(
    stringsAsFactors = FALSE,
    a = c(1, 2, 3, 4, 5, NA),
    b = c(1, 2, 3, 4, 5, 6),
    c = c(1L, 2L, 3L, 4L, 5L, NA),
    d = factor(c(letters[1:5], NA), levels = letters[1:6]),
    e = factor(letters[1:6], levels = letters[1:6]),
    f = ordered(c(letters[1:5], NA), levels = letters[1:6]),
    g = ordered(letters[1:6], levels = letters[1:6]),
    j = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    k = c(TRUE, FALSE, TRUE, FALSE, TRUE, NA),
    l = factor(letters[rep(1:2, 3)])
  )
  task = TaskClassif$new("mdata", as_data_backend(mdata), target = "l")

  expect_datapreproc_pipeop_class(PipeOpImputeLearner, task = task,
    affect_context_independent = FALSE, predict_rows_independent = FALSE,
    constargs = list("learner" = lrn("regr.rpart")),
    check_ps_default_values = FALSE)

  expect_datapreproc_pipeop_class(PipeOpImputeLearner, task = task,
    affect_context_independent = FALSE, predict_rows_independent = FALSE,
    constargs = list("learner" = lrn("classif.rpart")),
    check_ps_default_values = FALSE)
})

test_that("Test imputation matches, edge cases", {
  mdata = data.table(
    stringsAsFactors = FALSE,
    b = c(2, 4, 2, 4, NA, NA),
    d = factor(c(letters[rep(1L, 4)], "b", NA)),
    l = factor(letters[rep(1:2, 3)])
  )
  task = TaskClassif$new("mdata", as_data_backend(mdata), target = "l")

  po = PipeOpImputeLearner$new(learner = lrn("regr.featureless"))
  out = po$train(list(task))[[1]]$data()
  expect_true(!any(is.na(out$b)))
  expect_true(all(out$b[5:6] == 3))
  out = po$predict(list(task))[[1]]$data()
  expect_true(!any(is.na(out$b)))
  expect_true(all(out$b[5:6] == 3))

  po = PipeOpImputeLearner$new(learner = lrn("classif.featureless"))
  out = po$train(list(task))[[1]]$data()
  expect_true(!any(is.na(out$d)))
  expect_true(all(out$d[6] == "a"))
  out = po$predict(list(task))[[1]]$data()
  expect_true(!any(is.na(out$d)))
  expect_true(all(out$d[6] == "a"))

  # Full NA
  mdata = data.table(
    stringsAsFactors = FALSE,
    b = as.numeric(c(NA, NA, NA)),
    d = factor(c(letters[rep(1L, 4)], "b", NA)),
    l = factor(letters[rep(1:2, 3)])
  )
  task = TaskClassif$new("mdata", mdata, target = "l")
  po = PipeOpImputeLearner$new(learner = lrn("regr.featureless"))
  out = po$train(list(task))[[1]]$data()
  expect_true(!any(is.na(out$b)))
  expect_true(all(out$b == 0))
  out = po$predict(list(task))[[1]]$data()
  expect_true(!any(is.na(out$b)))
  expect_true(all(out$b == 0))

  mdata = data.table(
    stringsAsFactors = FALSE,
    b = as.numeric(c(NA, NA, NA)),
    d = factor(rep(NA, 6), levels = letters[1]),
    l = factor(letters[rep(1:2, 3)])
  )
  task = TaskClassif$new("mdata", mdata, target = "l")
  po = PipeOpImputeLearner$new(learner = lrn("classif.featureless"))
  out = po$train(list(task))[[1]]$data()
  expect_true(!any(is.na(out$d)))
  expect_true(all(out$d == "a"))
  out = po$predict(list(task))[[1]]$data()
  expect_true(!any(is.na(out$d)))
  expect_true(all(out$d == "a"))
})

test_that("PipeOpImputeLearner - model active binding to state", {
  po = PipeOpImputeLearner$new(learner = lrn("regr.featureless"))
  task = mlr_tasks$get("pima")

  # before training states are NULL and learner_models are list()
  expect_null(po$state)
  expect_equal(po$learner$state, po$state)
  expect_equal(po$learner_models, list())

  # after training learner's state is NULL
  train_out = po$train(list(task))
  train_state = po$state
  expect_null(po$learner$state)
  # after training models in the learner_models are equivalent to the models in the state
  models = po$state$model
  expect_equal(names(models), names(po$learner_models))
  expect_true(all(pmap_lgl(list(map(models, .f = "model"), map(po$learner_models, .f = "model")), .f = all.equal)))

  # after predicting state is unchanged and models still are equivalent
  predict_out = po$predict(list(task))
  expect_equal(po$state[-which(names(po$state) == "predict_time")], train_state)
  expect_null(po$learner$state)
  expect_equal(names(models), names(po$learner_models))
  expect_true(all(pmap_lgl(list(map(models, .f = "model"), map(po$learner_models, .f = "model")), .f = all.equal)))
})
