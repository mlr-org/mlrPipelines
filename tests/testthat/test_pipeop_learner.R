context("PipeOpLearner")

test_that("PipeOpLearner - basic properties", {
  lrn = mlr_learners$get("classif.featureless")
  po = PipeOpLearner$new(lrn)
  expect_pipeop(po, check_ps_default_values = FALSE)
  expect_data_table(po$input, nrows = 1)
  expect_data_table(po$output, nrows = 1)

  task = mlr_tasks$get("iris")
  result = train_pipeop(po, list(task = task))
  expect_null(result[[1L]])

  result = predict_pipeop(po, list(task = task))
  expect_class(result[[1L]], "Prediction")

  expect_pipeop_class(PipeOpLearner, list(lrn), check_ps_default_values = FALSE)
  expect_error(PipeOpLearner$new())
})

test_that("PipeOpLearner - param_set and values", {
  lrn = mlr_learners$get("classif.rpart")
  po = PipeOpLearner$new(lrn)

  # Setting and getting pipeops works
  expect_pipeop(po, check_ps_default_values = FALSE)
  expect_equal(po$param_set, po$learner$param_set)


  expect_equal(po$param_set$values, po$learner$param_set$values)
  expect_error({
    po$param_set$values$minsplit = "foo"
  })
  po$param_set$values$minsplit = 2L
  expect_equal(po$param_set$values, po$learner$param_set$values)
  expect_equal(po$param_set$values, list(xval = 0L, minsplit = 2L))
  po$param_set$values$maxdepth = 1L
  expect_equal(po$param_set$values, list(xval = 0L, minsplit = 2L, maxdepth = 1L))
  po$param_set$values = list(minsplit = 1L)
  expect_equal(po$param_set$values, list(minsplit = 1L))
  expect_error({
    po$param_set$values = list(minsplit = "foo")
  })
  expect_error({
    po$param_set$values = list(foo = "foo")
  })
})

test_that("PipeOpLearner - graph but no id", {
  g = PipeOpNOP$new() %>>% PipeOpLearner$new(LearnerClassifRpart$new())
  po = PipeOpLearner$new(g)
  expect_string(po$id)
})

test_that("PipeOpLearner - model active binding to state", {
  lrn = mlr_learners$get("classif.featureless")
  po = PipeOpLearner$new(lrn)
  task = mlr_tasks$get("iris")

  # before training states are NULL
  expect_null(po$state)
  expect_equal(po$learner$state, po$state)
  expect_equal(po$learner_model$state, po$state)

  # after training learner_model's state and state are equal
  train_out = po$train(list(task))
  train_state = po$state
  expect_null(po$learner$state)
  expect_equal(po$learner_model$state, train_state)

  # after predicting states are unchanged
  predict_out = po$predict(list(task))
  expect_equal(po$state[-which(names(po$state) == "predict_time")], train_state)
  expect_null(po$learner$state)
  expect_equal(po$learner_model$state, po$state)
})
