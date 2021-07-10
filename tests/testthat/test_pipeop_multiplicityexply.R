context("PipeOpMultiplicityExply")

test_that("multiplicityexply - basic properties", {
  po = PipeOpMultiplicityExply$new(3)
  expect_pipeop(po)
  expect_data_table(po$input, nrows = 1L)
  expect_data_table(po$output, nrows = 3L)

  expect_pipeop_class(PipeOpMultiplicityExply, list(1))
  expect_pipeop_class(PipeOpMultiplicityExply, list(3))
})

test_that("multiplicityexply - train and predict", {
  tsk = mlr_tasks$get("iris")
  keep1 = setdiff(tsk$feature_names, c("Sepal.Length", "Sepal.Width"))
  keep2 = setdiff(tsk$feature_names, c("Petal.Length", "Petal.Width"))
  t1 = tsk$clone()$select(keep1)
  t2 = tsk$clone()$select(keep2)

  po = PipeOpMultiplicityExply$new(2)
  tout = train_pipeop(po, list(as.Multiplicity(list(t1, t2))))
  expect_list(po$state, len = 2L)
  expect_setequal(names(po$state), c("log", "train_time"))
  expect_list(tout, len = 2L)
  expect_equal(tout[[1L]], t1)
  expect_equal(tout[[2L]], t2)
  pout = predict_pipeop(po, list(as.Multiplicity(list(t1, t2))))
  expect_list(po$state, len = 3L)
  expect_setequal(names(po$state), c("log", "train_time", "predict_time"))
  expect_list(pout, len = 2L)
  expect_equal(pout[[1L]], t1)
  expect_equal(pout[[2L]], t2)
})
