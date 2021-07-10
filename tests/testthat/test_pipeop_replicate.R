context("PipeOpReplicate")

test_that("PipeOpReplicate - basic properties", {
  po = PipeOpReplicate$new()
  expect_pipeop(po)
  expect_data_table(po$input, nrows = 1L)
  expect_data_table(po$output, nrows = 1L)

  expect_pipeop_class(PipeOpReplicate)
})

test_that("PipeOpReplicate - train and predict", {
  tsk = mlr_tasks$get("iris")
  nreps = 3L
  po = PipeOpReplicate$new(param_vals = list(reps = nreps))

  tout = train_pipeop(po, list(tsk))
  expect_list(po$state, len = 2L)
  expect_multiplicity(tout[[1L]])
  expect_list(tout[[1L]], len = nreps)
  expect_true(all(map_lgl(tout[[1L]], .f = function(x) all.equal(x, tsk))))
  pout = predict_pipeop(po, list(tsk))
  expect_multiplicity(pout[[1L]])
  expect_list(pout[[1L]], len = 3L)
  expect_true(all(map_lgl(pout[[1L]], .f = function(x) all.equal(x, tsk))))
})
