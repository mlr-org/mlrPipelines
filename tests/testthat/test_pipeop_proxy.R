context("PipeOpProxy")

test_that("PipeOpProxy - basic properties", {
  task = mlr_tasks$get("iris")
  pop = PipeOpProxy$new(param_vals = list(content = PipeOpNOP$new()))
  expect_pipeop(pop)
  expect_equal(train_pipeop(pop, inputs = list(task))[[1L]], task)
  expect_length(pop$state, 3L)
  expect_length(pop$state$nop, 2L)
  expect_equal(predict_pipeop(pop, inputs = list(task))[[1L]], task)
})

test_that("PipeOpProxy - datapreproc", {
  pop = PipeOpScale$new()
  expect_datapreproc_pipeop_class(PipeOpProxy, constargs = list(param_vals = list(content = pop)), task = mlr_tasks$get("iris"))
})

test_that("PipeOpProxy - content error handling", {
  expect_error(PipeOpProxy$new(param_vals = list(content = "error")), regexp = "`content` must be an object that can be converted to a Graph")
  expect_error(PipeOpProxy$new(param_vals = list(content = PipeOpCopy$new(outnum = 2L))), regexp = "Graph's output number must either be 1 or match `outnum`")
  expect_error(PipeOpProxy$new(param_vals = list(content = PipeOpFeatureUnion$new(innum = 3L)), innum = 2), regexp = "Graph's input number .* match `innum`")
})

test_that("PipeOpProxy - several inputs via featureunion", {
  task1 = mlr_tasks$get("iris")
  task2 = mlr_tasks$get("iris")
  gr1 = PipeOpCopy$new(outnum = 2L) %>>% gunion(list(PipeOpNOP$new(), PipeOpPCA$new())) %>>% PipeOpProxy$new(param_vals = list(content = PipeOpFeatureUnion$new()))
  gr2 = PipeOpCopy$new(outnum = 2L) %>>% gunion(list(PipeOpNOP$new(), PipeOpPCA$new())) %>>% PipeOpFeatureUnion$new()
  tout1 = gr1$train(task1)
  tout2 = gr2$train(task2)
  expect_equal(tout1[[1L]], tout2[[1L]])
})

test_that("PipeOpProxy - several outputs", {
  task1 = mlr_tasks$get("iris")
  task2 = mlr_tasks$get("iris")
  pop = PipeOpProxy$new(outnum = 2L, param_vals = list(content = PipeOpCopy$new(outnum = 2L)))
  pop_copy = PipeOpCopy$new(outnum = 2L)
  tout1 = pop$train(list(task1))
  tout2 = pop_copy$train(list(task2))
  expect_equal(tout1, tout2)
})

test_that("PipeOpProxy - PCA proxied", {
  task1 = mlr_tasks$get("iris")
  task2 = mlr_tasks$get("iris")
  pop = PipeOpProxy$new(param_vals = list(content =
    PipeOpPCA$new(param_vals = list(center = TRUE, scale. = TRUE, rank. = 1L))))
  pop_pca = PipeOpPCA$new(param_vals = list(center = TRUE, scale. = TRUE, rank. = 1L))
  task1 = mlr_tasks$get("iris")
  task2 = task1$clone(deep = TRUE)
  tout = train_pipeop(pop, list(task1))
  tout_pca = train_pipeop(pop_pca, list(task2))
  expect_equal(tout[[1L]], tout_pca[[1L]])
})

test_that("PipeOpProxy - Graph proxied", {
  pop = PipeOpProxy$new(param_vals = list(content =
    PipeOpScale$new() %>>%
    PipeOpPCA$new(param_vals = list(center = TRUE, scale. = TRUE, rank. = 1L))))
  gr = PipeOpScale$new() %>>%
    PipeOpPCA$new(param_vals = list(center = TRUE, scale. = TRUE, rank. = 1L))
  task1 = mlr_tasks$get("iris")
  task2 = task1$clone(deep = TRUE)
  tout = train_pipeop(pop, list(task1))
  tout_gr = gr$train(task2)
  expect_equal(tout[[1L]], tout_gr[[1L]])
})

test_that("PipeOpProxy - Complicated Graphs", {

  # ---------
  # two parallel single-input-single-output pipeops

  pop = po("proxy", content = list(PipeOpDebugMulti$new(1, 1, "debug1"), PipeOpDebugMulti$new(1, 1, "debug2")), outnum = 2L)

  expect_output(
    expect_equal(pop$train(list(1)), list(output1 = 2, output2 = 2)),
    "^Training debug1 with input list\\(input_1 = 1\\)
Training debug2 with input list\\(input_1 = 1\\)$")

  expect_output(
    expect_equal(pop$predict(list(1)), list(output1 = 2, output2 = 2)),
    "^Predicting debug1 with input list\\(input_1 = 1\\) and state list\\(input_1 = 1\\)
Predicting debug2 with input list\\(input_1 = 1\\) and state list\\(input_1 = 1\\)$")

  expect_output(
    expect_equal(pop$train(list(1, 2)), list(output1 = 2, output2 = 3)),
    "^Training debug1 with input list\\(input_1 = 1\\)
Training debug2 with input list\\(input_1 = 2\\)$")

  expect_output(
    expect_equal(pop$predict(list(3, 4)), list(output1 = 4, output2 = 5)),
    "^Predicting debug1 with input list\\(input_1 = 3\\) and state list\\(input_1 = 1\\)
Predicting debug2 with input list\\(input_1 = 4\\) and state list\\(input_1 = 2\\)$")


  # ---------
  # NOP | feature union | NOP; feature union has vararg

  pop = po("proxy", content = list(PipeOpDebugMulti$new(1, 1, "debug1"), PipeOpFeatureUnion$new(), PipeOpDebugMulti$new(1, 1, "debug2")), outnum = 3L)

  tsk1 = tsk("iris")
  tsk2 = po("pca")$train(list(tsk1))[[1L]]
  tsk3 = po("ica")$train(list(tsk1))[[1L]]

  expect_output(
      expect_equal(pop$train(list(1, tsk1, tsk2, tsk3, 2)),
        list(output1 = 2, output2 = po("featureunion")$train(list(tsk1, tsk2, tsk3))[[1L]], output3 = 3)),
    "^Training debug1 with input list\\(input_1 = 1\\)
Training debug2 with input list\\(input_1 = 2\\)$")

  expect_output(
      expect_equal(pop$train(list(1, tsk2, 2)),
        list(output1 = 2, output2 = tsk2, output3 = 3)),
    "^Training debug1 with input list\\(input_1 = 1\\)
Training debug2 with input list\\(input_1 = 2\\)$")

})
