disable_encapsulation = function(x) {
  x$encapsulate = c(train = "none", predict = "none")
  x
}

enable_encapsulation = function(x) {
  x$encapsulate = c(train = "evaluate", predict = "evaluate")
  x
}

test_that("Encapsulate - PipeOp", {
  pipeop = PipeOpDebugEncapsulate$new(param_vals = list(
    message_train = TRUE, warning_train = TRUE, message_predict = TRUE, warning_predict = TRUE)
  )

  expect_message(expect_warning(disable_encapsulation(pipeop)$train(list(1))))
  log = pipeop$log
  expect_data_table(log)

  expect_silent(enable_encapsulation(pipeop)$train(list(1)))
  log = pipeop$log
  expect_data_table(log)
  expect_data_table(log, nrows = 2L, ncols = 3L, any.missing = FALSE)
  expect_factor(log$class)
  expect_set_equal(as.character(log$class), c("output", "warning"))
  expect_true(all(grepl("->train()", log$msg, fixed = TRUE)))
  expect_true("output" %in% log$class)
  expect_true("warning" %in% log$class)
  expect_false("error" %in% log$class)

  expect_message(expect_warning(disable_encapsulation(pipeop)$predict(list(1))))
  log = pipeop$log[stage == "predict"]
  expect_data_table(log)
  expect_equal(nrow(log), 0L)

  enable_encapsulation(pipeop)$predict(list(1))
  log = pipeop$log[stage == "predict"]
  expect_data_table(log)
  expect_data_table(log, nrows = 2L, ncols = 3L, any.missing = FALSE)
  expect_factor(log$class)
  expect_equal(as.character(log$class), c("output", "warning"))
  expect_true(all(grepl("->predict()", log$msg, fixed = TRUE)))
})

test_that("Encapsulate - Graph", {
  task = tsk("iris")

  pipeop = PipeOpDebugEncapsulate$new(param_vals = list(
    message_train = TRUE, warning_train = TRUE, message_predict = TRUE, warning_predict = TRUE)
  )

  learner = lrn("classif.debug")
  learner$param_set$values = list(message_train = 1, warning_train = 1, message_predict = 1, warning_predict = 1)

  # encapsulation of pipeops within graph disabled
  g1 = disable_encapsulation(pipeop) %>>% disable_encapsulation(learner)
  expect_true(all(unlist(map(g1$pipeops, "encapsulate")) == "none"))

  expect_message(expect_warning(g1$train(task)))  # train
  log = map(g1$pipeops, "log")
  for (i in seq_along(log)) {
    l = log[[i]]
    expect_data_table(l)
  }

  for (i in seq_along(g1$pipeops)) {
    g1$pipeops[[i]]$encapsulate = c(train = "evaluate", predict = "evaluate")
  }
  expect_true(all(unlist(map(g1$pipeops, "encapsulate")) == "evaluate"))
  expect_silent(g1$train(task))  # train, encapsulation of pipeops within graph enabled
  log = map(g1$pipeops, "log")
  for (i in seq_along(log)) {
    l = log[[i]]
    expect_data_table(l)
    expect_data_table(l, nrows = 2L, ncols = 3L, any.missing = FALSE)
  }

  for (i in seq_along(g1$pipeops)) {
    g1$pipeops[[i]]$encapsulate = c(train = "none", predict = "none")
  }
  expect_true(all(unlist(map(g1$pipeops, "encapsulate")) == "none"))
  expect_message(expect_warning(g1$predict(task)))  # predict, encapsulation of pipeops within graph disabled
  log = map(g1$pipeops, function(pipeop) pipeop$log[stage == "predict"])
  for (i in seq_along(log)) {
    l = log[[i]]
    expect_data_table(l)
  }

  for (i in seq_along(g1$pipeops)) {
    g1$pipeops[[i]]$encapsulate = c(train = "evaluate", predict = "evaluate")
  }
  expect_true(all(unlist(map(g1$pipeops, "encapsulate")) == "evaluate"))
  expect_silent(g1$predict(task))  # predict, encapsulation of graph
  log = map(g1$pipeops, function(pipeop) pipeop$log[stage == "predict"])
  for (i in seq_along(log)) {
    l = log[[i]]
    expect_data_table(l)
    expect_data_table(l, nrows = 2L, ncols = 3L, any.missing = FALSE)
  }
})

test_that("Encapsulate - GraphLearner", {
  task = tsk("iris")

  pipeop = PipeOpDebugEncapsulate$new(param_vals = list(
    message_train = TRUE, warning_train = TRUE, message_predict = TRUE, warning_predict = TRUE)
  )

  # encapsulation of pipeops within graphlearner will always be set to "none"
  gl1 = GraphLearner$new(enable_encapsulation(pipeop) %>>% enable_encapsulation(lrn("classif.debug")))
  expect_true(all(unlist(map(gl1$graph$pipeops, "encapsulate")) == "none"))

  expect_message(expect_warning(disable_encapsulation(gl1)$train(task)))  # train, no encapsulation of graph
  log = gl1$log
  expect_data_table(log)
  expect_true(every(map(gl1$graph$pipeops, "log"), is.null))  # will always be NULL

  expect_silent(enable_encapsulation(gl1)$train(task))  # train, encapsulation of graph
  log = gl1$log
  expect_data_table(log)
  expect_data_table(log, nrows = 2L, ncols = 3L, any.missing = FALSE)
  expect_true(every(map(gl1$graph$pipeops, "log"), is.null))

  expect_message(expect_warning(disable_encapsulation(gl1)$predict(task)))  # predict, no encapsulation of graph
  log = gl1$log[stage == "predict"]
  expect_data_table(log)
  expect_equal(nrow(log), 0L)
  expect_true(every(map(gl1$graph$pipeops, "log"), is.null))

  expect_silent(enable_encapsulation(gl1)$predict(task))  # predict, encapsulation of graph
  log = gl1$log[stage == "predict"]
  expect_data_table(log)
  expect_true(every(map(gl1$graph$pipeops, "log"), is.null))
})

test_that("Encapsulate - PipeOpLearner and PipeOpLearnerCV", {
  learner = lrn("classif.debug")
  learner$encapsulate = c(train = "evaluate", predict = "none")

  pl1 = PipeOpLearner$new(learner)
  pl2 = as_pipeop(learner)

  plcv = PipeOpLearnerCV$new(learner)

  expect_equal(learner$encapsulate, pl1$encapsulate)
  expect_equal(pl1$encapsulate, pl2$encapsulate)
  expect_equal(pl2$encapsulate, plcv$encapsulate)
})
