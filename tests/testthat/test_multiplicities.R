context("Multiplicities")

test_that("Multiplicity class and methods", {
  mp = Multiplicity(0)
  expect_multiplicity(mp)
  expect_equal(mp, as.Multiplicity(0))
  expect_error(assert_multiplicity(0, .var.name = "x"), regexp = "Must inherit from class 'Multiplicity'")
  expect_multiplicity(assert_multiplicity(mp))
  nmp = Multiplicity(Multiplicity(0))
  expect_multiplicity(assert_multiplicity(nmp))
  expect_multiplicity(assert_multiplicity(nmp, check_nesting = TRUE))
  expect_error(assert_multiplicity(as.Multiplicity(list(0, Multiplicity(0))), .var.name = "y", check_nesting = TRUE), regexp = "Inconsistent multiplicity nesting level")
  expect_output(print(mp), regexp = "Multiplicity:")
  expect_output(print(Multiplicity()), regexp = "Empty Multiplicity.")
})

test_that("multiplicity_nests_deeper_than", {
  expect_true(multiplicity_nests_deeper_than(Multiplicity(Multiplicity(1)), 1))
  expect_false(multiplicity_nests_deeper_than(Multiplicity(Multiplicity(1)), 2))
  expect_true(is.na(multiplicity_nests_deeper_than(Multiplicity(Multiplicity()), 2)))
  expect_false(multiplicity_nests_deeper_than(Multiplicity(Multiplicity(), Multiplicity(1)), 2))
  expect_true(multiplicity_nests_deeper_than(Multiplicity(Multiplicity(), Multiplicity(Multiplicity())), 2))
})

test_that("PipeOp - assert_connection_table", {
  # toy PipeOp that should fail to be constructed
  PipeOpTestMultiplicitesTable = R6Class("PipeOpTestMultiplicitesTable", inherit = PipeOp,
    public = list(
      initialize = function(id = "multiplicitiestable", param_vals = list()) {
        super$initialize(id, param_vals = param_vals,
          input = data.table(name = "input", train = "[*]", predict = "*"),
          output = data.table(name = "output", train = "*", predict = "*"),
          tags = "multiplicity"
        )
      }
    )
  )
  expect_error(PipeOpTestMultiplicitesTable$new(), regexp = "Multiplicity during train and predict conflicts")
})

# FIXME: check_types in PipeOp

test_that("PipeOp - multiplicity_type_nesting_level", {
  expect_equal(multiplicity_type_nesting_level(c("Task", "[Prediction]", "[[*]]")), c(0L, 1L, 2L))
  expect_error(multiplicity_type_nesting_level("[wrong", varname = "test"), regexp = "square bracket mismatch")
})

test_that("PipeOp - unpack_multiplicities", {
  expect_equal(unpack_multiplicities(list(a = Multiplicity(1, 2), b = 4), c(0, 0), c("a", "b"), "test"),
    list(list(a = 1, b = 4), list(a = 2, b = 4)))
  expect_equal(unpack_multiplicities(list(a = Multiplicity(x = 1, y = 2), b = 4), c(0, 0), c("a", "b"), "test"),
    list(x = list(a = 1, b = 4), y = list(a = 2, b = 4)))
  expect_equal(unpack_multiplicities(list(a = Multiplicity(x = 1, y = 2), b = Multiplicity(x = 10, y = 20)), c(0, 0), c("a", "b"), "test"),
    list(x = list(a = 1, b = 10), y = list(a = 2, b = 20)))
  expect_error(unpack_multiplicities(list(a = Multiplicity(x = 1, z = 2), b = Multiplicity(x = 10, y = 20)), c(0, 0), c("a", "b"), "test"), regexp = "bad multiplicities")
  expect_equal(unpack_multiplicities(list(a = Multiplicity(x = 1, z = 2), b = Multiplicity(x = 10, y = 20)), c(0, 1), c("a", "b"), "test"),
    list(x = list(a = 1, b = Multiplicity(x = 10, y = 20)), z = list(a = 2, b = Multiplicity(x = 10, y = 20))))
  expect_equal(unpack_multiplicities(list(0), 0, "a", "test"), NULL)
})

test_that("PipeOp - evaluate_multiplicities", {
# toy PipeOp only for testing
  PipeOpTestMultiplicites = R6Class("PipeOpTestMultiplicites", inherit = PipeOp,
    public = list(
      initialize = function(num, id = "multiplicities", param_vals = list()) {
        assert_int(num, lower = 1L)
        ps = ParamSet$new(params = list(
          ParamUty$new("state", tags = "train")
        ))
        super$initialize(id, param_set = ps, param_vals = param_vals,
          input = data.table(name = rep_suffix("input", num), train = "*", predict = "*"),
          output = data.table(name = rep_suffix("output", num), train = "*", predict = "*"),
          tags = "multiplicity"
        )
      }
    ),
    private = list(
      # allows to stop with an error on purpose
      .train = function(inputs) {
        if (self$param_set$values$state == "error") stop("Error.")
        self$state = self$param_set$values$state
        inputs
      },
      .predict = function(inputs) {
        if (self$param_set$values$state == "error") stop("Error.")
        inputs
      }
    )
  )

  task = mlr_tasks$get("iris")
  po = PipeOpTestMultiplicites$new(2)
  expect_null(po$state)

  po$param_set$values$state = list("trained")
  train_out1 = po$train(as.Multiplicity(list(0, as.Multiplicity(0))))
  expect_multiplicity(train_out1[[1L]])
  expect_equal(po$state[[1L]][[1L]], "trained")
  expect_setequal(names(po$state[[1L]]), c("", "log", "train_time"))
  predict_out1 = po$predict(as.Multiplicity(list(0, as.Multiplicity(0))))
  expect_equal(po$state[[1L]][[1L]], "trained")
  expect_setequal(names(po$state[[1L]]), c("", "log", "train_time", "predict_time"))
  expect_multiplicity(predict_out1[[1L]])

  po$state = list("no_multiplicties")
  expect_error(po$predict(as.Multiplicity(list(0, as.Multiplicity(0)))), regexp = "state was not a multiplicity")
  expect_equal(po$state, list("no_multiplicties"))

  po$state = as.Multiplicity(NULL)
  expect_error(po$predict(as.Multiplicity(list(0, as.Multiplicity(0)))), regexp = "state had different length / names than input")
  expect_equal(po$state, as.Multiplicity(NULL))

  po$param_set$values$state = list("trained")
  train_out2 = po$train(as.Multiplicity(list(0, as.Multiplicity(0))))
  expect_multiplicity(train_out2[[1]])
  old_state = po$state
  po$param_set$values$state = list("error")
  expect_error(po$train(as.Multiplicity(list(0, as.Multiplicity(0)))), regexp = "Error")
  expect_equal(po$state, NULL)  # state is completely reset to NULL
})

test_that("Graph - add_edge", {
  learner = lrn("classif.rpart")
  g1 = PipeOpOVRSplit$new() %>>% learner %>>% PipeOpOVRUnite$new()
  g2 = Graph$new()
  g2$add_pipeop(PipeOpOVRSplit$new())
  g2$add_pipeop(learner)
  g2$add_pipeop(PipeOpOVRUnite$new())
  g2$add_edge("ovrsplit", "classif.rpart")
  g2$add_edge("classif.rpart", "ovrunite")
  expect_identical(g1$edges, g2$edges)
})

