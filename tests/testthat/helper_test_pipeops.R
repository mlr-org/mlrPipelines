PipeOpDebugBasic = R6Class("PipeOpDebugBasic",
  inherit = PipeOp,
  public = list(
    initialize = function(id = "debug.basic", param_set = ParamSet$new()) {
      super$initialize(id = id, param_set = param_set,
        input = data.table(name = "input", train = "*", predict = "*"),
        output = data.table(name = "output", train = "*", predict = "*")
      )
    }),
  private = list(
    .train = function(inputs) {
      catf("Training %s", self$id)
      self$state = inputs
    },
    .predict = function(inputs) {
      catf("Predicting %s", self$id)
      self$state = c(self$state, inputs)
      inputs
    }
  )
)

deparse_list_safe = function(l) {
  # deparse list, but work the same on all systems
  if (!is.null(names(l))) {
    sprintf("list(%s)", paste(names(l), sapply(l, deparse), sep = " = ", collapse = ", "))
  } else {
    sprintf("list(%s)", paste(as.character(l), sep = ", "))
  }
}

# initialize with inputs / outputs: input / output channel names, or number of channels
PipeOpDebugMulti = R6Class("PipeOpDebugMulti",
  inherit = PipeOp,
  public = list(
    nin = NULL,
    nout = NULL,
    initialize = function(inputs, outputs, id = "debug.multi") {
      if (is.numeric(inputs)) {
        inputs = paste0("input_", seq_len(inputs))
      }
      if (is.numeric(outputs)) {
        outputs = paste0("output_", seq_len(outputs))
      }
      p = ParamInt$new(id = "par", lower = 0, upper = 10, default = 0, tags = c("train", "predict"))
      self$nin = length(inputs)
      self$nout = length(outputs)
      super$initialize(id, ParamSet$new(list(p)),
        input = data.table(name = inputs, train = "*", predict = "*"),
        output = data.table(name = outputs, train = "*", predict = "*"))
    }),
  private = list(
    .train = function(inputs) {
      catf("Training %s with input %s", self$id, deparse_list_safe(inputs))
      self$state = inputs
      iin = inputs[[1]]
      as.list(iin + seq_len(self$nout))
    },
    .predict = function(inputs) {
      catf("Predicting %s with input %s and state %s",
        self$id, deparse_list_safe(inputs), deparse_list_safe(self$state[-which(names(self$state) %in% c("log", "train_time"))]))
      iin = inputs[[1]]
      as.list(iin + seq_len(self$nout))
    }
  )
)

VarargPipeop = R6Class("VarargPipeop",
  inherit = PipeOp,
  public = list(
    initialize = function(id = "vararg", innum = 0, param_vals = list()) {
      super$initialize(id, param_vals = param_vals,
        input = data.table(name = c("...", rep_suffix("input", innum)), train = "*", predict = "*"),
        output = data.table(name = "output", train = "*", predict = "*")
      )
    }),
    private = list(
    .train = function(inputs) {
      self$state = inputs
      list(inputs)
    },
    .predict = function(inputs) {
      self$state = inputs
      list(inputs)
    }
  )
)

PipeOpDebugEncapsulate = R6Class("PipeOpDebugEncapsulate",
  inherit = PipeOp,
  public = list(
    initialize = function(id = "debug.encapsulate", param_vals = list()) {
      param_set = ParamSet$new(list(
        ParamLgl$new("message_train", default = FALSE, tags = "train"),
        ParamLgl$new("message_predict", default = FALSE, tags = "predict"),
        ParamLgl$new("warning_train", default = FALSE, tags = "train"),
        ParamLgl$new("warning_predict", default = FALSE, tags = "predict"),
        ParamLgl$new("error_train", default = FALSE, tags = "train"),
        ParamLgl$new("error_predict", default = FALSE, tags = "predict"),
        ParamLgl$new("segfault_train", default = FALSE, tags = "train"),
        ParamLgl$new("segfault_predict", default = FALSE, tags = "predict"),
        ParamLgl$new("predict_missing", default = FALSE, tags = "predict")
      ))
      super$initialize(id = id, param_set = param_set, param_vals = param_vals,
        input = data.table(name = "input", train = "*", predict = "*"),
        output = data.table(name = "output", train = "*", predict = "*")
      )
    }),
  private = list(
    .train = function(inputs) {
      pv = self$param_set$get_values(tags = "train")

      if (isTRUE(pv[["message_train"]])) {
        message("Message from classif.debug->train()")
      }
      if (isTRUE(pv[["warning_train"]])) {
        warning("Warning from classif.debug->train()")
      }
      if (isTRUE(pv[["error_train"]])) {
        stop("Error from classif.debug->train()")
      }
      if (isTRUE(pv[["segfault_train"]])) {
        get("attach")(structure(list(), class = "UserDefinedDatabase"))
      }
      self$state = inputs
    },
    .predict = function(inputs) {
      pv = self$param_set$get_values(tags = "predict")

      if (isTRUE(pv[["message_predict"]])) {
        message("Message from classif.debug->predict()")
      }
      if (isTRUE(pv[["warning_predict"]])) {
        warning("Warning from classif.debug->predict()")
      }
      if (isTRUE(pv[["error_predict"]])) {
        stop("Error from classif.debug->predict()")
      }
      if (isTRUE(pv[["segfault_predict"]])) {
        get("attach")(structure(list(), class = "UserDefinedDatabase"))
      }
      self$state = c(self$state, inputs)
      inputs
    }
  )
)

