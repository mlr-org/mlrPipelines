#' @include utils.R
#' @include mlr_pipeops.R
#'
#' @title PipeOp Base Class
#'
#' @usage NULL
#' @format Abstract [`R6Class`].
#'
#' @description
#' A [`PipeOp`] represents a transformation of a given "input" into a given "output", with two stages: "training"
#' and "prediction". It can be understood as a generalized function that not only has multiple inputs, but
#' also multiple outputs (as well as two stages). The "training" stage is used when training a machine learning pipeline or
#' fitting a statistical model, and the "predicting" stage is then used for making predictions
#' on new data.
#'
#' To perform training, the `$train()` function is called which takes inputs and transforms them, while simultaneously storing information
#' in its `$state` slot. For prediction, the `$predict()` function is called, where the `$state` information can be used to influence the transformation
#' of the new data.
#'
#' A [`PipeOp`] is usually used in a [`Graph`] object, a representation of a computational graph. It can have
#' multiple **input channels**---think of these as multiple arguments to a function, for example when averaging
#' different models---, and multiple **output channels**---a transformation may
#' return different objects, for example different subsets of a [`Task`][mlr3::Task]. The purpose of the [`Graph`] is to
#' connect different outputs of some [`PipeOp`]s to inputs of other [`PipeOp`]s.
#'
#' Input and output channel information of a [`PipeOp`] is defined in the `$input` and `$output` slots; each channel has a *name*, a required
#' type during training, and a required type during prediction. The `$train()` and `$predict()` function are called with a `list` argument
#' that has one entry for each declared channel (with one exception, see next paragraph). The `list` is automatically type-checked
#' for each channel against `$input` and then passed on to the `private$.train()` or `private$.predict()` functions. There the data is processed and
#' a result `list` is created. This `list` is again type-checked for declared output types of each channel. The length and types of the result
#' `list` is as declared in `$output`.
#'
#' A special input channel name is `"..."`, which creates a *vararg* channel that takes arbitrarily many arguments, all of the same type. If the `$input`
#' table contains an `"..."`-entry, then the input given to `$train()` and `$predict()` may be longer than the number of declared input channels.
#'
#' This class is an abstract base class that all [`PipeOp`]s being used in a [`Graph`] should inherit from,  and
#' is not intended to be instantiated.
#'
#' @section Construction:
#' ```
#' PipeOp$new(id, param_set = ParamSet$new(), param_vals = list(), input, output, packages = character(0), tags = character(0))
#' ```
#'
#' * `id` :: `character(1)`\cr
#'   Identifier of resulting object. See `$id` slot.
#' * `param_set` :: [`ParamSet`][paradox::ParamSet] | `list` of `expression`\cr
#'   Parameter space description. This should be created by the subclass and given to `super$initialize()`.
#'   If this is a [`ParamSet`][paradox::ParamSet], it is used as the [`PipeOp`]'s [`ParamSet`][paradox::ParamSet]
#'   directly. Otherwise it must be a `list` of expressions e.g. created by `alist()` that evaluate to [`ParamSet`][paradox::ParamSet]s.
#'   These [`ParamSet`][paradox::ParamSet] are combined using a [`ParamSetCollection`][paradox::ParamSetCollection].
#' * `param_vals` :: named `list`\cr
#'   List of hyperparameter settings, overwriting the hyperparameter settings given in `param_set`. The
#'   subclass should have its own `param_vals` parameter and pass it on to `super$initialize()`. Default `list()`.
#' * input :: [`data.table`] with columns `name` (`character`), `train` (`character`), `predict` (`character`)\cr
#'   Sets the `$input` slot of the resulting object; see description there.
#' * output :: [`data.table`] with columns `name` (`character`), `train` (`character`), `predict` (`character`)\cr
#'   Sets the `$output` slot of the resulting object; see description there.
#' * packages :: `character`\cr
#'   Set of all required packages for the [`PipeOp`]'s `$train` and `$predict` methods. See `$packages` slot.
#'   Default is `character(0)`.
#' * `tags` ::`character`\cr
#'   A set of tags associated with the [`PipeOp`]. Tags describe a [`PipeOp`]'s purpose.
#'   Can be used to filter `as.data.table(mlr_pipeops)`. Default is `"abstract"`, indicating an abstract [`PipeOp`].
#'
#' @section Internals:
#' [`PipeOp`] is an abstract class with abstract functions `private$.train()` and `private$.predict()`. To create a functional
#' [`PipeOp`] class, these two methods must be implemented. Each of these functions receives a named `list` according to
#' the [`PipeOp`]'s input channels, and must return a `list` (names are ignored) with values in the order of output
#' channels in `$output`. The `private$.train()` and `private$.predict()` function should not be called by the user;
#' instead, a `$train()` and `$predict()` should be used. The most convenient usage is to add the [`PipeOp`]
#' to a `Graph` (possibly as singleton in that `Graph`), and using the `Graph`'s `$train()` / `$predict()` methods.
#'
#' @section Fields:
#' * `id` :: `character`\cr
#'   ID of the [`PipeOp`]. IDs are user-configurable, and IDs of [`PipeOp`]s must be unique within a [`Graph`]. IDs of
#'   [`PipeOp`]s must not be changed once they are part of a [`Graph`], instead the [`Graph`]'s `$set_names()` method
#'   should be used.
#' * `packages` :: `character`\cr
#'   Packages required for the [`PipeOp`]. Functions that are not in base R should still be called using `::`
#'   (or explicitly attached using `require()`) in `private$.train()` *and* `private$.predict()`, but
#'   packages declared here are checked before any (possibly expensive) processing has started within a [`Graph`].
#' * `param_set` :: [`ParamSet`][paradox::ParamSet]\cr
#'   Parameters and parameter constraints. Parameter values that influence the functioning of `$train` and / or `$predict` are
#'   in the `$param_set$values` slot; these are automatically checked against parameter constraints in `$param_set`.
#' * `state` :: `any` | `NULL`\cr
#'   Method-dependent state obtained during training step, and usually required for the prediction step. This is `NULL`
#'   if and only if the [`PipeOp`] has not been trained. The `$state` is the *only* slot that can be reliably modified during
#'   `$train()`, because `private$.train()` may theoretically be executed in a different `R`-session (e.g. for parallelization).
#' * input :: [`data.table`] with columns `name` (`character`), `train` (`character`), `predict` (`character`)\cr
#'   Input channels of [`PipeOp`]. Column `name` gives the names (and order) of values in the list given to
#'   `$train()` and `$predict()`. Column `train` is the (S3) class that an input object must conform to during
#'   training, column `predict` is the (S3) class that an input object must conform to during prediction. Types
#'   are checked by the [`PipeOp`] itself and do not need to be checked by `private$.train()` / `private$.predict()` code.\cr
#'   A special name is `"..."`, which creates a *vararg* input channel that accepts a variable number of inputs.\cr
#'   If a row has both `train` and `predict` values enclosed by square brackets ("`[`", "`]`), then this channel is
#'   [`Multiplicity`]-aware. If the [`PipeOp`] receives a [`Multiplicity`] value on these channels, this [`Multiplicity`]
#'   is given to the `.train()` and `.predict()` functions directly. Otherwise, the [`Multiplicity`] is transparently
#'   unpacked and the `.train()` and `.predict()` functions are called multiple times, once for each [`Multiplicity`] element.
#'   The type enclosed by square brackets indicates that only a [`Multiplicity`] containing values of this type are accepted.
#'   See [`Multiplicity`] for more information.
#' * output :: [`data.table`] with columns `name` (`character`), `train` (`character`), `predict` (`character`)\cr
#'   Output channels of [`PipeOp`], in the order in which they will be given in the list returned by `$train` and
#'   `$predict` functions. Column `train` is the (S3) class that an output object must conform to during training,
#'   column `predict` is the (S3) class that an output object must conform to during prediction. The [`PipeOp`] checks
#'   values returned by `private$.train()` and `private$.predict()` against these types specifications.\cr
#'   If a row has both `train` and `predict` values enclosed by square brackets ("`[`", "`]`), then this signals that the channel
#'   emits a [`Multiplicity`] of the indicated type. See [`Multiplicity`] for more information.
#' * `innum` :: `numeric(1)` \cr
#'   Number of input channels. This equals `nrow($input)`.
#' * `outnum` :: `numeric(1)` \cr
#'   Number of output channels. This equals `nrow($output)`.
#' * `is_trained` :: `logical(1)` \cr
#'   Indicate whether the [`PipeOp`] was already trained and can therefore be used for prediction.
#' * `tags` :: `character`\cr
#'   A set of tags associated with the [`PipeOp`]. Tags describe a [`PipeOp`]'s purpose.
#'   Can be used to filter `as.data.table(mlr_pipeops)`.
#'   [`PipeOp`] tags are inherited and child classes can introduce additional tags.
#' * `hash` :: `character(1)` \cr
#'   Checksum calculated on the [`PipeOp`], depending on the [`PipeOp`]'s `class` and the slots `$id` and `$param_set$values`. If a
#'   [`PipeOp`]'s functionality may change depending on more than these values, it should inherit the `$hash` active
#'   binding and calculate the hash as `digest(list(super$hash, <OTHER THINGS>), algo = "xxhash64")`.
#' * `timings` :: `numeric(2)` \cr
#'   Elapsed time in seconds for the steps `"train"` and `"predict"`.
#'   Measured via [mlr3misc::encapsulate()].
#' * `log` :: [`data.table`]\cr
#'   Returns the output (including warning and errors) as table with columns `"stage"` ("train" or "predict"),
#'   `"class"` ("output", "warning", or "error"), and `"msg"` (`character()`).
#' * `warnings` :: `character()`\cr
#'   Logged warnings as vector.
#' * `errors` :: `character()`\cr
#'   Logged errors as vector.
#' * `encapsulate` :: named `character()`\cr
#'   Controls how to execute the code in internal train and predict methods.
#'   Must be a named character vector with names `"train"` and `"predict"`.
#'   Possible values are `"none"`, `"evaluate"` (requires package \CRANpkg{evaluate}) and `"callr"` (requires package \CRANpkg{callr}).
#'   See [mlr3misc::encapsulate()] for more details.
#' * `.result` :: `list` \cr
#'   If the [`Graph`]'s `$keep_results` flag is set to `TRUE`, then the intermediate Results of `$train()` and `$predict()`
#'   are saved to this slot, exactly as they are returned by these functions. This is mainly for debugging purposes
#'   and done, if requested, by the [`Graph`] backend itself; it should *not* be done explicitly by `private$.train()` or `private$.predict()`.
#'
#' @section Methods:
#' * `train(input)`\cr
#'   (`list`) -> named `list`\cr
#'   Train [`PipeOp`] on `inputs`, transform it to output and store the learned `$state`. If the [`PipeOp`] is already
#'   trained, already present `$state` is overwritten. Input list is typechecked against the `$input` `train` column.
#'   Return value is a list with as many entries as `$output` has
#'   rows, with each entry named after the `$output` `name` column and class according to the `$output` `train` column.
#'   The workhorse function for training each [`PipeOp`] is the private
#'   `.train(input)`\cr: (named `list`) -> `list`\cr function.
#'   It's an Abstract function that must be implemented by concrete subclasses. `private$.train()` is called by `$train()` after
#'   typechecking. It must change the `$state` value to something non-`NULL` and return a list of transformed data according to
#'   the `$output` `train` column. Names of the returned list are ignored.\cr
#'   The `private$.train()` method should not be called by a user; instead, the `$train()` method should be used which does some
#'   checking and possibly type conversion.
#' * `predict(input)` \cr
#'   (`list`) -> named `list`\cr
#'   Predict on new data in `input`, possibly using the stored `$state`. Input and output are specified by `$input` and `$output`
#'   in the same way as for `$train()`, except that
#'   the `predict` column is used for type checking.
#'   The workhorse function for predicting in each using each [`PipeOp`] is
#'   `.predict(input)`\cr (named `list`) -> `list`\cr
#'   Abstract function that must be implemented by concrete subclasses. `private$.predict()` is called by `$predict()` after
#'   typechecking and works analogously to `private$.train()`. Unlike `private$.train()`, `private$.predict()` should not modify
#'   the [`PipeOp`] in any way.\cr
#'   Just as `private$.train()`, `private$.predict()` should not be called by a user; instead, the `$predict()` method should be used.
#' * `print()` \cr
#'   () -> `NULL` \cr
#'   Prints the [`PipeOp`]s most salient information: `$id`, `$is_trained`, `$param_set$values`, `$input` and `$output`.
#' @examples
#' # example (bogus) PipeOp that returns the sum of two numbers during $train()
#' # as well as a letter of the alphabet corresponding to that sum during $predict().
#'
#' PipeOpSumLetter = R6::R6Class("sumletter",
#'   inherit = PipeOp,  # inherit from PipeOp
#'   public = list(
#'     initialize = function(id = "posum", param_vals = list()) {
#'       super$initialize(id, param_vals = param_vals,
#'         # declare "input" and "output" during construction here
#'         # training takes two 'numeric' and returns a 'numeric';
#'         # prediction takes 'NULL' and returns a 'character'.
#'         input = data.table::data.table(name = c("input1", "input2"),
#'           train = "numeric", predict = "NULL"),
#'         output = data.table::data.table(name = "output",
#'           train = "numeric", predict = "character")
#'       )
#'     }
#'   ),
#'   private = list(
#'     # PipeOp deriving classes must implement .train and
#'     # .predict; each taking an input list and returning
#'     # a list as output.
#'     .train = function(input) {
#'       sum = input[[1]] + input[[2]]
#'       self$state$sum = sum
#'       list(sum)
#'     },
#'     .predict = function(input) {
#'       list(letters[self$state$sum])
#'     }
#'   )
#' )
#' posum = PipeOpSumLetter$new()
#'
#' print(posum)
#'
#' posum$train(list(1, 2))
#' # note the name 'output' is the name of the output channel specified
#' # in the $output data.table.
#'
#' posum$predict(list(NULL, NULL))
#' @name PipeOp
#' @family mlr3pipelines backend related
#' @family PipeOps
#' @seealso https://mlr3book.mlr-org.com/list-pipeops.html
#' @export
PipeOp = R6Class("PipeOp",
  public = list(
    packages = NULL,
    state = NULL,
    input = NULL,
    output = NULL,
    .result = NULL,
    tags = NULL,

    initialize = function(id, param_set = ParamSet$new(), param_vals = list(), input, output, packages = character(0), tags = "abstract") {
      if (inherits(param_set, "ParamSet")) {
        private$.param_set = assert_param_set(param_set)
        private$.param_set_source = NULL
      } else {
        lapply(param_set, function(x) assert_param_set(eval(x)))
        private$.param_set_source = param_set
      }
      self$id = assert_string(id)

      self$param_set$values = insert_named(self$param_set$values, param_vals)
      self$input = assert_connection_table(input)
      self$output = assert_connection_table(output)
      self$packages = assert_character(packages, any.missing = FALSE, unique = TRUE)
      self$tags = assert_subset(tags, mlr_reflections$pipeops$valid_tags)
      private$.encapsulate = private$.learner$encapsulate %??% c(train = "none", predict = "none")  # propagate a learner's encapsulate in case of as_pipeop.Learner calls etc.
    },

    print = function(...) {
      type_table_printout = function(table) {
        strings = do.call(sprintf, cbind(fmt = "%s`[%s,%s]", table[, c("name", "train", "predict")]))
        strings = strwrap(paste(strings, collapse = ", "), indent = 2, exdent = 2)
        if (length(strings) > 6) {
          strings = c(strings[1:5], sprintf("  [... (%s lines omitted)]", length(strings) - 5))
        }
        gsub("`", " ", paste(strings, collapse = "\n"))
      }

      catf("PipeOp: <%s> (%strained)", self$id, if (self$is_trained) "" else "not ")
      catf("values: <%s>", as_short_string(self$param_set$values))
      catf("Input channels <name [train type, predict type]>:\n%s", type_table_printout(self$input))
      catf("Output channels <name [train type, predict type]>:\n%s", type_table_printout(self$output))
    },

    train = function(input) {
      pipeop_train(self, input)
    },
    predict = function(input) {
      pipeop_predict(self, input)
    }
  ),

  active = list(
    id = function(val) {
      if (!missing(val)) {
        private$.id = val
        if (!is.null(private$.param_set)) {
          # private$.param_set may be NULL if it is constructed dynamically by active binding
          private$.param_set$set_id = val
        }
      }
      private$.id
    },
    param_set = function(val) {
      if (is.null(private$.param_set)) {
        sourcelist = lapply(private$.param_set_source, function(x) eval(x))
        if (length(sourcelist) > 1) {
          private$.param_set = ParamSetCollection$new(sourcelist)
        } else {
          private$.param_set = sourcelist[[1]]
        }
        if (!is.null(self$id)) {
          private$.param_set$set_id = self$id
        }
      }
      if (!missing(val) && !identical(val, private$.param_set)) {
        stop("param_set is read-only.")
      }
      private$.param_set
    },
    predict_type = function(val) {
      if (!missing(val)) {
        if (!identical(val, private$.learner)) {
          stop("$predict_type is read-only.")
        }
      }
      return(NULL)
    },
    innum = function() nrow(self$input),
    outnum = function() nrow(self$output),
    is_trained = function() !is.null(self$state),
    hash = function() {
      digest(list(class(self), self$id, lapply(self$param_set$values, function(val) {
        # ideally we would just want to hash `param_set$values`, but one of the values
        # could be an R6 object with a `$hash` slot as well, in which case we take that
        # slot's value. This is to avoid different hashes from essentially the same
        # objects.
        # In the following we also avoid accessing `val$hash` twice, because it could
        # potentially be an expensive AB.
        if (is.environment(val) && !is.null({vhash = get0("hash", val, mode = "any", inherits = FALSE, ifnotfound = NULL)})) {
          vhash
        } else {
          val
        }
      })), algo = "xxhash64")
    },
    timings = function(rhs) {
      assert_ro_binding(rhs)
      set_names(c(self$state$train_time %??% NA_real_, self$state$predict_time %??% NA_real_), c("train", "predict"))
    },
    log = function(rhs) {
      assert_ro_binding(rhs)
      self$state$log
    },
    warnings = function(rhs) {
      assert_ro_binding(rhs)
      get_log_condition(self$state, "warning")
    },
    errors = function(rhs) {
      assert_ro_binding(rhs)
      get_log_condition(self$state, "error")
    },
    encapsulate = function(rhs) {
      if (missing(rhs)) {
        return(private$.encapsulate)
      }
      assert_character(rhs)
      assert_names(names(rhs), subset.of = c("train", "predict"))
      private$.encapsulate = insert_named(c(train = "none", predict = "none"), rhs)
    }
  ),

  private = list(
    deep_clone = function(name, value) {
      if (!is.null(private$.param_set_source)) {
        private$.param_set = NULL  # required to keep clone identical to original, otherwise tests get really ugly
        if (name == ".param_set_source") {
          value = lapply(value, function(x) {
            if (inherits(x, "R6")) x$clone(deep = TRUE) else x
          })
        }
      }
      if (is.environment(value) && !is.null(value[[".__enclos_env__"]])) {
        return(value$clone(deep = TRUE))
      }
      value
    },
    .train = function(input) stop("abstract"),
    .predict = function(input) stop("abstract"),
    .param_set = NULL,
    .param_set_source = NULL,
    .id = NULL,
    .encapsulate = NULL
  )
)

# Asserts that input and output tables are correctly specified
# @param table: `data.table`: either input or output
assert_connection_table = function(table) {
  varname = deparse(substitute(table))
  assert_data_table(table, .var.name = varname, min.rows = 1)
  assert_names(names(table), permutation.of = c("name", "train", "predict"), .var.name = varname)
  assert_character(table$name, any.missing = FALSE, unique = TRUE, .var.name = paste0("'name' column in ", varname))
  if (!all(multiplicity_type_nesting_level(table$train) == multiplicity_type_nesting_level(table$predict))) {
    stop("Multiplicity during train and predict conflicts.")
  }
  table
}

# Checks that data conforms to the type specifications given
# Handles multiplicities: if a type is in square brackets ("[<TYPE>]"), then a "Multiplicity" that contains the type is checked.
# Yes, this can handle nested multiplicities: "[[<TYPE>]]" etc. works.
# @param data: `list of any`: is either the input or output given to a train/predict function. it is checked to be a *list* first
#   and then to have the types as given by the `$input` or `$output` data.table
# @param direction: `character(1)`: is either `"input"` or `"output"`
# @param operation: `character(1)`: is either `"train"` or `"predict"`
# @return an instance of data, possibly converted, with names added according to `$input`/`$output` "name" column
check_types = function(self, data, direction, operation) {
  typetable = self[[direction]]
  if (direction == "input" && "..." %in% typetable$name) {
    assert_list(data, min.len = nrow(typetable) - 1)
    typetable = typetable[rep(1:.N, ifelse(get("name") == "...", length(data) - nrow(typetable) + 1, 1))]
  } else {
    assert_list(data, len = nrow(typetable))
  }

  check_item = function(data_element, typereq, varname) {
    if (multiplicity_type_nesting_level(typereq, varname)) {
      # unpack multiplicity
      assert_multiplicity(data_element, varname)
      typereq = substr(typereq, 2, nchar(typereq) - 1)
      for (midx in seq_along(data_element)) {
        # recursively call check_item for each multiplicity-item
        data_element[midx] = list(check_item(data_element[[midx]], typereq, sprintf("Multiplicity element %s of %s", midx, varname)))
      }
      # done checking, return early.
      return(data_element)
    }
    if (is.Multiplicity(data_element)) {
      stopf("%s contained Multiplicity when it shouldn't have.", data_element)
    }
    if (typereq == "*") return(data_element)
    if (typereq %in% class(data_element)) return(data_element)
    autoconverter = get_autoconverter(typereq)
    msg = ""
    if (!is.null(autoconverter)) {
      mlr3misc::require_namespaces(autoconverter$packages,
        sprintf("The following packages are required to convert object of class %s to class %s: %%s.", class(data_element)[1], typereq))
      msg = tryCatch({
        data_element = autoconverter$fun(data_element)
        ""
      }, error = function(e) sprintf("\nConversion from given data to %s produced message:\n%s.", typereq, e$message))
    }
    assert_class(data_element, typereq, .var.name = paste0(varname, msg))
  }

  for (idx in seq_along(data)) {
    data[idx] = list(check_item(data[[idx]], typetable[[operation]][[idx]],
      varname = sprintf("%s %s (\"%s\") of PipeOp %s",
        direction, idx, self$input$name[[idx]], self$id)))
  }
  names(data) = typetable$name
  data
}

# get the number of `[` `]` nestings of a variable name
# E.g. multiplicity_type_nesting_level(c("Task", "[Prediction]", "[[*]]")) --> c(0, 1, 2)
# @param str: `character`: type descriptors to check
# @param varname `character(1)`: where the value is found, used to print error message
# @return `integer`
multiplicity_type_nesting_level = function(str, varname) {
  beginning = map_int(gregexpr("^\\[*", str), attr, "match.length")
  end = map_int(gregexpr("\\]*$", str), attr, "match.length")
  if (any(beginning != end)) {
    stopf("Invalid type(s) %s in %s: square bracket mismatch.", str_collapse(str[beginning != end]), varname)
  }
  beginning
}

# unpacks pipeop arguments with multiplicities, if necessary, into (possibly named) lists that can be iterated over
# @param input: `list` of multiplicities: multiplicities to unpack
# @param expected_nesting_level: `integer`: expected nesting level of the multiplicities
# @param inputnames: `character`: names of the resulting lists
# @param poid: `character(1)`: character id of the PipeOp
# @return `list`
unpack_multiplicities = function(input, expected_nesting_level, inputnames, poid) {
  assert_list(input)
  unpacking = mapply(multiplicity_nests_deeper_than, input, expected_nesting_level)
  if (!any(unpacking)) {
    return(NULL)  # no unpacking
  }
  prototype_index = which(unpacking)[[1]]
  prototype = input[[prototype_index]]
  if (sum(unpacking) > 1) {
    # check that all elements being unpacked are the same multiplicity (length, names)
    # in the future we may be a bit more lax here and allow "vectorization" or cartesian products, but
    # for that we may rather want to have explicit pipeops
    for (comparing_index in which(unpacking)[-1]) {
      comparing = input[[comparing_index]]
      if (length(comparing) != length(prototype) || !identical(names(comparing), names(prototype))) {
        stopf("Input of %s has bad multiplicities: %s has different length and/or names than %s.",
          poid, inputnames[[prototype_index]], inputnames[[comparing_index]])
      }
    }
  }
  set_names(map(seq_along(prototype), function(idx) {
    map_at(input, unpacking, function(x) x[[idx]])
  }), names(prototype))
}

# evaluate multiplicities
# @param self: typically a `PipeOp`
# @param unpacked: `list` of unpacked multiplicities
# @param evalcall: either `train` or `predict`
# @param instate: typically `self$state`
evaluate_multiplicities = function(self, unpacked, evalcall, instate) {
  force(instate)
  on.exit({self$state = instate})
  if (!is.null(instate)) {
    if (!is.Multiplicity(instate)) {
      stopf("PipeOp %s received multiplicity input but state was not a multiplicity.", self$id)
    }
    if (length(instate) != length(unpacked) || !identical(names(instate), names(unpacked))) {
      stopf("PipeOp %s received multiplicity input but state had different length / names than input.", self$id)
    }
  }
  result = imap(unpacked, function(input, reference) {
    self$state = if (!is.null(instate)) instate[[reference]]
    list(output = self[[evalcall]](input), state = self$state)
  })
  on.exit({self$state = as.Multiplicity(map(result, "state"))})

  map(transpose_list(map(result, "output")), as.Multiplicity)
}

pipeop_train = function(pipeop, input) {
  # This wrapper calls pipeop$train, and additionally performs some basic checks that the training was successful.
  # Exceptions here are possibly encapsulated, so that they get captured and turned into log messages.
  train_wrapper = function(pipeop, input) {
    output = get_private(pipeop)$.train(input)

    if (is.null(output)) {
      stopf("PipeOp '%s' on input '%s' returned NULL during internal train()", pipeop$id, deparse(substitute(input)))
    }

    output
  }

  pipeop$state = NULL  # reset to untrained state first
  #require_namespaces(pipeop$packages)

  if (every(input, is_noop)) {
    pipeop$state = NO_OP
    return(named_list(pipeop$output$name, NO_OP))
  }

  unpacked = unpack_multiplicities(input, multiplicity_type_nesting_level(pipeop$input$train), pipeop$input$name, pipeop$id)
  if (!is.null(unpacked)) {
    return(evaluate_multiplicities(pipeop, unpacked, "train", NULL))
  }

  input = check_types(pipeop, input, "input", "train")
  on.exit({pipeop$state = NULL})  # if any of the following fails, make sure to reset pipeop$state

  lg$debug("Calling train method of PipeOp '%s' on input '%s'",
    pipeop$id, deparse(substitute(input)), pipeop = pipeop$clone())

  # call train_wrapper with encapsulation
  result = encapsulate(pipeop$encapsulate["train"],
    .f = train_wrapper,
    .args = list(pipeop = pipeop, input = input),
    .pkgs = pipeop$packages,
    .seed = NA_integer_
  )

  output = check_types(pipeop, result$result, "output", "train")
  on.exit()  # don't reset state any more

  pipeop$state$log = append_log(pipeop$state$log, "train", result$log$class, result$log$msg)
  pipeop$state$train_time = result$elapsed

  if (is.null(output)) {
    lg$debug("PipeOp '%s' on input '%s' failed to return a state",
      pipeop$id, deparse(substitute(input)), pipeop = pipeop$clone(), messages = result$log$msg)
  } else {
    lg$debug("PipeOp '%s' on input '%s' succeeded to return a state",
      pipeop$id, deparse(substitute(input)), pipeop = pipeop$clone(), messages = result$log$msg)
  }

  output
}

pipeop_predict = function(pipeop, input) {
  # This wrapper calls pipeop$predict, and additionally performs some basic checks that the prediction was successful.
  # Exceptions here are possibly encapsulated, so that they get captured and turned into log messages.
  predict_wrapper = function(pipeop, input) {
    # NOTE: may actually be sensible to check this
    #if (is.null(pipeop$state)) {
    #  stopf("No trained state available for PipeOp '%s' on input '%s'", pipeop$id, deparse(substitute(input)))
    #}

    get_private(pipeop)$.predict(input)
  }

  # need to load packages in train *and* predict, because they might run in different R instances
  #require_namespaces(pipeop$packages)

  if (every(input, is_noop)) {
    return(named_list(pipeop$output$name, NO_OP))
  }

  if (is_noop(pipeop$state)) {
    stopf("Pipeop %s got NO_OP during train but no NO_OP during predict.", pipeop$id)
  }

  unpacked = unpack_multiplicities(input, multiplicity_type_nesting_level(pipeop$input$predict), pipeop$input$name, pipeop$id)
  if (!is.null(unpacked)) {
    return(evaluate_multiplicities(pipeop, unpacked, "predict", pipeop$state))
  }

  input = check_types(pipeop, input, "input", "predict")

  # call predict with encapsulation
  lg$debug("Calling predict method of PipeOp '%s' on input '%s'",
    pipeop$id, deparse(substitute(input)), pipeop = pipeop$clone())

  result = encapsulate(
    pipeop$encapsulate["predict"],
    .f = predict_wrapper,
    .args = list(pipeop = pipeop, input = input),
    .pkgs = pipeop$packages,
    .seed = NA_integer_
  )

  output = check_types(pipeop, result$result, "output", "predict")

  pipeop$state$log = append_log(pipeop$state$log, "predict", result$log$class, result$log$msg)
  pipeop$state$predict_time = result$elapsed

  output
}

#FIXME: need this from mlr3
assert_ro_binding = mlr3:::assert_ro_binding
get_private = mlr3:::get_private
append_log = mlr3:::append_log
get_log_condition = mlr3:::get_log_condition

