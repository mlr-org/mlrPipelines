#' @title Graph Base Class
#' @format [R6Class] Graph
#'
#' @usage NULL
#' @format [`R6Class`].
#'
#' @description
#' A [`Graph`] is a representation of a machine learning pipeline graph. It can be *trained*, and subsequently used for *prediction*.
#'
#' A [`Graph`] is most useful when used together with [`Learner`][mlr3::Learner] objects encapsulated as [`PipeOpLearner`]. In this case,
#' the [`Graph`] produces [`Prediction`][mlr3::Prediction] data during its `$predict()` phase and can be used as a [`Learner`][mlr3::Learner]
#' itself (using the [`GraphLearner`] wrapper). However, the [`Graph`] can also be used without [`Learner`][mlr3::Learner] objects to simply
#' perform preprocessing of data, and, in principle, does not even need to handle data at all but can be used for general processes with
#' dependency structure (although the [`PipeOp`]s for this would need to be written).
#'
#' @section Construction:
#' ```
#' Graph$new()
#' ```
#'
#' @section Internals:
#' A [`Graph`] is made up of a list of [`PipeOp`]s, and a [`data.table`] of edges. Both for training and prediction, the [`Graph`]
#' performs topological sorting of the [`PipeOp`]s and executes their respective `$train()` or `$predict()` functions in order, moving
#' the [`PipeOp`] results along the edges as input to other [`PipeOp`]s.
#'
#' @section Fields:
#' * `pipeops` :: named `list` of [`PipeOp`] \cr
#'   Contains all [`PipeOp`]s in the [`Graph`], named by the [`PipeOp`]'s `$id`s.
#' * `edges` :: [`data.table`]  with columns `src_id` (`character`), `src_channel` (`character`), `dst_id` (`character`), `dst_channel` (`character`)\cr
#'   Table of connections between the [`PipeOp`]s. A [`data.table`]. `src_id` and `dst_id` are `$id`s of [`PipeOp`]s that must be present in
#'   the `$pipeops` list. `src_channel` and `dst_channel` must respectively be `$output` and `$input` channel names of the
#'   respective [`PipeOp`]s.
#' * `is_trained` :: `logical(1)` \cr
#'   Is the [`Graph`], i.e. are all of its [`PipeOp`]s, trained, and can the [`Graph`] be used for prediction?
#' * `lhs` :: `character` \cr
#'   Ids of the 'left-hand-side' [`PipeOp`]s that have some unconnected input channels and therefore act as [`Graph`] input layer.
#' * `rhs` :: `character` \cr
#'   Ids of the 'right-hand-side' [`PipeOp`]s that have some unconnected output channels and therefore act as [`Graph`] output layer.
#' * `input` :: [`data.table`] with columns `name` (`character`), `train` (`character`), `predict` (`character`), `op.id` (`character`), `channel.name` (`character`)\cr
#'   Input channels of the [`Graph`]. For each channel lists the name, input type during training, input type during prediction,
#'   [`PipeOp`] `$id` of the [`PipeOp`] the channel pertains to, and channel name as the [`PipeOp`] knows it.
#' * `output` :: [`data.table`] with columns `name` (`character`), `train` (`character`), `predict` (`character`), `op.id` (`character`), `channel.name` (`character`)\cr
#'   Output channels of the [`Graph`]. For each channel lists the name, output type during training, output type during prediction,
#'   [`PipeOp`] `$id` of the [`PipeOp`] the channel pertains to, and channel name as the [`PipeOp`] knows it.
#' * `packages` :: `character`\cr
#'   Set of all required packages for the various methods in the [`Graph`], a set union of all required packages of all contained
#'   [`PipeOp`] objects.
#' * `state` :: named `list`\cr
#'   Get / Set the `$state` of each of the members of [`PipeOp`].
#' * `param_set` :: [`ParamSet`][paradox::ParamSet]\cr
#'   Parameters and parameter constraints. Parameter values are in `$param_set$values`. These are the union of `$param_set`s
#'   of all [`PipeOp`]s in the [`Graph`]. Parameter names
#'   as seen by the [`Graph`] have the naming scheme `<PipeOp$id>.<PipeOp original parameter name>`.
#'   Changing `$param_set$values` also propagates the changes directly to the contained
#'   [`PipeOp`]s and is an alternative to changing a [`PipeOp`]s `$param_set$values` directly.
#' * `hash` :: `character(1)` \cr
#'   Stores a checksum calculated on the [`Graph`] configuration, which includes all [`PipeOp`] hashes
#'   (and therefore their `$param_set$values`) and a hash of `$edges`.
#' * `keep_results` :: `logical(1)` \cr
#'   Whether to store intermediate results in the [`PipeOp`]'s `$.result` slot, mostly for debugging purposes. Default `FALSE`.
#'
#' @section Methods:
#' * `ids(sorted = FALSE)` \cr
#'   (`logical(1)`) -> `character` \cr
#'   Get IDs of all [`PipeOp`]s. This is in order that [`PipeOp`]s were added if
#'   `sorted` is `FALSE`, and topologically sorted if `sorted` is `TRUE`.
#' * `add_pipeop(op)` \cr
#'   ([`PipeOp`] | [`Learner`][mlr3::Learner] | [`Filter`][mlr3filters::Filter] | `...`) -> `self` \cr
#'   Mutates [`Graph`] by adding a [`PipeOp`] to the [`Graph`]. This does not add any edges, so the new [`PipeOp`]
#'   will not be connected within the [`Graph`] at first.\cr
#'   Instead of supplying a [`PipeOp`] directly, an object that can naturally be converted to a [`PipeOp`] can also
#'   be supplied, e.g. a [`Learner`][mlr3::Learner] or a [`Filter`][mlr3filters::Filter]; see [`as_pipeop()`].
#' * `add_edge(src_id, dst_id, src_channel = NULL, dst_channel = NULL)` \cr
#'   (`character(1)`, `character(1)`,
#'   `character(1)` | `numeric(1)` | `NULL`,
#'   `character(1)` | `numeric(1)` | `NULL`) -> `self` \cr
#'   Add an edge from [`PipeOp`] `src_id`, and its channel `src_channel`
#'   (identified by its name or number as listed in the [`PipeOp`]'s `$output`), to [`PipeOp`] `dst_id`'s
#'   channel `dst_channel` (identified by its name or number as listed in the [`PipeOp`]'s `$input`).
#'   If source or destination [`PipeOp`] have only one input / output channel and `src_channel` / `dst_channel`
#'   are therefore unambiguous, they can be omitted (i.e. left as `NULL`).
#' * `plot(html)` \cr
#'   (`logical(1)`) -> `NULL` \cr
#'   Plot the [`Graph`], using either the \pkg{igraph} package (for `html = FALSE`, default) or
#'   the `visNetwork` package for `html = TRUE` producing a [`htmlWidget`][htmlwidgets::htmlwidgets].
#'   The [`htmlWidget`][htmlwidgets::htmlwidgets] can be rescaled using [`visOptions`][visNetwork::visOptions].
#' * `print(dot = FALSE, dotname = "dot", fontsize = 24L)` \cr
#'   (`logical(1)`, `character(1)`, `integer(1)`) -> `NULL` \cr
#'   Print a representation of the [`Graph`] on the console. If `dot` is `FALSE`, output is a table with one row for each contained [`PipeOp`] and
#'   columns `ID` (`$id` of `PipeOp`), `State` (short representation of `$state` of [`PipeOp`]), `sccssors` ([`PipeOp`]s that
#'   take their input directly from the [`PipeOp`] on this line), and `prdcssors` (the [`PipeOp`]s that produce the data
#'   that is read as input by the [`PipeOp`] on this line). If `dot` is `TRUE`, print a DOT representation of the [`Graph`] on the console.
#'   The DOT output can be named via the argument `dotname` and the `fontsize` can also be specified.
#' * `set_names(old, new)` \cr
#'   (`character`, `character`) -> `self` \cr
#'   Rename [`PipeOp`]s: Change ID of each [`PipeOp`] as identified by `old` to the corresponding item in `new`. This should be used
#'   instead of changing a [`PipeOp`]'s `$id` value directly!
#' * `update_ids(prefix = "", postfix = "")` \cr
#'   (`character`, `character`) -> `self` \cr
#'   Pre- or postfix [`PipeOp`]'s existing ids. Both `prefix` and `postfix` default to `""`, i.e. no changes.
#' * `train(input, single_input = TRUE)` \cr
#'   (`any`, `logical(1)`) -> named `list`\cr
#'   Train [`Graph`] by traversing the [`Graph`]s' edges and calling all the [`PipeOp`]'s `$train` methods in turn.
#'   Return a named `list` of outputs for each unconnected
#'   [`PipeOp`] out-channel, named according to the [`Graph`]'s `$output` `name` column. During training, the `$state`
#'   member of each [`PipeOp`]s will be set and the `$is_trained` slot of the `Graph` (and each individual `PipeOp`) will
#'   consequently be set to `TRUE`.\cr
#'   If `single_input` is `TRUE`, the `input` value will be sent to each unconnected [`PipeOp`]'s input channel
#'   (as listed in the [`Graph`]'s `$input`). Typically, `input` should be a [`Task`][mlr3::Task], although this is dependent
#'   on the [`PipeOp`]s in the [`Graph`]. If `single_input` is `FALSE`, then
#'   `input` should be a `list` with the same length as the [`Graph`]'s `$input` table has rows; each list item will be sent
#'   to a corresponding input channel of the [`Graph`]. If `input` is a named `list`, names must correspond to input channel
#'   names (`$input$name`) and inputs will be sent to the channels by name; otherwise they will be sent to the channels
#'   in order in which they are listed in `$input`.
#' * `predict(input, single_input = TRUE)` \cr
#'   (`any`, `logical(1)`) -> `list` of `any` \cr
#'   Predict with the [`Graph`] by calling all the [`PipeOp`]'s `$train` methods. Input and output, as well as the function
#'   of the `single_input` argument, are analogous to `$train()`.
#'
#' @name Graph
#' @family mlr3pipelines backend related
#' @export
#' @examples
#' library("mlr3")
#'
#' g = Graph$new()$
#'   add_pipeop(PipeOpScale$new(id = "scale"))$
#'   add_pipeop(PipeOpPCA$new(id = "pca"))$
#'   add_edge("scale", "pca")
#' g$input
#' g$output
#'
#' task = tsk("iris")
#' trained = g$train(task)
#' trained[[1]]$data()
#'
#' task$filter(1:10)
#' predicted = g$predict(task)
#' predicted[[1]]$data()
Graph = R6Class("Graph",
  public = list(
    pipeops = NULL,
    edges = NULL,
    keep_results = NULL,
    initialize = function() {
      self$pipeops = list()
      self$edges = setDT(named_list(c("src_id", "src_channel", "dst_id", "dst_channel"), character()))
      self$keep_results = FALSE
    },

    ids = function(sorted = FALSE) {
      assert_flag(sorted)
      if (!sorted || nrow(self$edges) == 0L) {
        return(names2(self$pipeops))
      }

      tmp = self$edges[, list(parents = list(unique(src_id))), by = list(id = dst_id)]
      orphans = setdiff(names(self$pipeops), self$edges$dst_id)  # the ones without parents
      if (length(orphans)) {
        # if orphans is empty either the Graph is empty (won't happen here) or has cycles, in
        # which case we still call topo_sort to get unified error messages.
        tmp = rbind(tmp, data.table(id = orphans, parents = list(character(0L))))
      }
      topo_sort(tmp)$id
    },

    add_pipeop = function(op) {
      op = as_pipeop(op)
      if (op$id %in% names(self$pipeops)) {
        stopf("PipeOp with id '%s' already in Graph", op$id)
      }
      self$pipeops[[op$id]] = op

      if (!is.null(private$.param_set)) {
        # param_set is built on-demand; if it has not been requested before, its value may be NULL
        # and we don't need to add anything.
        private$.param_set$add(self$pipeops[[op$id]]$param_set)
      }
      invisible(self)
    },

    add_edge = function(src_id, dst_id, src_channel = NULL, dst_channel = NULL) {

      if (!length(self$pipeops)) {
        stop("Cannot add edge to empty Graph.")
      }
      assert_choice(src_id, names(self$pipeops))
      assert_choice(dst_id, names(self$pipeops))
      if (is.null(src_channel)) {
        if (length(self$pipeops[[src_id]]$output$name) > 1) {
          stopf("src_channel must not be NULL if src_id pipeop has more than one output channel.")
        }
        src_channel = 1L
      }
      if (is.null(dst_channel)) {
        if (length(self$pipeops[[dst_id]]$input$name) > 1) {
          stopf("dst_channel must not be NULL if src_id pipeop has more than one input channel.")
        }
        dst_channel = 1L
      }
      assert(
        check_integerish(src_channel, lower = 1L,
          upper = nrow(self$pipeops[[src_id]]$output), any.missing = FALSE),
        check_choice(src_channel, self$pipeops[[src_id]]$output$name)
      )
      if (is.numeric(src_channel)) {
        src_channel = self$pipeops[[src_id]]$output$name[src_channel]
      }
      assert(
        check_integerish(dst_channel, lower = 1,
          upper = nrow(self$pipeops[[dst_id]]$input), any.missing = FALSE),
        check_choice(dst_channel, self$pipeops[[dst_id]]$input$name)
      )
      if (is.numeric(dst_channel)) {
        dst_channel = self$pipeops[[dst_id]]$input$name[dst_channel]
      }

      types_src = self$pipeops[[src_id]]$output[get("name") == src_channel, c("train", "predict")]
      types_dst = self$pipeops[[dst_id]]$input[get("name") == dst_channel, c("train", "predict")]

      if (!are_types_compatible(strip_multiplicity_type(types_src$train), strip_multiplicity_type(types_dst$train))) {
        stopf("Output type of PipeOp %s during training (%s) incompatible with input type of PipeOp %s (%s)",
          src_id, types_src$train, dst_id, types_dst$train)
      }
      if (!are_types_compatible(strip_multiplicity_type(types_src$predict), strip_multiplicity_type(types_dst$predict))) {
        stopf("Output type of PipeOp %s during prediction (%s) incompatible with input type of PipeOp %s (%s)",
          src_id, types_src$predict, dst_id, types_dst$predict)
      }

      bad_rows = (self$edges$dst_id == dst_id & self$edges$dst_channel == dst_channel & self$edges$dst_channel != "...")
      if (any(bad_rows)) {
        prior_con = self$edges[bad_rows]
        stopf("Cannot add multiple edges to a channel.\n%s",
          paste(sprintf("Channel %s of node %s already connected to channel %s of node %s.\nMultiple connections to input channels is only possible for vararg (i.e. '...') channels.",
            prior_con$src_channel, prior_con$src_id, prior_con$dst_channel, prior_con$dst_id), collapse = "\n"))
      }
      row = data.table(src_id, src_channel, dst_id, dst_channel)
      old_edges = self$edges
      self$edges = rbind(self$edges, row)
      # check for loops
      on.exit({
        self$edges = old_edges
      })
      self$ids(sorted = TRUE)  # if we fail here, edges get reset.
      on.exit()
      invisible(self)
    },

    plot = function(html = FALSE) {
      assert_flag(html)
      if (!length(self$pipeops)) {
        cat("Empty Graph, not plotting.\n")
        return(invisible(NULL))
      }
      require_namespaces("igraph")
      if (nrow(self$edges) == 0L) {
        ig = igraph::make_empty_graph()
        extra_vertices = names(self$pipeops)
      } else {
        df = self$edges[, list(from = src_id, to = dst_id)]
        df = rbind(df, self$input[, list(from = "<INPUT>", to = op.id)])
        output = self$output
        if (nrow(output) > 1) {
          # In case we have multiple outputs, we add an output for every final node
          df = rbind(df, output[, list(from = op.id, to = paste0("<OUTPUT>\n", name))])
        } else {
          df = rbind(df, output[, list(from = op.id, to = "<OUTPUT>")])
        }
        ig = igraph::graph_from_data_frame(df)
        extra_vertices = setdiff(names(self$pipeops), c(df$from, df$to))
      }
      ig = igraph::add_vertices(ig, length(extra_vertices), name = extra_vertices)
      layout = igraph::layout_with_sugiyama(ig)$layout
      if (!is.matrix(layout)) {
        layout = t(layout)  # bug in igraph, dimension is dropped
      }
      if (html) {
        require_namespaces("visNetwork")
        ig_data = visNetwork::toVisNetworkData(ig)
        # Map color / shape of the nodes depending on the node type (input, output, actual node)
        ig_data$nodes$shape = map_chr(ig_data$nodes$id, function(x) switch(x, "<INPUT>" = "database", "<OUTPUT>" = "ellipse", "box"))
        ig_data$nodes$color = map_chr(ig_data$nodes$id, function(x) switch(x, "<INPUT>" = "rgba(0,204,102,0.2)", "<OUTPUT>" = "rgba(255,51,51,0.2)", "lightblue"))
        ig_data$nodes$value = map_dbl(ig_data$nodes$id, function(x) switch(x, "<INPUT>" = .8, "<OUTPUT>" = .8, 1))

        # This constructs the info displayed when hovering over the node in html:
        # Basically gets the print() output of the PipeOp.
        ig_data$nodes$title = map_chr(ig_data$nodes$id, function(node) {
          null_str = function(x) x %??% "NULL"
          if (node == "<INPUT>") {
            txt = paste0("Input:<br>Name: ", self$input$name, "<br>Train: ", null_str(self$input$train), "<br>Predict: ", null_str(self$input$predict))
          } else if (grepl("<OUTPUT>", node)) {
            if (nrow(self$output) > 1) {
              out = self$output[self$output$name == gsub("<OUTPUT>\n", "", node), ]  # Deal with multiple outputs
            } else {
              out = self$output  # Standard case, single output
            }
            txt = paste0("Output:<br>Name: ", out$name, "<br>Train: ", null_str(out$train), "<br>Predict: ", null_str(out$predict))
          } else {
            txt = paste((gsub("<(.*)>", utils::capture.output(self$pipeops[[node]]), replacement = "<b>\\1</b>", perl = TRUE)), collapse = "<br>")
          }
          # Deal with special case: multiple edges between two pipeops
          if (length(txt) > 1) txt = paste0(txt, collapse = "<br>")
          return(txt)
        })
        ig_data$nodes$title = paste0("<p>", ig_data$nodes$title, "</p>")
        edges = NROW(ig_data$edges)
        if (edges) ig_data$edges$color = "lightblue"  # Only if more than one pipeop
        # Visualize the nodes
        p = visNetwork::visNetwork(nodes = ig_data$nodes, edges = ig_data$edges, height = "400px", width = "50%")
        if (edges) { # Only if more than one pipeop
          # Set up layout
          p = visNetwork::visIgraphLayout(p, layout = "layout_with_sugiyama", type = "full")
          # Draw edges between points
          p = visNetwork::visEdges(p, arrows = "to", smooth = list(enabled = FALSE, forceDirection = "vertical"))
        }
        p
      } else {
        suppressWarnings(graphics::plot(ig, layout = layout))  # suppress partial matching warning
      }
    },

    print = function(dot = FALSE, dotname = "dot", fontsize = 24L) {
      assert_flag(dot)
      if (dot) {
        assert_character(dotname, any.missing = FALSE, len = 1L)
        assert_count(fontsize, positive = TRUE)
        if (!length(self$pipeops)) {
          return(cat(paste0("digraph ", dotname, " {\n", "", "\n}\n")))
        }
        if (nrow(self$edges) == 0L) {
          all_names = gsub("\\.", "_", self$ids(TRUE))
          dot = paste0(paste0(seq_along(all_names), " [label=", '"', all_names, '"',
            ",fontsize=", fontsize, ']'),
            collapse = ";\n")
        } else {
          df = self$edges[, list(from = src_id, to = dst_id)]
          df = rbind(df, self$input[, list(from = "INPUT", to = op.id)])
          output = self$output
          if (nrow(output) > 1L) {
            df = rbind(df, output[, list(from = op.id, to = paste0("OUTPUT\n", name))])
          } else {
            df = rbind(df, output[, list(from = op.id, to = "OUTPUT")])
          }
          ids = self$ids(TRUE)
          extra_vertices = setdiff(ids, c(df$from, df$to))

          all_names = unique(unlist(df))
          df = data.table::setDT(mlr3misc::map(df, function(x) match(x, all_names)))
          gr = paste0(map(seq_len(nrow(df)), function(x) {
            paste0(df[x, ][[1L]], " -> ", df[x, ][[2L]])
          }), collapse = ";\n")

          all_names = gsub("\\.", "_", all_names)
          labels = paste0(unlist(mlr3misc::map(unique(unlist(df)), function(x) {
            paste0(x, " [label=", '"', all_names[x], '"', ",fontsize=", fontsize, ']')
          })), collapse = ";\n")
          dot = paste0(gr, ";\n", labels)

          if (length(extra_vertices)) {
             ev_names = gsub("\\.", "_", extra_vertices)
             ev = paste0(paste0(length(all_names) + seq_along(ev_names), " [label=",
               '"', ev_names, '"', ",fontsize=", fontsize, ']'),
               collapse = ";\n")
             dot = paste0(dot, ";\n", ev)
          }
        }
        cat(paste0("digraph ", dotname, " {\n", dot, "\n}\n"))
      } else {
        # print table <id>, <state>, where <state> is `class(pipeop$state)`
        lines = rbindlist(map(self$pipeops[self$ids(sorted = TRUE)], function(pipeop) {
          data.table(ID = pipeop$id, State = sprintf("<%s>",
            map_values(class(pipeop$state)[1], "NULL", "<UNTRAINED>")))
        }), use.names = TRUE)
        if (nrow(lines)) {
          prd = self$edges[, list(prdcssors = paste(unique(src_id), collapse = ",")), by = list(ID = dst_id)]
          scc = self$edges[, list(sccssors = paste(unique(dst_id), collapse = ",")), by = list(ID = src_id)]
          lines = scc[prd[lines, on = "ID"], on = "ID"][, c("ID", "State", "sccssors", "prdcssors")]
          lines[is.na(lines)] = ""
          catf("Graph with %s PipeOps:", nrow(lines))
          ## limit column width ##

          outwidth = getOption("width") %??% 80  # output width we want (default 80)
          colwidths = map_int(lines, function(x) max(nchar(x), na.rm = TRUE))  # original width of columns
          collimit = calculate_collimit(colwidths, outwidth)
          with_options(list(datatable.prettyprint.char = collimit), {
            print(lines, row.names = FALSE)
          })
        } else {
          cat("Empty Graph.\n")
        }
        invisible(self)
      }
    },

    # Mutator to change PipeOp IDs
    # Modifies both the index in $pipeops, as well as the respective PipeOp's ID. Do this here and not
    # by setting `graph$pipeops[[x]]$id <- y`!
    set_names = function(old, new) {
      ids = names2(self$pipeops)
      assert_subset(old, ids)
      assert_character(new, any.missing = FALSE)
      new_ids = map_values(ids, old, new)
      names(self$pipeops) = new_ids
      imap(self$pipeops, function(x, nn) x$id = nn)

      self$edges[, c("src_id", "dst_id") := list(map_values(src_id, old, new), map_values(dst_id, old, new))]
      invisible(self)
    },
    update_ids = function(prefix = "", postfix = "") {
      ids = names2(self$pipeops)
      self$set_names(ids, sprintf("%s%s%s", assert_string(prefix), ids, assert_string(postfix)))
      invisible(self)
    },

    train = function(input, single_input = TRUE) {
      graph_load_namespaces(self, "train")
      graph_reduce(self, input, "train", single_input)
    },

    predict = function(input, single_input = TRUE) {
      graph_load_namespaces(self, "predict")
      graph_reduce(self, input, "predict", single_input)
    }
  ),

  active = list(
    is_trained = function() all(map_lgl(self$pipeops, "is_trained")),
    lhs = function() unique(self$input$op.id),
    rhs = function() unique(self$output$op.id),
    input = function() {
      graph_channels(self$edges$dst_id, self$edges$dst_channel, self$pipeops, "input")
    },
    output = function() {
      graph_channels(self$edges$src_id, self$edges$src_channel, self$pipeops, "output")
    },
    packages = function() unique(unlist(map(self$pipeops, "packages"))),
    hash = function() {
      digest(
        list(map(self$pipeops, "hash"), self$edges),
        algo = "xxhash64")
    },
    param_set = function(val) {
      # FIXME: It would be nice if we didn't need to do this.
      if (is.null(private$.param_set)) {
        private$.param_set = ParamSetCollection$new(map(self$pipeops, "param_set"))
      }
      if (!missing(val) && !identical(val, private$.param_set)) {
        stop("param_set is read-only.")
      }
      private$.param_set
    },
    state = function(val) {
      if (!missing(val)) {
        assert_list(val, names = "unique", null.ok = TRUE)
        assert_subset(names(val), c(names(self$pipeops), "log", "train_time", "predict_time"))
        imap(self$pipeops, function(pipeop, pname) pipeop$state = val[[pname]])
        val
      } else {
        map(self$pipeops, "state")
      }
    }
  ),

  private = list(
    deep_clone = function(name, value) {
      private$.param_set = NULL  # required to keep clone identical to original, otherwise tests get really ugly
      switch(name,
        edges = copy(value),
        pipeops = map(value, function(x) x$clone(deep = TRUE)),
        value
      )
    },
    .param_set = NULL
  )
)

# build the '$input' or '$output' active binding
# ids, channels: columns of $edges DT. Either src_{ids,channels} or dst_{ids,channels}, depending on if we are input or output
# pipeops: list of pipeops
# direction: "input" or "output"
graph_channels = function(ids, channels, pipeops, direction) {
  if (!length(pipeops)) {
    return(data.table(name = character(), train = character(),
      predict = character(), op.id = character(), channel.name = character()))
  }
  map_dtr(pipeops, function(po) {

    # Note: This uses data.frame and is 20% faster than the fastest data.table I could come up with
    # (and factor 2 faster than a naive data.table implementation below).
    # $input and $output is actually a bottleneck for %>>%, so we want this to be fast.
    # Please don't change without benchmark.
    df = as.data.frame(po[[direction]], stringsAsFactors = FALSE)
    rows = df$name %nin% channels[ids == po$id]
    if (!any(rows)) {
      return(data.frame(name = character(),
        train = character(), predict = character(),
        op.id = character(), channel.name = character(),
        stringsAsFactors = FALSE))
    }
    df$op.id = po$id
    df = df[rows,
      c("name", "train", "predict", "op.id", "name")]
    df[[1]] = paste0(po$id, ".", df[[1]])
    names(df)[5] = "channel.name"
    df
  })
}

graph_channels_dt = function(ids, channels, pipeops, direction) {
  # for reference: this is the above in data.table
  if (!length(pipeops)) {
    return(data.table(name = character(), train = character(),
      predict = character(), op.id = character(), channel.name = character()))
  }
  map_dtr(pipeops, function(po) {
    po[[direction]][get("name") %nin% channels[ids == po$id],
      list(name = paste0(po$id, ".", get("name")),
        train = get("train"), predict = get("predict"), op.id = po$id, channel.name = get("name"))]
  })
}

# walk along Graph edges, evaluate [[fun]](), return named list of output
# self: Graph's `$self`
# input: as given by `$train`, `$predict`. single valued to be copied (if
#   `single_input` is `TRUE`) or (possibly named) list of values for each
#   incoming edge.
# fun: function of each `PipeOp` to call; should be `train` or
#   `predict`.
# single_input: whether `input` is to be copied to all input channels
#   (`TRUE`) or is a list with different input for each channel (`FALSE`).
graph_reduce = function(self, input, fun, single_input) {

  assert_flag(single_input)

  graph_input = self$input
  graph_output = self$output

  edges = copy(self$edges)

  # create virtual "__initial__" and "__terminal__" nodes with edges to inputs / outputs of graph.
  # if we have `single_input == FALSE` and one(!) vararg channel, we widen the vararg input
  # appropriately.
  if (!single_input && length(assert_list(input)) > nrow(graph_input) && "..." %in% graph_input$channel.name) {
    if (sum("..." == graph_input$channel.name) != 1) {
      stop("Ambiguous distribution of inputs to vararg channels.\nAssigning more than one input to vararg channels when there are multiple vararg inputs does not work.")
    }
    # repeat the "..." as often as necessary
    repeats = ifelse(graph_input$channel.name == "...", length(input) - nrow(graph_input) + 1, 1)
    graph_input = graph_input[rep(graph_input$name, repeats), , on = "name"]
  }

  edges = rbind(edges,
    data.table(src_id = "__initial__", src_channel = graph_input$name,
      dst_id = graph_input$op.id, dst_channel = graph_input$channel.name),
    data.table(src_id = graph_output$op.id, src_channel = graph_output$channel.name,
      dst_id = "__terminal__", dst_channel = graph_output$name))

  # add new column to store content that is sent along an edge
  edges$payload = list()

  if (!single_input) {
    # we need the input list length to be equal to the number of channels. This number was
    # already increased appropriately if there is a single vararg channel.
    assert_list(input, len = nrow(graph_input))
    # input can be a named list (will be distributed to respective edges) or unnamed.
    # if it is named, we check that names are unambiguous.
    if (!is.null(names(input))) {
      if (anyDuplicated(graph_input$name)) {
        # FIXME this will unfortunately trigger if there is more than one named input for a vararg channel.
        stopf("'input' must not be a named list because Graph %s input channels have duplicated names.", self$id)
      }
      assert_names(names(input), subset.of = graph_input$name)
      edges[list("__initial__", names(input)), "payload" := list(input), on = c("src_id", "src_channel")]
    } else {
      # don't rely on unique graph_input$name!
      edges[get("src_id") == "__initial__", "payload" := list(input)]
    }
  } else {
    edges[get("src_id") == "__initial__", "payload" := list(list(input))]
  }

  # get the topo-sorted pipeop ids
  ids = self$ids(sorted = TRUE)  # won't contain __initial__  or __terminal__ which are only in our local copy

  # walk over ids, learning each operator
  for (id in ids) {
    op = self$pipeops[[id]]
    input_tbl = edges[get("dst_id") == id, list(name = get("dst_channel"), payload = get("payload"))][op$input$name, , on = "name"]
    edges[get("dst_id") == id, "payload" := list(list(NULL))]
    input = input_tbl$payload
    names(input) = input_tbl$name

    lg$debug("Running PipeOp '%s$%s()'", id, fun, pipeop = op, input = input)

    output = op[[fun]](input)
    if (self$keep_results) {
      op$.result = output
    }
    edges[list(id, op$output$name), "payload" := list(output), on = c("src_id", "src_channel")]
  }

  # get payload of edges that go to terminal node.
  # can't use 'dst_id == "__terminal__", because output channel names for Graphs may be duplicated.
  output_tbl = edges[list(graph_output$op.id, graph_output$channel.name),
    c("dst_channel", "payload"), on = c("src_id", "src_channel")]
  output = output_tbl$payload
  names(output) = output_tbl$dst_channel
  filter_noop(output)
}

# load namespaces for graph and give an informative error message if this fails
# self: graph
# info: what is being done, probably "train" or "predict"
graph_load_namespaces = function(self, info) {
  assert_string(info)
  pkenv = new.env(parent = emptyenv())
  for (po in self$pipeops) {
    for (pkg in po$packages) {
      pkenv[[pkg]] = c(pkenv[[pkg]], po$id)
    }
  }
  errors = imap(as.list(pkenv), function(pipeops, package) {
    tryCatch({
      # not using requireNamespace here because a present package
      # could still throw an error while being loaded.
      suppressPackageStartupMessages(loadNamespace(package))
      NULL
    }, error = function(e) {
      sprintf("Error loading package %s (required by %s):\n    %s",
        package, str_collapse(pipeops, n = 4), e$message)
    })
  })
  errors = discard(errors, is.null)
  if (length(errors)) {
    stopf("Error during %s:\n  %s", info, str_collapse(errors, sep = "\n  "))
  }
}


#' @export
predict.Graph = function(object, newdata, ...) {
  if (!object$is_trained) {
    stop("Graph is not trained.")
  }
  output = object$output
  if (nrow(output) != 1) {
    stop("Graph has more than one output channel")
  }
  if (!are_types_compatible(output$predict, "Prediction")) {
    stop("Graph output type not 'Prediction' (or compatible with it)")
  }
  plain = "Task" %nin% class(newdata)
  if (plain) {
    assert_data_frame(newdata)
    newdata = TaskRegr$new("predicttask", as_data_backend(cbind(newdata, `...dummytarget...` = NA_real_)), "...dummytarget...")
  }
  result = object$predict(newdata)
  assert_list(result, types = "Prediction", any.missing = FALSE, len = 1)
  result = result[[1]]
  if (plain) {
    result = result$data$response %??% result$data$prob
  }
  result
}
