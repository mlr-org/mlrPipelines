context("Graph")

test_that("linear graph", {
  g = Graph$new()
  expect_equal(g$ids(sorted = TRUE), character(0))

  # FIXME: we should use "dummy" ops, so we can change properties of the ops at will
  # we should NOT use PipeOpNOP, because we want to check that $train/$predict actually does something.
  # FIXME: we should check packages of the graph
  op_ds = PipeOpSubsample$new()
  op_pca = PipeOpPCA$new()
  op_lrn = PipeOpLearner$new(mlr_learners$get("classif.rpart"))
  g$add_pipeop(op_ds)
  g$add_pipeop(op_pca)

  expect_graph(g)

  g$add_edge("subsample", "pca")

  expect_graph(g)

  expect_output(print(g), "Graph with 2 PipeOps.*subsample.*UNTRAINED.*pca.*UNTRAINED")



  inputs = mlr_tasks$get("iris")
  x = g$train(inputs)
  expect_task(x[[1]])

  expect_output(print(g), "Graph with 2 PipeOps.*subsample.*list.*pca.*prcomp")

  out = g$predict(inputs)
  expect_task(x[[1]])

  g$add_pipeop(op_lrn)

  expect_graph(g)

  g$add_edge("subsample", "classif.rpart")

  expect_error(g$add_edge("pca", "classif.rpart"),
    "Channel.*output.*of node.*subsample.*already connected to channel.*input.*of node classif.rpart")

  expect_error(g$add_pipeop(op_lrn), "PipeOp with id.*rpart.*already in Graph")

  expect_deep_clone(g, g$clone(deep = TRUE))
})

test_that("complex graph", {
  # test that debug pipeops exist
  expect_pipeop(PipeOpDebugBasic$new())
  expect_pipeop(PipeOpDebugMulti$new(1, 1))
  expect_pipeop(PipeOpDebugMulti$new(3, 2))

  expect_graph(PipeOpDebugBasic$new() %>>% PipeOpDebugMulti$new(1, 2) %>>% PipeOpDebugMulti$new(2, 1, "debug2"))

  expect_graph(PipeOpDebugBasic$new() %>>% PipeOpDebugMulti$new(1, 2) %>>%
    pipeline_greplicate(PipeOpDebugMulti$new(1, 2, "debug2"), 2))

  biggraph = PipeOpDebugBasic$new() %>>%
    PipeOpDebugMulti$new(1, 2) %>>%
    pipeline_greplicate(PipeOpDebugMulti$new(1, 2, "debug2"), 2) %>>%
    gunion(list(PipeOpDebugBasic$new("basictop"),
      PipeOpDebugMulti$new(2, 1, "debug2"),
      PipeOpDebugBasic$new("basicbottom"))) %>>%
    PipeOpDebugMulti$new(3, 1, "debug3")

  lines = strsplit(capture_output(biggraph$train(1)), "\n")[[1]]

  expect_set_equal(lines,
    c("Training debug.basic",
      "Training debug.multi with input list(input_1 = 1)",
      "Training debug2_1 with input list(input_1 = 2)",
      "Training debug2_2 with input list(input_1 = 3)",
      "Training basictop",
      "Training basicbottom",
      "Training debug2 with input list(input_1 = 4, input_2 = 4)",
      "Training debug3 with input list(input_1 = 3, input_2 = 5, input_3 = 5)"),
    info = paste0("'", lines, "'", collapse = "', '"))

  pdf(file = NULL)  # don't show plot. It is annoying.
  biggraph$plot()
  dev.off()

  p = biggraph$plot(TRUE)
  expect_class(p, "htmlwidget")
  expect_class(p, "visNetwork")
})


test_that("input / output lists and naming", {
  gr = Graph$new()$add_pipeop(PipeOpDebugMulti$new(2, 2))

  expect_equal(csvify(gr$input),
    c("debug.multi.input_1,*,*,debug.multi,input_1",
      "debug.multi.input_2,*,*,debug.multi,input_2"))
  expect_equal(csvify(gr$output),
    c("debug.multi.output_1,*,*,debug.multi,output_1",
      "debug.multi.output_2,*,*,debug.multi,output_2"))

  expect_output(gr$train(9), "list\\(input_1 = 9, input_2 = 9\\)")

  expect_output(gr$train(-9), "list\\(input_1 = -9, input_2 = -9\\)")

  expect_output(gr$train(list(1, 2), single_input = FALSE),
    "list\\(input_1 = 1, input_2 = 2\\)")

  expect_output(gr$train(list(debug.multi.input_1 = 1, debug.multi.input_2 = 2), single_input = FALSE),
    "list\\(input_1 = 1, input_2 = 2\\)")

  expect_output(gr$train(list(debug.multi.input_2 = 2, debug.multi.input_1 = 1), single_input = FALSE),
    "list\\(input_1 = 1, input_2 = 2\\)")


  expect_error(gr$train(list(debug.multi.input_2 = 2, debug.multi.input_99 = 1), single_input = FALSE),
    "debug.multi.input_1,debug.multi.input_2")

  expect_error(gr$train(list(), single_input = FALSE), "have length 2")
  expect_error(gr$train(list(1, 2, 3), single_input = FALSE), "have length 2")

  expect_equal(gr$train(1), list(debug.multi.output_1 = 2, debug.multi.output_2 = 3))

  gr$add_pipeop(PipeOpDebugMulti$new(2, 3, "debug2"))
  gr$add_pipeop(PipeOpDebugMulti$new(3, 2, "debug3"))

  gr$add_edge("debug.multi", "debug3", 1, 2)
  gr$add_edge("debug.multi", "debug3", "output_2", "input_1")

  gr$add_edge("debug2", "debug.multi", 2, 1)
  gr$add_edge("debug2", "debug.multi", "output_1", "input_2")

  # layout now:
  #
  # |     1\    2/
  # |     <debug2>
  # |     1/ 2| 3\
  # |     /   |   \
  # |   2|   1|    \
  # |   <multi>    |
  # |   1|   2|    |
  # \    |   /     |
  # 3\  2| 1/      |
  #   <debug3>     |
  #    1/   2\     |
  #
  # input should be debug2.1, debug2.2, debug3.3
  # output should be debug2.3, debug3.1, debug3.2
  # (inputs and outputs in PipeOp order first, in channel order second)

  pdf(file = NULL)  # don't show plot. It is annoying.
  gr$plot()
  dev.off()

  p = gr$plot(TRUE)
  expect_class(p, "htmlwidget")
  expect_class(p, "visNetwork")

  # test output 1: debug.multi was already trained above
  expect_output(print(gr),
    "debug2.*<<UNTRAINED>>.*debug.multi.*\n.*debug.multi.*<list>.*debug3.*debug2.*\n.*debug3.*<<UNTRAINED>>.*debug.multi")

  gr$pipeops$debug.multi$state = NULL
  expect_output(print(gr),
    "debug2.*<<UNTRAINED>>.*debug.multi.*\n.*debug.multi.*<<UNTRAINED>>.*debug3.*debug2.*\n.*debug3.*<<UNTRAINED>>.*debug.multi")

  expect_equal(csvify(gr$input),
    c("debug2.input_1,*,*,debug2,input_1",
      "debug2.input_2,*,*,debug2,input_2",
      "debug3.input_3,*,*,debug3,input_3"))

  expect_equal(csvify(gr$output),
    c("debug2.output_3,*,*,debug2,output_3",
      "debug3.output_1,*,*,debug3,output_1",
      "debug3.output_2,*,*,debug3,output_2"))

  lines = strsplit(capture_output({
    trained = gr$train(list(debug2.input_2 = 10, debug3.input_3 = 100, debug2.input_1 = 1000), single_input = FALSE)
  }),
  "\n")[[1]]

  expect_equal(lines,
    c("Training debug2 with input list(input_1 = 1000, input_2 = 10)",
      "Training debug.multi with input list(input_1 = 1002, input_2 = 1001)",
      "Training debug3 with input list(input_1 = 1004, input_2 = 1003, input_3 = 100)"))

  expect_equal(trained, list(debug2.output_3 = 1003, debug3.output_1 = 1005, debug3.output_2 = 1006))

  # test output II
  expect_output(print(gr),
    "debug2.*<list>.*debug.multi.*\n.*debug.multi.*<list>.*debug3.*debug2.*\n.*debug3.*<list>.*debug.multi")
})

test_that("edges that introduce loops cannot be added", {
  g = Graph$new()$
    add_pipeop(PipeOpNOP$new("p1"))$
    add_pipeop(PipeOpNOP$new("p2"))

  gclone = g$clone(deep = TRUE)
  expect_deep_clone(g, gclone)

  expect_error(g$add_edge("p1", "p1", 1, 1), "Cycle detected")

  expect_deep_clone(g, gclone)  # check that edges did not change

  g$add_edge("p1", "p2")

  gclone = g$clone(deep = TRUE)
  expect_deep_clone(g, gclone)

  expect_error(g$add_edge("p2", "p1", 1, 1), "Cycle detected")

  expect_deep_clone(g, gclone)  # check that edges did not change
})


test_that("assert_graph test", {
  gr = PipeOpNOP$new() %>>% PipeOpDebugMulti$new(1, 1)

  gr2 = assert_graph(gr)

  expect_error(expect_deep_clone(gr, gr2), class = "error", regexp = "addresses differ.*isn't true|addresses differ.*is not TRUE")

  gr2 = as_graph(gr, clone = TRUE)

  expect_deep_clone(gr, gr2)

  expect_error(assert_graph(PipeOpNOP$new()), "inherit from class.*Graph")

  assert_graph(as_graph(PipeOpNOP$new()))

  po = PipeOpNOP$new()

  expect_error(expect_deep_clone(po, po), class = "error", regexp = "addresses differ.*isn't true|addresses differ.*is not TRUE")

  po2 = as_graph(po, clone = TRUE)$pipeops[[1]]

  expect_deep_clone(po, po2)
})

test_that("Empty Graph", {
  expect_equal(length(Graph$new()$pipeops), 0)

  expect_equal(nrow(Graph$new()$edges), 0)

  expect_output(print(Graph$new()), "^Empty Graph\\.$")

  expect_output(Graph$new()$plot(), "^Empty Graph, not plotting\\.$")

  expect_equal(gunion(list()), Graph$new())

  expect_equal(gunion(list(Graph$new())), Graph$new())

  expect_equal(gunion(list(Graph$new(), Graph$new())), Graph$new())

  expect_equal(pipeline_greplicate(Graph$new(), 100), Graph$new())

  expect_error(Graph$new()$add_edge("a", "b"), "Cannot add edge to empty Graph")

  expect_equal(Graph$new()$state, list())

  expect_equal(Graph$new()$update_ids()$ids(), character(0))
})

test_that("Graph printer aux function calculates col widths well", {
  skip_on_cran()
  set.seed(8008135)

  effective_outwidth = function(colwidths, collimit) {
    colwidths[colwidths > collimit] = collimit + 3  # this is how data.table does it
    sum(colwidths + 1) + 4  # add 1 spacer between columns, and an extra margin of 4
  }

  test_outlimit = function(colwidths, outwidth) {
    collimit = calculate_collimit(colwidths, outwidth)
    if (effective_outwidth(colwidths, collimit) > outwidth) {
      # expectations are *really* slow, so only do them if they fail in this high volume test
      expect_lte(effective_outwidth(colwidths, collimit), outwidth)
    }
  }

  test_all_outlimits = function(colwidths) {
    for (outwidth in seq(max(length(colwidths), length(colwidths) * min(colwidths) - 2),
      length(colwidths) * max(colwidths) + 2)) {
      test_outlimit(colwidths, outwidth)
    }
  }

  for (numcols in 1:10) {
    replicate(1000, test_all_outlimits(rgeom(numcols, 0.1)))
  }

  expect_string("spurious expectation to show that we didn't skip this test")
})

test_that("Intermediate results are saved to Graph if requested", {
  g = PipeOpPCA$new() %>>% PipeOpCopy$new(2) %>>%
    gunion(list(PipeOpScale$new(), PipeOpNOP$new()))

  task = mlr_tasks$get("iris")

  res = g$train(task)

  expect_null(g$pipeops$pca$.result)

  g$keep_results = TRUE

  res = g$predict(task)

  restask = PipeOpPCA$new()$train(list(task))

  expect_equal(g$pipeops$pca$.result, restask)

  restask2 = PipeOpScale$new()$train(restask)

  expect_equal(g$pipeops$scale$.result, restask2)

  expect_equal(g$pipeops$nop$.result, restask)
})

test_that("Namespaces get loaded", {
  g = PipeOpPCA$new() %>>% PipeOpCopy$new(2) %>>%
    gunion(list(PipeOpScale$new(), PipeOpNOP$new()))

  g$train(mlr_tasks$get("iris"))

  g$pipeops$scale$packages = c("4rfjfw", "324r32")
  g$pipeops$nop$packages = c("4rfjfw", "9422228u")

  res = try(g$train(mlr_tasks$get("iris")), silent = TRUE)

  expect_class(res, "try-error")
  expect_subset(c(
    "  Error loading package 9422228u (required by nop):",
    "  Error loading package 4rfjfw (required by scale, nop):",
    "  Error loading package 324r32 (required by scale):"),
  strsplit(res, "\n")[[1]])
})

test_that("Graph State", {
  g = PipeOpPCA$new() %>>% PipeOpCopy$new(2) %>>%
    gunion(list(PipeOpScale$new(), PipeOpNOP$new()))

  g_clone = g$clone(deep = TRUE)
  task = mlr_tasks$get("iris")

  res = g$train(task)

  g_clone$state = g$state
  expect_deep_clone(g_clone, g$clone(deep = TRUE))

  expect_equal(g$predict(task), g_clone$predict(task))

})

test_that("Graph with vararg input", {
  t1 = tsk("iris")
  t2 = PipeOpPCA$new()$train(list(t1))[[1]]
  tcombined = PipeOpFeatureUnion$new()$train(list(t1, t2))[[1]]
  gr = as_graph(PipeOpFeatureUnion$new())
  expect_equal(tcombined, gr$train(list(t1, t2), single_input = FALSE)[[1]])
  expect_equal(tcombined, gr$predict(list(t1, t2), single_input = FALSE)[[1]])

  gr = gunion(list(PipeOpNOP$new(), gr, PipeOpNOP$new(id = "nop2"), PipeOpNOP$new(id = "nop3")))

  expect_equal(list(nop.output = 1, featureunion.output = tcombined, nop2.output = 2, nop3.output = 3),
               gr$train(list(1, t1, t2, 2, 3), single_input = FALSE))
})

test_that("single pipeop plot", {
  imp_num = po("imputehist")
  graph = as_graph(imp_num)

  pdf(file = NULL)
  graph$plot()
  dev.off()

  p = graph$plot(TRUE)
  expect_class(p, "htmlwidget")
  expect_class(p, "visNetwork")
})

test_that("dot output", {
  gr = Graph$new()$add_pipeop(PipeOpDebugMulti$new(2, 2))
  gr$add_pipeop(PipeOpDebugMulti$new(2, 3, "debug2"))
  gr$add_pipeop(PipeOpDebugMulti$new(3, 2, "debug3"))
  gr$add_edge("debug.multi", "debug3", 1, 2)
  gr$add_edge("debug.multi", "debug3", "output_2", "input_1")
  gr$add_edge("debug2", "debug.multi", 2, 1)
  gr$add_edge("debug2", "debug.multi", "output_1", "input_2")
  out = capture.output(gr$print(dot = TRUE, dotname = "test", fontsize = 20))
  expect_true(out[1L] == "digraph test {")
  expect_true(out[22L] == "}")
  expect_setequal(c("1 -> 4;", "1 -> 4;", "2 -> 1;", "2 -> 1;", "3 -> 2;", "3 -> 2;",
    "3 -> 4;", "2 -> 5;", "4 -> 6;", "4 -> 7;",
    "1 [label=\"debug_multi\",fontsize=20];",
    "2 [label=\"debug2\",fontsize=20];",
    "3 [label=\"INPUT\",fontsize=20];",
    "4 [label=\"debug3\",fontsize=20];",
    "5 [label=\"OUTPUT",
    "debug2_output_3\",fontsize=20];",
    "6 [label=\"OUTPUT",
    "debug3_output_1\",fontsize=20];",
    "7 [label=\"OUTPUT",
    "debug3_output_2\",fontsize=20]"), out[-c(1L, 22L)])

  gr = Graph$new()
  out = capture.output(print(gr, dot = TRUE))
  expect_setequal(c("digraph dot {", "", "}"), out)

  gr = Graph$new()$add_pipeop(po("scale"))
  out = capture.output(gr$print(dot = TRUE))
  expect_setequal(c("digraph dot {", "1 [label=\"scale\",fontsize=24]", "}"), out)

  gr = po("scale") %>>% po("pca")
  gr$add_pipeop(po("nop"))
  out = capture.output(gr$print(dot = TRUE))
  expect_true(out[1L] == "digraph dot {")
  expect_true(out[15L] == "}")
  expect_setequal(c("1 -> 3;", "2 -> 1;", "2 -> 4;", "3 -> 5;", "4 -> 6;",
    "1 [label=\"scale\",fontsize=24];",
    "2 [label=\"INPUT\",fontsize=24];",
    "3 [label=\"pca\",fontsize=24];",
    "4 [label=\"nop\",fontsize=24];",
    "5 [label=\"OUTPUT",
    "pca_output\",fontsize=24];",
    "6 [label=\"OUTPUT",
    "nop_output\",fontsize=24]"), out[-c(1L, 15L)])
})



test_that("replace_subgraph", {
  task = tsk("iris")

  # Basics
  gr = Graph$new()$add_pipeop(PipeOpDebugMulti$new(2, 2))
  address_old = address(gr)
  gr_old = gr$clone(deep = TRUE)
  expect_error(gr$replace_subgraph("id_not_present", PipeOpDebugMulti$new(2, 2)),
    regexp = "Assertion on 'ids' failed")
  expect_error(gr$replace_subgraph("debug.multi", NULL),
    regexp = "op can not be converted to PipeOp")
  expect_equal(gr, gr_old)  # error results in a clean reset
  expect_true(address_old == address(gr))
  expect_deep_clone(gr_old, gr)

  gr$replace_subgraph("debug.multi", substitute = PipeOpDebugMulti$new(2, 2))
  expect_equal(gr_old, gr)
  expect_true(address_old == address(gr))  # in place modification
  expect_deep_clone(gr_old, gr)  # replacing with exactly the same pipeop is the same as a deep clone

  # Linear Graph
  gr = po("scale") %>>% po("pca") %>>% lrn("classif.rpart")
  gr_old = gr$clone(deep = TRUE)
  gr$replace_subgraph("scale", substitute = po("scalemaxabs"))  # replace beginning
  expect_set_equal(gr$ids(), c("scalemaxabs", "pca", "classif.rpart"))
  expect_true(gr$input$op.id == "scalemaxabs")
  expect_true(gr$output$op.id == "classif.rpart")
  expect_null(gr$train(task)[[1L]])
  expect_prediction_classif(gr$predict(task)[[1L]])

  gr = gr_old$clone(deep = TRUE)
  gr$replace_subgraph("classif.rpart", substitute = lrn("classif.featureless"))  # replace end
  expect_set_equal(gr$ids(), c("scale", "pca", "classif.featureless"))
  expect_true(gr$input$op.id == "scale")
  expect_true(gr$output$op.id == "classif.featureless")
  expect_null(gr$train(task)[[1L]])
  expect_prediction_classif(gr$predict(task)[[1L]])

  gr = gr_old$clone(deep = TRUE)
  gr$replace_subgraph(c("scale", "pca", "classif.rpart"), substitute = po("scalemaxabs") %>>% po("ica") %>>% lrn("classif.featureless"))  # replace whole graph
  expect_set_equal(gr$ids(), c("scalemaxabs", "ica", "classif.featureless"))
  expect_true(gr$input$op.id == "scalemaxabs")
  expect_true(gr$output$op.id == "classif.featureless")
  expect_null(gr$train(task)[[1L]])
  expect_prediction_classif(gr$predict(task)[[1L]])

  gr = gr_old$clone(deep = TRUE)
  gr$replace_subgraph(c("pca", "scale"), substitute = po("scalemaxabs") %>>% po("ica"))  # replace linear subgraph
  expect_set_equal(gr$ids(), c("scalemaxabs", "ica", "classif.rpart"))
  expect_true(gr$input$op.id == "scalemaxabs")
  expect_true(gr$output$op.id == "classif.rpart")
  expect_null(gr$train(task)[[1L]])
  expect_prediction_classif(gr$predict(task)[[1L]])

  # Non linear Graph
  gr = po("scale") %>>% po("branch", c("pca", "nop")) %>>% gunion(list(po("pca"), po("nop"))) %>>% po("unbranch") %>>% lrn("classif.rpart")
  gr_old = gr$clone(deep = TRUE)
  #expect_error(gr$replace_subgraph(c("nop"), substitute = po("ica")), regexp = "connected to a vararg channel is not supported")  # FIXME:
  expect_error(gr$replace_subgraph(c("branch", "pca", "nop", "unbranch"), substitute = lrn("classif.featureless")),
    regexp = "Output type of PipeOp classif.featureless during training")
  gr$replace_subgraph(c("branch", "pca", "nop", "unbranch"), substitute = po("branch", c("pca", "ica")) %>>% gunion(list(po("pca"), po("ica"))) %>>% po("unbranch"))
  expect_set_equal(gr$ids(TRUE), c("scale", "branch", "pca", "ica", "unbranch", "classif.rpart"))
  expect_true(gr$input$op.id == "scale")
  expect_true(gr$output$op.id == "classif.rpart")
  expect_null(gr$train(task)[[1L]])
  state1 = gr$state
  gr$param_set$values$branch.selection = "ica"
  expect_null(gr$train(task)[[1L]])
  state2 = gr$state
  expect_true(test_r6(state1$ica, classes = "NO_OP"))
  expect_true(test_r6(state2$pca, classes = "NO_OP"))
  expect_prediction_classif(gr$predict(task)[[1L]])
})
