module @jit_step attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["data"=8]> {stablehlo.mesh = {axes = [{name = "data", size = 8 : i64}]}}
  func.func public @main(%arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg2: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg3: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg4: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg5: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg6: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg7: tensor<128x384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg8: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg9: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg10: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg11: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg12: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg13: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg14: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg15: tensor<128x384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg16: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg17: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg18: tensor<10x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg19: tensor<15x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg20: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg21: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg22: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg23: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg24: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg25: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg26: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg27: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg28: tensor<128x384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg29: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg30: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg31: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg32: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg33: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg34: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg35: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg36: tensor<128x384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg37: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg38: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg39: tensor<10x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg40: tensor<15x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg41: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg42: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg43: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg44: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg45: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg46: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg47: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg48: tensor<128x384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg49: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg50: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg51: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg52: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg53: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg54: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg55: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg56: tensor<128x384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg57: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg58: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg59: tensor<10x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg60: tensor<15x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg61: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg62: tensor<128x10xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}, %arg63: tensor<128x9xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}) -> (tensor<128x512xf32> {jax.result_info = "result[0]['blocks'][0]['fc']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][0]['ln1']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][0]['ln1']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][0]['ln2']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][0]['ln2']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<512x128xf32> {jax.result_info = "result[0]['blocks'][0]['out']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x128xf32> {jax.result_info = "result[0]['blocks'][0]['proj']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x384xf32> {jax.result_info = "result[0]['blocks'][0]['qkv']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x512xf32> {jax.result_info = "result[0]['blocks'][1]['fc']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][1]['ln1']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][1]['ln1']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][1]['ln2']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['blocks'][1]['ln2']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<512x128xf32> {jax.result_info = "result[0]['blocks'][1]['out']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x128xf32> {jax.result_info = "result[0]['blocks'][1]['proj']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x384xf32> {jax.result_info = "result[0]['blocks'][1]['qkv']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[0]['ln_f']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[0]['ln_f']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<10x128xf32> {jax.result_info = "result[0]['pos']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<15x128xf32> {jax.result_info = "result[0]['tok']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<i32> {jax.result_info = "result[1][1][0].count", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<128x512xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['fc']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['ln1']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['ln1']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['ln2']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['ln2']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<512x128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['out']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['proj']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x384xf32> {jax.result_info = "result[1][1][0].mu['blocks'][0]['qkv']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x512xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['fc']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['ln1']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['ln1']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['ln2']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['ln2']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<512x128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['out']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x128xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['proj']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x384xf32> {jax.result_info = "result[1][1][0].mu['blocks'][1]['qkv']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['ln_f']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].mu['ln_f']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<10x128xf32> {jax.result_info = "result[1][1][0].mu['pos']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<15x128xf32> {jax.result_info = "result[1][1][0].mu['tok']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x512xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['fc']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['ln1']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['ln1']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['ln2']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['ln2']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<512x128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['out']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['proj']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x384xf32> {jax.result_info = "result[1][1][0].nu['blocks'][0]['qkv']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x512xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['fc']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['ln1']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['ln1']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['ln2']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['ln2']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<512x128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['out']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x128xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['proj']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x384xf32> {jax.result_info = "result[1][1][0].nu['blocks'][1]['qkv']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['ln_f']['b']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<128xf32> {jax.result_info = "result[1][1][0].nu['ln_f']['g']", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<10x128xf32> {jax.result_info = "result[1][1][0].nu['pos']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<15x128xf32> {jax.result_info = "result[1][1][0].nu['tok']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<i32> {jax.result_info = "result[1][1][2].count", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[2]", sdy.sharding = #sdy.sharding<@mesh, []>}) {
    %0 = stablehlo.slice %arg62 [0:128, 0:9] : (tensor<128x10xi32>) -> tensor<128x9xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x9xi32>
    %2 = stablehlo.compare LT, %0, %1, SIGNED : (tensor<128x9xi32>, tensor<128x9xi32>) -> tensor<128x9xi1>
    %c_0 = stablehlo.constant dense<15> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128x9xi32>
    %4 = stablehlo.add %0, %3 : tensor<128x9xi32>
    %5 = stablehlo.select %2, %4, %0 : tensor<128x9xi1>, tensor<128x9xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<128x9xi32>) -> tensor<128x9x1xi32>
    %7 = "stablehlo.gather"(%arg19, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<15x128xf32>, tensor<128x9x1xi32>) -> tensor<128x9x128xf32>
    %8 = stablehlo.slice %arg18 [0:9, 0:128] : (tensor<10x128xf32>) -> tensor<9x128xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [1, 2] : (tensor<9x128xf32>) -> tensor<1x9x128xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x9x128xf32>) -> tensor<128x9x128xf32>
    %11 = stablehlo.add %7, %10 : tensor<128x9x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_1 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %15 = stablehlo.divide %13, %14 : tensor<128x9x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %17 = stablehlo.subtract %11, %16 : tensor<128x9x128xf32>
    %18 = stablehlo.multiply %17, %17 : tensor<128x9x128xf32>
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x9x128xf32>
    %20 = stablehlo.multiply %19, %17 : tensor<128x9x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%18 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_4 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %24 = stablehlo.divide %22, %23 : tensor<128x9x1xf32>
    %25 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %26 = stablehlo.subtract %11, %25 : tensor<128x9x128xf32>
    %cst_5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %28 = stablehlo.add %24, %27 : tensor<128x9x1xf32>
    %29 = stablehlo.sqrt %28 : tensor<128x9x1xf32>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %31 = stablehlo.divide %30, %29 : tensor<128x9x1xf32>
    %32 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %33 = stablehlo.divide %26, %32 : tensor<128x9x128xf32>
    %34 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %36 = stablehlo.multiply %33, %35 : tensor<128x9x128xf32>
    %37 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %39 = stablehlo.add %36, %38 : tensor<128x9x128xf32>
    %40 = stablehlo.dot_general %39, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x384xf32>) -> tensor<128x9x384xf32>
    %41 = stablehlo.slice %40 [0:128, 0:9, 0:128] : (tensor<128x9x384xf32>) -> tensor<128x9x128xf32>
    %42 = stablehlo.slice %40 [0:128, 0:9, 128:256] : (tensor<128x9x384xf32>) -> tensor<128x9x128xf32>
    %43 = stablehlo.slice %40 [0:128, 0:9, 256:384] : (tensor<128x9x384xf32>) -> tensor<128x9x128xf32>
    %44 = stablehlo.reshape %41 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %45 = stablehlo.transpose %44, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %46 = stablehlo.reshape %42 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %47 = stablehlo.transpose %46, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %48 = stablehlo.reshape %43 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %49 = stablehlo.transpose %48, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %50 = stablehlo.transpose %47, dims = [0, 1, 3, 2] : (tensor<128x4x9x32xf32>) -> tensor<128x4x32x9xf32>
    %51 = stablehlo.dot_general %45, %50, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x32xf32>, tensor<128x4x32x9xf32>) -> tensor<128x4x9x9xf32>
    %cst_7 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %52 = stablehlo.sqrt %cst_7 : tensor<f32>
    %53 = stablehlo.convert %52 : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<f32>) -> tensor<128x4x9x9xf32>
    %55 = stablehlo.divide %51, %54 : tensor<128x4x9x9xf32>
    %c_8 = stablehlo.constant dense<true> : tensor<i1>
    %56 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i1>) -> tensor<9x9xi1>
    %57 = call @tril(%56) : (tensor<9x9xi1>) -> tensor<9x9xi1>
    %cst_9 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %58:2 = call @_where(%57, %55, %cst_9) : (tensor<9x9xi1>, tensor<128x4x9x9xf32>, tensor<f32>) -> (tensor<128x4x9x9xf32>, tensor<128x4x9x9xi1>)
    %cst_10 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %59 = stablehlo.reduce(%58#0 init: %cst_10) applies stablehlo.maximum across dimensions = [3] : (tensor<128x4x9x9xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %cst_11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x4x9xf32>
    %61 = stablehlo.maximum %60, %59 : tensor<128x4x9xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2] : (tensor<128x4x9xf32>) -> tensor<128x4x9x1xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<128x4x9x1xf32>) -> tensor<128x4x9x9xf32>
    %64 = stablehlo.subtract %58#0, %63 : tensor<128x4x9x9xf32>
    %65 = stablehlo.exponential %64 : tensor<128x4x9x9xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %66 = stablehlo.reduce(%65 init: %cst_12) applies stablehlo.add across dimensions = [3] : (tensor<128x4x9x9xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<128x4x9xf32>) -> tensor<128x4x9x1xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<128x4x9x1xf32>) -> tensor<128x4x9x9xf32>
    %69 = stablehlo.divide %65, %68 : tensor<128x4x9x9xf32>
    %70 = stablehlo.dot_general %69, %49, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x9xf32>, tensor<128x4x9x32xf32>) -> tensor<128x4x9x32xf32>
    %71 = stablehlo.transpose %70, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %72 = stablehlo.reshape %71 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %73 = stablehlo.dot_general %72, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x128xf32>) -> tensor<128x9x128xf32>
    %74 = stablehlo.add %11, %73 : tensor<128x9x128xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %75 = stablehlo.reduce(%74 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_14 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %77 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %78 = stablehlo.divide %76, %77 : tensor<128x9x1xf32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %80 = stablehlo.subtract %74, %79 : tensor<128x9x128xf32>
    %81 = stablehlo.multiply %80, %80 : tensor<128x9x128xf32>
    %cst_15 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %82 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x9x128xf32>
    %83 = stablehlo.multiply %82, %80 : tensor<128x9x128xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %84 = stablehlo.reduce(%81 init: %cst_16) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_17 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %87 = stablehlo.divide %85, %86 : tensor<128x9x1xf32>
    %88 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %89 = stablehlo.subtract %74, %88 : tensor<128x9x128xf32>
    %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %91 = stablehlo.add %87, %90 : tensor<128x9x1xf32>
    %92 = stablehlo.sqrt %91 : tensor<128x9x1xf32>
    %cst_19 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %94 = stablehlo.divide %93, %92 : tensor<128x9x1xf32>
    %95 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %96 = stablehlo.divide %89, %95 : tensor<128x9x128xf32>
    %97 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %99 = stablehlo.multiply %96, %98 : tensor<128x9x128xf32>
    %100 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %102 = stablehlo.add %99, %101 : tensor<128x9x128xf32>
    %103 = stablehlo.dot_general %102, %arg0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x512xf32>) -> tensor<128x9x512xf32>
    %104 = stablehlo.multiply %103, %103 : tensor<128x9x512xf32>
    %105 = stablehlo.multiply %104, %103 : tensor<128x9x512xf32>
    %106 = stablehlo.multiply %103, %103 : tensor<128x9x512xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %108 = stablehlo.multiply %107, %106 : tensor<128x9x512xf32>
    %cst_21 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %109 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %110 = stablehlo.multiply %109, %105 : tensor<128x9x512xf32>
    %111 = stablehlo.add %103, %110 : tensor<128x9x512xf32>
    %cst_22 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %112 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %113 = stablehlo.multiply %112, %111 : tensor<128x9x512xf32>
    %114 = stablehlo.tanh %113 : tensor<128x9x512xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %115 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %116 = stablehlo.add %115, %114 : tensor<128x9x512xf32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %118 = stablehlo.subtract %117, %114 : tensor<128x9x512xf32>
    %119 = stablehlo.multiply %116, %118 : tensor<128x9x512xf32>
    %cst_25 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %120 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %121 = stablehlo.add %120, %114 : tensor<128x9x512xf32>
    %cst_26 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %122 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %123 = stablehlo.multiply %122, %121 : tensor<128x9x512xf32>
    %124 = stablehlo.multiply %103, %123 : tensor<128x9x512xf32>
    %125 = stablehlo.dot_general %124, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x512xf32>, tensor<512x128xf32>) -> tensor<128x9x128xf32>
    %126 = stablehlo.add %74, %125 : tensor<128x9x128xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %127 = stablehlo.reduce(%126 init: %cst_27) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_28 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %129 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %130 = stablehlo.divide %128, %129 : tensor<128x9x1xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %132 = stablehlo.subtract %126, %131 : tensor<128x9x128xf32>
    %133 = stablehlo.multiply %132, %132 : tensor<128x9x128xf32>
    %cst_29 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %134 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<128x9x128xf32>
    %135 = stablehlo.multiply %134, %132 : tensor<128x9x128xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %136 = stablehlo.reduce(%133 init: %cst_30) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %137 = stablehlo.broadcast_in_dim %136, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_31 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %138 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %139 = stablehlo.divide %137, %138 : tensor<128x9x1xf32>
    %140 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %141 = stablehlo.subtract %126, %140 : tensor<128x9x128xf32>
    %cst_32 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %142 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %143 = stablehlo.add %139, %142 : tensor<128x9x1xf32>
    %144 = stablehlo.sqrt %143 : tensor<128x9x1xf32>
    %cst_33 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %145 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %146 = stablehlo.divide %145, %144 : tensor<128x9x1xf32>
    %147 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %148 = stablehlo.divide %141, %147 : tensor<128x9x128xf32>
    %149 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %151 = stablehlo.multiply %148, %150 : tensor<128x9x128xf32>
    %152 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %154 = stablehlo.add %151, %153 : tensor<128x9x128xf32>
    %155 = stablehlo.dot_general %154, %arg15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x384xf32>) -> tensor<128x9x384xf32>
    %156 = stablehlo.slice %155 [0:128, 0:9, 0:128] : (tensor<128x9x384xf32>) -> tensor<128x9x128xf32>
    %157 = stablehlo.slice %155 [0:128, 0:9, 128:256] : (tensor<128x9x384xf32>) -> tensor<128x9x128xf32>
    %158 = stablehlo.slice %155 [0:128, 0:9, 256:384] : (tensor<128x9x384xf32>) -> tensor<128x9x128xf32>
    %159 = stablehlo.reshape %156 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %160 = stablehlo.transpose %159, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %161 = stablehlo.reshape %157 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %162 = stablehlo.transpose %161, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %163 = stablehlo.reshape %158 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %164 = stablehlo.transpose %163, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %165 = stablehlo.transpose %162, dims = [0, 1, 3, 2] : (tensor<128x4x9x32xf32>) -> tensor<128x4x32x9xf32>
    %166 = stablehlo.dot_general %160, %165, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x32xf32>, tensor<128x4x32x9xf32>) -> tensor<128x4x9x9xf32>
    %cst_34 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %167 = stablehlo.sqrt %cst_34 : tensor<f32>
    %168 = stablehlo.convert %167 : tensor<f32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f32>) -> tensor<128x4x9x9xf32>
    %170 = stablehlo.divide %166, %169 : tensor<128x4x9x9xf32>
    %c_35 = stablehlo.constant dense<true> : tensor<i1>
    %171 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i1>) -> tensor<9x9xi1>
    %172 = call @tril(%171) : (tensor<9x9xi1>) -> tensor<9x9xi1>
    %cst_36 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %173:2 = call @_where(%172, %170, %cst_36) : (tensor<9x9xi1>, tensor<128x4x9x9xf32>, tensor<f32>) -> (tensor<128x4x9x9xf32>, tensor<128x4x9x9xi1>)
    %cst_37 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %174 = stablehlo.reduce(%173#0 init: %cst_37) applies stablehlo.maximum across dimensions = [3] : (tensor<128x4x9x9xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %cst_38 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %175 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<128x4x9xf32>
    %176 = stablehlo.maximum %175, %174 : tensor<128x4x9xf32>
    %177 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2] : (tensor<128x4x9xf32>) -> tensor<128x4x9x1xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<128x4x9x1xf32>) -> tensor<128x4x9x9xf32>
    %179 = stablehlo.subtract %173#0, %178 : tensor<128x4x9x9xf32>
    %180 = stablehlo.exponential %179 : tensor<128x4x9x9xf32>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %181 = stablehlo.reduce(%180 init: %cst_39) applies stablehlo.add across dimensions = [3] : (tensor<128x4x9x9xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<128x4x9xf32>) -> tensor<128x4x9x1xf32>
    %183 = stablehlo.broadcast_in_dim %182, dims = [0, 1, 2, 3] : (tensor<128x4x9x1xf32>) -> tensor<128x4x9x9xf32>
    %184 = stablehlo.divide %180, %183 : tensor<128x4x9x9xf32>
    %185 = stablehlo.dot_general %184, %164, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x9xf32>, tensor<128x4x9x32xf32>) -> tensor<128x4x9x32xf32>
    %186 = stablehlo.transpose %185, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %187 = stablehlo.reshape %186 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %188 = stablehlo.dot_general %187, %arg14, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x128xf32>) -> tensor<128x9x128xf32>
    %189 = stablehlo.add %126, %188 : tensor<128x9x128xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.reduce(%189 init: %cst_40) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_41 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %193 = stablehlo.divide %191, %192 : tensor<128x9x1xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %195 = stablehlo.subtract %189, %194 : tensor<128x9x128xf32>
    %196 = stablehlo.multiply %195, %195 : tensor<128x9x128xf32>
    %cst_42 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %197 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<128x9x128xf32>
    %198 = stablehlo.multiply %197, %195 : tensor<128x9x128xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %199 = stablehlo.reduce(%196 init: %cst_43) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_44 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %201 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %202 = stablehlo.divide %200, %201 : tensor<128x9x1xf32>
    %203 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %204 = stablehlo.subtract %189, %203 : tensor<128x9x128xf32>
    %cst_45 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %205 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %206 = stablehlo.add %202, %205 : tensor<128x9x1xf32>
    %207 = stablehlo.sqrt %206 : tensor<128x9x1xf32>
    %cst_46 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %208 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %209 = stablehlo.divide %208, %207 : tensor<128x9x1xf32>
    %210 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %211 = stablehlo.divide %204, %210 : tensor<128x9x128xf32>
    %212 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %214 = stablehlo.multiply %211, %213 : tensor<128x9x128xf32>
    %215 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %217 = stablehlo.add %214, %216 : tensor<128x9x128xf32>
    %218 = stablehlo.dot_general %217, %arg8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x512xf32>) -> tensor<128x9x512xf32>
    %219 = stablehlo.multiply %218, %218 : tensor<128x9x512xf32>
    %220 = stablehlo.multiply %219, %218 : tensor<128x9x512xf32>
    %221 = stablehlo.multiply %218, %218 : tensor<128x9x512xf32>
    %cst_47 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %222 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %223 = stablehlo.multiply %222, %221 : tensor<128x9x512xf32>
    %cst_48 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %224 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %225 = stablehlo.multiply %224, %220 : tensor<128x9x512xf32>
    %226 = stablehlo.add %218, %225 : tensor<128x9x512xf32>
    %cst_49 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %227 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %228 = stablehlo.multiply %227, %226 : tensor<128x9x512xf32>
    %229 = stablehlo.tanh %228 : tensor<128x9x512xf32>
    %cst_50 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %230 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %231 = stablehlo.add %230, %229 : tensor<128x9x512xf32>
    %cst_51 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %232 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %233 = stablehlo.subtract %232, %229 : tensor<128x9x512xf32>
    %234 = stablehlo.multiply %231, %233 : tensor<128x9x512xf32>
    %cst_52 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %236 = stablehlo.add %235, %229 : tensor<128x9x512xf32>
    %cst_53 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %237 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %238 = stablehlo.multiply %237, %236 : tensor<128x9x512xf32>
    %239 = stablehlo.multiply %218, %238 : tensor<128x9x512xf32>
    %240 = stablehlo.dot_general %239, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x512xf32>, tensor<512x128xf32>) -> tensor<128x9x128xf32>
    %241 = stablehlo.add %189, %240 : tensor<128x9x128xf32>
    %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %242 = stablehlo.reduce(%241 init: %cst_54) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_55 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %244 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %245 = stablehlo.divide %243, %244 : tensor<128x9x1xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %247 = stablehlo.subtract %241, %246 : tensor<128x9x128xf32>
    %248 = stablehlo.multiply %247, %247 : tensor<128x9x128xf32>
    %cst_56 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %249 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<128x9x128xf32>
    %250 = stablehlo.multiply %249, %247 : tensor<128x9x128xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %251 = stablehlo.reduce(%248 init: %cst_57) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_58 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %253 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %254 = stablehlo.divide %252, %253 : tensor<128x9x1xf32>
    %255 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %256 = stablehlo.subtract %241, %255 : tensor<128x9x128xf32>
    %cst_59 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %257 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %258 = stablehlo.add %254, %257 : tensor<128x9x1xf32>
    %259 = stablehlo.sqrt %258 : tensor<128x9x1xf32>
    %cst_60 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %260 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %261 = stablehlo.divide %260, %259 : tensor<128x9x1xf32>
    %262 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %263 = stablehlo.divide %256, %262 : tensor<128x9x128xf32>
    %264 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %266 = stablehlo.multiply %263, %265 : tensor<128x9x128xf32>
    %267 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %269 = stablehlo.add %266, %268 : tensor<128x9x128xf32>
    %270 = stablehlo.transpose %arg19, dims = [1, 0] : (tensor<15x128xf32>) -> tensor<128x15xf32>
    %271 = stablehlo.dot_general %269, %270, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x15xf32>) -> tensor<128x9x15xf32>
    %272:3 = call @log_softmax(%271) : (tensor<128x9x15xf32>) -> (tensor<128x9x15xf32>, tensor<128x9x15xf32>, tensor<128x9x1xf32>)
    %273 = stablehlo.slice %arg62 [0:128, 1:10] : (tensor<128x10xi32>) -> tensor<128x9xi32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1] : (tensor<128x9xi32>) -> tensor<128x9x1xi32>
    %275:2 = call @take_along_axis(%272#0, %274) : (tensor<128x9x15xf32>, tensor<128x9x1xi32>) -> (tensor<128x9x1xf32>, tensor<128x9x1x1xi32>)
    %276 = stablehlo.reshape %275#0 : (tensor<128x9x1xf32>) -> tensor<128x9xf32>
    %277 = stablehlo.negate %276 : tensor<128x9xf32>
    %278 = stablehlo.multiply %277, %arg63 : tensor<128x9xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %279 = stablehlo.reduce(%278 init: %cst_61) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9xf32>, tensor<f32>) -> tensor<f32>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %280 = stablehlo.reduce(%arg63 init: %cst_62) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9xf32>, tensor<f32>) -> tensor<f32>
    %cst_63 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %281 = stablehlo.maximum %280, %cst_63 : tensor<f32>
    %282 = stablehlo.divide %279, %281 : tensor<f32>
    %cst_64 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %283 = stablehlo.divide %cst_64, %281 : tensor<f32>
    %284 = stablehlo.broadcast_in_dim %283, dims = [] : (tensor<f32>) -> tensor<128x9xf32>
    %285 = stablehlo.multiply %284, %arg63 : tensor<128x9xf32>
    %286 = stablehlo.negate %285 : tensor<128x9xf32>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %288 = call @take_along_axis_0(%275#1, %287) : (tensor<128x9x1x1xi32>, tensor<128x9x1xf32>) -> tensor<128x9x15xf32>
    %289 = call @log_softmax_1(%272#1, %272#2, %288) : (tensor<128x9x15xf32>, tensor<128x9x1xf32>, tensor<128x9x15xf32>) -> tensor<128x9x15xf32>
    %290 = stablehlo.dot_general %289, %269, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x15xf32>, tensor<128x9x128xf32>) -> tensor<15x128xf32>
    %291 = stablehlo.transpose %290, dims = [1, 0] : (tensor<15x128xf32>) -> tensor<128x15xf32>
    %292 = stablehlo.dot_general %289, %270, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x15xf32>, tensor<128x15xf32>) -> tensor<128x9x128xf32>
    %293 = stablehlo.transpose %291, dims = [1, 0] : (tensor<128x15xf32>) -> tensor<15x128xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %294 = stablehlo.reduce(%292 init: %cst_65) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %295 = stablehlo.reshape %294 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %296 = stablehlo.reduce(%295 init: %cst_66) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %297 = stablehlo.multiply %263, %292 : tensor<128x9x128xf32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %298 = stablehlo.reduce(%297 init: %cst_67) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %299 = stablehlo.reshape %298 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %300 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %301 = stablehlo.multiply %292, %300 : tensor<128x9x128xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %302 = stablehlo.reduce(%299 init: %cst_68) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %303 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %304 = stablehlo.divide %301, %303 : tensor<128x9x128xf32>
    %305 = stablehlo.negate %304 : tensor<128x9x128xf32>
    %306 = stablehlo.multiply %263, %305 : tensor<128x9x128xf32>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %307 = stablehlo.reduce(%306 init: %cst_69) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %308 = stablehlo.reshape %307 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %309 = stablehlo.multiply %308, %261 : tensor<128x9x1xf32>
    %310 = stablehlo.negate %304 : tensor<128x9x128xf32>
    %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %311 = stablehlo.reduce(%310 init: %cst_70) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %312 = stablehlo.reshape %311 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %cst_71 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %313 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %314 = stablehlo.divide %309, %313 : tensor<128x9x1xf32>
    %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %315 = stablehlo.reduce(%314 init: %cst_72) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %317 = stablehlo.multiply %316, %250 : tensor<128x9x128xf32>
    %318 = stablehlo.negate %317 : tensor<128x9x128xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %319 = stablehlo.reduce(%318 init: %cst_73) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %320 = stablehlo.reshape %319 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %321 = stablehlo.add %304, %317 : tensor<128x9x128xf32>
    %322 = stablehlo.add %312, %320 : tensor<128x9x1xf32>
    %cst_74 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %323 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %324 = stablehlo.divide %322, %323 : tensor<128x9x1xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %325 = stablehlo.reduce(%324 init: %cst_75) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %327 = stablehlo.add %321, %326 : tensor<128x9x128xf32>
    %328 = stablehlo.dot_general %327, %239, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x9x512xf32>) -> tensor<128x512xf32>
    %329 = stablehlo.transpose %328, dims = [1, 0] : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %330 = stablehlo.dot_general %327, %arg13, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<512x128xf32>) -> tensor<128x9x512xf32>
    %331 = stablehlo.multiply %218, %330 : tensor<128x9x512xf32>
    %332 = stablehlo.multiply %330, %238 : tensor<128x9x512xf32>
    %cst_76 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %333 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %334 = stablehlo.multiply %333, %331 : tensor<128x9x512xf32>
    %335 = stablehlo.multiply %334, %234 : tensor<128x9x512xf32>
    %cst_77 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %336 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %337 = stablehlo.multiply %336, %335 : tensor<128x9x512xf32>
    %338 = stablehlo.add %332, %337 : tensor<128x9x512xf32>
    %cst_78 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %339 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %340 = stablehlo.multiply %339, %337 : tensor<128x9x512xf32>
    %341 = stablehlo.multiply %340, %223 : tensor<128x9x512xf32>
    %342 = stablehlo.add %338, %341 : tensor<128x9x512xf32>
    %343 = stablehlo.dot_general %342, %217, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x512xf32>, tensor<128x9x128xf32>) -> tensor<512x128xf32>
    %344 = stablehlo.transpose %343, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %345 = stablehlo.dot_general %342, %arg8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x512xf32>, tensor<128x512xf32>) -> tensor<128x9x128xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %346 = stablehlo.reduce(%345 init: %cst_79) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %347 = stablehlo.reshape %346 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %348 = stablehlo.reduce(%347 init: %cst_80) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %349 = stablehlo.multiply %211, %345 : tensor<128x9x128xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %350 = stablehlo.reduce(%349 init: %cst_81) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %351 = stablehlo.reshape %350 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %352 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %353 = stablehlo.multiply %345, %352 : tensor<128x9x128xf32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %354 = stablehlo.reduce(%351 init: %cst_82) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %355 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %356 = stablehlo.divide %353, %355 : tensor<128x9x128xf32>
    %357 = stablehlo.negate %356 : tensor<128x9x128xf32>
    %358 = stablehlo.multiply %211, %357 : tensor<128x9x128xf32>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %359 = stablehlo.reduce(%358 init: %cst_83) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %360 = stablehlo.reshape %359 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %361 = stablehlo.multiply %360, %209 : tensor<128x9x1xf32>
    %362 = stablehlo.negate %356 : tensor<128x9x128xf32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %363 = stablehlo.reduce(%362 init: %cst_84) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %364 = stablehlo.reshape %363 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %365 = stablehlo.add %327, %356 : tensor<128x9x128xf32>
    %cst_85 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %366 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %367 = stablehlo.divide %361, %366 : tensor<128x9x1xf32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %368 = stablehlo.reduce(%367 init: %cst_86) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %370 = stablehlo.multiply %369, %198 : tensor<128x9x128xf32>
    %371 = stablehlo.negate %370 : tensor<128x9x128xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %372 = stablehlo.reduce(%371 init: %cst_87) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %373 = stablehlo.reshape %372 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %374 = stablehlo.add %365, %370 : tensor<128x9x128xf32>
    %375 = stablehlo.add %364, %373 : tensor<128x9x1xf32>
    %cst_88 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %376 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %377 = stablehlo.divide %375, %376 : tensor<128x9x1xf32>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %378 = stablehlo.reduce(%377 init: %cst_89) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %379 = stablehlo.broadcast_in_dim %378, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %380 = stablehlo.add %374, %379 : tensor<128x9x128xf32>
    %381 = stablehlo.dot_general %380, %187, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x9x128xf32>) -> tensor<128x128xf32>
    %382 = stablehlo.transpose %381, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %383 = stablehlo.dot_general %380, %arg14, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x128xf32>) -> tensor<128x9x128xf32>
    %384 = stablehlo.reshape %383 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %385 = stablehlo.transpose %384, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %386 = stablehlo.dot_general %385, %184, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x32xf32>, tensor<128x4x9x9xf32>) -> tensor<128x4x32x9xf32>
    %387 = stablehlo.transpose %386, dims = [0, 1, 3, 2] : (tensor<128x4x32x9xf32>) -> tensor<128x4x9x32xf32>
    %388 = stablehlo.dot_general %385, %164, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x32xf32>, tensor<128x4x9x32xf32>) -> tensor<128x4x9x9xf32>
    %389 = stablehlo.multiply %184, %388 : tensor<128x4x9x9xf32>
    %390 = stablehlo.negate %389 : tensor<128x4x9x9xf32>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %391 = stablehlo.reduce(%390 init: %cst_90) applies stablehlo.add across dimensions = [3] : (tensor<128x4x9x9xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %392 = stablehlo.reshape %391 : (tensor<128x4x9xf32>) -> tensor<128x4x9x1xf32>
    %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %393 = stablehlo.reduce(%392 init: %cst_91) applies stablehlo.add across dimensions = [3] : (tensor<128x4x9x1xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2] : (tensor<128x4x9xf32>) -> tensor<128x4x9x9xf32>
    %395 = stablehlo.multiply %184, %394 : tensor<128x4x9x9xf32>
    %396 = stablehlo.add %389, %395 : tensor<128x4x9x9xf32>
    %397 = call @_where_2(%173#1, %396) : (tensor<128x4x9x9xi1>, tensor<128x4x9x9xf32>) -> tensor<128x4x9x9xf32>
    %398 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f32>) -> tensor<128x4x9x9xf32>
    %399 = stablehlo.divide %397, %398 : tensor<128x4x9x9xf32>
    %400 = stablehlo.dot_general %399, %160, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x9xf32>, tensor<128x4x9x32xf32>) -> tensor<128x4x9x32xf32>
    %401 = stablehlo.transpose %400, dims = [0, 1, 3, 2] : (tensor<128x4x9x32xf32>) -> tensor<128x4x32x9xf32>
    %402 = stablehlo.dot_general %399, %165, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x9xf32>, tensor<128x4x32x9xf32>) -> tensor<128x4x9x32xf32>
    %403 = stablehlo.transpose %401, dims = [0, 1, 3, 2] : (tensor<128x4x32x9xf32>) -> tensor<128x4x9x32xf32>
    %404 = stablehlo.transpose %387, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %405 = stablehlo.reshape %404 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %406 = stablehlo.transpose %403, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %407 = stablehlo.reshape %406 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %408 = stablehlo.transpose %402, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %409 = stablehlo.reshape %408 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %410 = stablehlo.concatenate %409, %407, %405, dim = 2 : (tensor<128x9x128xf32>, tensor<128x9x128xf32>, tensor<128x9x128xf32>) -> tensor<128x9x384xf32>
    %411 = stablehlo.dot_general %410, %154, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x384xf32>, tensor<128x9x128xf32>) -> tensor<384x128xf32>
    %412 = stablehlo.transpose %411, dims = [1, 0] : (tensor<384x128xf32>) -> tensor<128x384xf32>
    %413 = stablehlo.dot_general %410, %arg15, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x384xf32>, tensor<128x384xf32>) -> tensor<128x9x128xf32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %414 = stablehlo.reduce(%413 init: %cst_92) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %415 = stablehlo.reshape %414 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %416 = stablehlo.reduce(%415 init: %cst_93) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %417 = stablehlo.multiply %148, %413 : tensor<128x9x128xf32>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %418 = stablehlo.reduce(%417 init: %cst_94) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %419 = stablehlo.reshape %418 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %420 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %421 = stablehlo.multiply %413, %420 : tensor<128x9x128xf32>
    %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %422 = stablehlo.reduce(%419 init: %cst_95) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %423 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %424 = stablehlo.divide %421, %423 : tensor<128x9x128xf32>
    %425 = stablehlo.negate %424 : tensor<128x9x128xf32>
    %426 = stablehlo.multiply %148, %425 : tensor<128x9x128xf32>
    %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %427 = stablehlo.reduce(%426 init: %cst_96) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %428 = stablehlo.reshape %427 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %429 = stablehlo.multiply %428, %146 : tensor<128x9x1xf32>
    %430 = stablehlo.negate %424 : tensor<128x9x128xf32>
    %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %431 = stablehlo.reduce(%430 init: %cst_97) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %432 = stablehlo.reshape %431 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %433 = stablehlo.add %380, %424 : tensor<128x9x128xf32>
    %cst_98 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %434 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %435 = stablehlo.divide %429, %434 : tensor<128x9x1xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %436 = stablehlo.reduce(%435 init: %cst_99) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %438 = stablehlo.multiply %437, %135 : tensor<128x9x128xf32>
    %439 = stablehlo.negate %438 : tensor<128x9x128xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %440 = stablehlo.reduce(%439 init: %cst_100) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %441 = stablehlo.reshape %440 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %442 = stablehlo.add %433, %438 : tensor<128x9x128xf32>
    %443 = stablehlo.add %432, %441 : tensor<128x9x1xf32>
    %cst_101 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %444 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %445 = stablehlo.divide %443, %444 : tensor<128x9x1xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %446 = stablehlo.reduce(%445 init: %cst_102) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %448 = stablehlo.add %442, %447 : tensor<128x9x128xf32>
    %449 = stablehlo.dot_general %448, %124, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x9x512xf32>) -> tensor<128x512xf32>
    %450 = stablehlo.transpose %449, dims = [1, 0] : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %451 = stablehlo.dot_general %448, %arg5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<512x128xf32>) -> tensor<128x9x512xf32>
    %452 = stablehlo.multiply %103, %451 : tensor<128x9x512xf32>
    %453 = stablehlo.multiply %451, %123 : tensor<128x9x512xf32>
    %cst_103 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %454 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %455 = stablehlo.multiply %454, %452 : tensor<128x9x512xf32>
    %456 = stablehlo.multiply %455, %119 : tensor<128x9x512xf32>
    %cst_104 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %457 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %458 = stablehlo.multiply %457, %456 : tensor<128x9x512xf32>
    %459 = stablehlo.add %453, %458 : tensor<128x9x512xf32>
    %cst_105 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %460 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<128x9x512xf32>
    %461 = stablehlo.multiply %460, %458 : tensor<128x9x512xf32>
    %462 = stablehlo.multiply %461, %108 : tensor<128x9x512xf32>
    %463 = stablehlo.add %459, %462 : tensor<128x9x512xf32>
    %464 = stablehlo.dot_general %463, %102, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x512xf32>, tensor<128x9x128xf32>) -> tensor<512x128xf32>
    %465 = stablehlo.transpose %464, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %466 = stablehlo.dot_general %463, %arg0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x512xf32>, tensor<128x512xf32>) -> tensor<128x9x128xf32>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %467 = stablehlo.reduce(%466 init: %cst_106) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %468 = stablehlo.reshape %467 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %469 = stablehlo.reduce(%468 init: %cst_107) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %470 = stablehlo.multiply %96, %466 : tensor<128x9x128xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %471 = stablehlo.reduce(%470 init: %cst_108) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %472 = stablehlo.reshape %471 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %473 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %474 = stablehlo.multiply %466, %473 : tensor<128x9x128xf32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %475 = stablehlo.reduce(%472 init: %cst_109) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %476 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %477 = stablehlo.divide %474, %476 : tensor<128x9x128xf32>
    %478 = stablehlo.negate %477 : tensor<128x9x128xf32>
    %479 = stablehlo.multiply %96, %478 : tensor<128x9x128xf32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %480 = stablehlo.reduce(%479 init: %cst_110) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %481 = stablehlo.reshape %480 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %482 = stablehlo.multiply %481, %94 : tensor<128x9x1xf32>
    %483 = stablehlo.negate %477 : tensor<128x9x128xf32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %484 = stablehlo.reduce(%483 init: %cst_111) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %485 = stablehlo.reshape %484 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %486 = stablehlo.add %448, %477 : tensor<128x9x128xf32>
    %cst_112 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %487 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %488 = stablehlo.divide %482, %487 : tensor<128x9x1xf32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %489 = stablehlo.reduce(%488 init: %cst_113) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %491 = stablehlo.multiply %490, %83 : tensor<128x9x128xf32>
    %492 = stablehlo.negate %491 : tensor<128x9x128xf32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %493 = stablehlo.reduce(%492 init: %cst_114) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %494 = stablehlo.reshape %493 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %495 = stablehlo.add %486, %491 : tensor<128x9x128xf32>
    %496 = stablehlo.add %485, %494 : tensor<128x9x1xf32>
    %cst_115 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %497 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %498 = stablehlo.divide %496, %497 : tensor<128x9x1xf32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %499 = stablehlo.reduce(%498 init: %cst_116) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %501 = stablehlo.add %495, %500 : tensor<128x9x128xf32>
    %502 = stablehlo.dot_general %501, %72, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x9x128xf32>) -> tensor<128x128xf32>
    %503 = stablehlo.transpose %502, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %504 = stablehlo.dot_general %501, %arg6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x128xf32>, tensor<128x128xf32>) -> tensor<128x9x128xf32>
    %505 = stablehlo.reshape %504 : (tensor<128x9x128xf32>) -> tensor<128x9x4x32xf32>
    %506 = stablehlo.transpose %505, dims = [0, 2, 1, 3] : (tensor<128x9x4x32xf32>) -> tensor<128x4x9x32xf32>
    %507 = stablehlo.dot_general %506, %69, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x32xf32>, tensor<128x4x9x9xf32>) -> tensor<128x4x32x9xf32>
    %508 = stablehlo.transpose %507, dims = [0, 1, 3, 2] : (tensor<128x4x32x9xf32>) -> tensor<128x4x9x32xf32>
    %509 = stablehlo.dot_general %506, %49, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x32xf32>, tensor<128x4x9x32xf32>) -> tensor<128x4x9x9xf32>
    %510 = stablehlo.multiply %69, %509 : tensor<128x4x9x9xf32>
    %511 = stablehlo.negate %510 : tensor<128x4x9x9xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %512 = stablehlo.reduce(%511 init: %cst_117) applies stablehlo.add across dimensions = [3] : (tensor<128x4x9x9xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %513 = stablehlo.reshape %512 : (tensor<128x4x9xf32>) -> tensor<128x4x9x1xf32>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %514 = stablehlo.reduce(%513 init: %cst_118) applies stablehlo.add across dimensions = [3] : (tensor<128x4x9x1xf32>, tensor<f32>) -> tensor<128x4x9xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2] : (tensor<128x4x9xf32>) -> tensor<128x4x9x9xf32>
    %516 = stablehlo.multiply %69, %515 : tensor<128x4x9x9xf32>
    %517 = stablehlo.add %510, %516 : tensor<128x4x9x9xf32>
    %518 = call @_where_2(%58#1, %517) : (tensor<128x4x9x9xi1>, tensor<128x4x9x9xf32>) -> tensor<128x4x9x9xf32>
    %519 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<f32>) -> tensor<128x4x9x9xf32>
    %520 = stablehlo.divide %518, %519 : tensor<128x4x9x9xf32>
    %521 = stablehlo.dot_general %520, %45, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x9xf32>, tensor<128x4x9x32xf32>) -> tensor<128x4x9x32xf32>
    %522 = stablehlo.transpose %521, dims = [0, 1, 3, 2] : (tensor<128x4x9x32xf32>) -> tensor<128x4x32x9xf32>
    %523 = stablehlo.dot_general %520, %50, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<128x4x9x9xf32>, tensor<128x4x32x9xf32>) -> tensor<128x4x9x32xf32>
    %524 = stablehlo.transpose %522, dims = [0, 1, 3, 2] : (tensor<128x4x32x9xf32>) -> tensor<128x4x9x32xf32>
    %525 = stablehlo.transpose %508, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %526 = stablehlo.reshape %525 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %527 = stablehlo.transpose %524, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %528 = stablehlo.reshape %527 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %529 = stablehlo.transpose %523, dims = [0, 2, 1, 3] : (tensor<128x4x9x32xf32>) -> tensor<128x9x4x32xf32>
    %530 = stablehlo.reshape %529 : (tensor<128x9x4x32xf32>) -> tensor<128x9x128xf32>
    %531 = stablehlo.concatenate %530, %528, %526, dim = 2 : (tensor<128x9x128xf32>, tensor<128x9x128xf32>, tensor<128x9x128xf32>) -> tensor<128x9x384xf32>
    %532 = stablehlo.dot_general %531, %39, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x384xf32>, tensor<128x9x128xf32>) -> tensor<384x128xf32>
    %533 = stablehlo.transpose %532, dims = [1, 0] : (tensor<384x128xf32>) -> tensor<128x384xf32>
    %534 = stablehlo.dot_general %531, %arg7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x9x384xf32>, tensor<128x384xf32>) -> tensor<128x9x128xf32>
    %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %535 = stablehlo.reduce(%534 init: %cst_119) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %536 = stablehlo.reshape %535 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %537 = stablehlo.reduce(%536 init: %cst_120) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %538 = stablehlo.multiply %33, %534 : tensor<128x9x128xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %539 = stablehlo.reduce(%538 init: %cst_121) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128xf32>
    %540 = stablehlo.reshape %539 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %541 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<128x9x128xf32>
    %542 = stablehlo.multiply %534, %541 : tensor<128x9x128xf32>
    %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %543 = stablehlo.reduce(%540 init: %cst_122) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %544 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x128xf32>
    %545 = stablehlo.divide %542, %544 : tensor<128x9x128xf32>
    %546 = stablehlo.negate %545 : tensor<128x9x128xf32>
    %547 = stablehlo.multiply %33, %546 : tensor<128x9x128xf32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %548 = stablehlo.reduce(%547 init: %cst_123) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %549 = stablehlo.reshape %548 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %550 = stablehlo.multiply %549, %31 : tensor<128x9x1xf32>
    %551 = stablehlo.negate %545 : tensor<128x9x128xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %552 = stablehlo.reduce(%551 init: %cst_124) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %553 = stablehlo.reshape %552 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %554 = stablehlo.add %501, %545 : tensor<128x9x128xf32>
    %cst_125 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %555 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %556 = stablehlo.divide %550, %555 : tensor<128x9x1xf32>
    %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %557 = stablehlo.reduce(%556 init: %cst_126) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %559 = stablehlo.multiply %558, %20 : tensor<128x9x128xf32>
    %560 = stablehlo.negate %559 : tensor<128x9x128xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %561 = stablehlo.reduce(%560 init: %cst_127) applies stablehlo.add across dimensions = [2] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<128x9xf32>
    %562 = stablehlo.reshape %561 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %563 = stablehlo.add %554, %559 : tensor<128x9x128xf32>
    %564 = stablehlo.add %553, %562 : tensor<128x9x1xf32>
    %cst_128 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %565 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %566 = stablehlo.divide %564, %565 : tensor<128x9x1xf32>
    %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %567 = stablehlo.reduce(%566 init: %cst_129) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %568 = stablehlo.broadcast_in_dim %567, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x128xf32>
    %569 = stablehlo.add %563, %568 : tensor<128x9x128xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %570 = stablehlo.reduce(%569 init: %cst_130) applies stablehlo.add across dimensions = [0] : (tensor<128x9x128xf32>, tensor<f32>) -> tensor<9x128xf32>
    %571 = stablehlo.reshape %570 : (tensor<9x128xf32>) -> tensor<1x9x128xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %572 = stablehlo.reduce(%571 init: %cst_131) applies stablehlo.add across dimensions = [0] : (tensor<1x9x128xf32>, tensor<f32>) -> tensor<9x128xf32>
    %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %573 = stablehlo.pad %572, %cst_132, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<9x128xf32>, tensor<f32>) -> tensor<10x128xf32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %574 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %575 = "stablehlo.scatter"(%574, %6, %569) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg64: tensor<f32>, %arg65: tensor<f32>):
      %1375 = stablehlo.add %arg64, %arg65 : tensor<f32>
      stablehlo.return %1375 : tensor<f32>
    }) : (tensor<15x128xf32>, tensor<128x9x1xi32>, tensor<128x9x128xf32>) -> tensor<15x128xf32>
    %576 = stablehlo.add %293, %575 : tensor<15x128xf32>
    %577 = chlo.square %465 : tensor<128x512xf32> -> tensor<128x512xf32>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %578 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %579 = chlo.square %578 : tensor<128x512xf32> -> tensor<128x512xf32>
    %580 = stablehlo.add %577, %579 : tensor<128x512xf32>
    %581 = chlo.square %537 : tensor<128xf32> -> tensor<128xf32>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %582 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %583 = chlo.square %582 : tensor<128xf32> -> tensor<128xf32>
    %584 = stablehlo.add %581, %583 : tensor<128xf32>
    %585 = chlo.square %543 : tensor<128xf32> -> tensor<128xf32>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %586 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %587 = chlo.square %586 : tensor<128xf32> -> tensor<128xf32>
    %588 = stablehlo.add %585, %587 : tensor<128xf32>
    %589 = chlo.square %469 : tensor<128xf32> -> tensor<128xf32>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %590 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %591 = chlo.square %590 : tensor<128xf32> -> tensor<128xf32>
    %592 = stablehlo.add %589, %591 : tensor<128xf32>
    %593 = chlo.square %475 : tensor<128xf32> -> tensor<128xf32>
    %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %594 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %595 = chlo.square %594 : tensor<128xf32> -> tensor<128xf32>
    %596 = stablehlo.add %593, %595 : tensor<128xf32>
    %597 = chlo.square %450 : tensor<512x128xf32> -> tensor<512x128xf32>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %598 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %599 = chlo.square %598 : tensor<512x128xf32> -> tensor<512x128xf32>
    %600 = stablehlo.add %597, %599 : tensor<512x128xf32>
    %601 = chlo.square %503 : tensor<128x128xf32> -> tensor<128x128xf32>
    %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %602 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %603 = chlo.square %602 : tensor<128x128xf32> -> tensor<128x128xf32>
    %604 = stablehlo.add %601, %603 : tensor<128x128xf32>
    %605 = chlo.square %533 : tensor<128x384xf32> -> tensor<128x384xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %606 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %607 = chlo.square %606 : tensor<128x384xf32> -> tensor<128x384xf32>
    %608 = stablehlo.add %605, %607 : tensor<128x384xf32>
    %609 = chlo.square %344 : tensor<128x512xf32> -> tensor<128x512xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %610 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %611 = chlo.square %610 : tensor<128x512xf32> -> tensor<128x512xf32>
    %612 = stablehlo.add %609, %611 : tensor<128x512xf32>
    %613 = chlo.square %416 : tensor<128xf32> -> tensor<128xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %614 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %615 = chlo.square %614 : tensor<128xf32> -> tensor<128xf32>
    %616 = stablehlo.add %613, %615 : tensor<128xf32>
    %617 = chlo.square %422 : tensor<128xf32> -> tensor<128xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %618 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %619 = chlo.square %618 : tensor<128xf32> -> tensor<128xf32>
    %620 = stablehlo.add %617, %619 : tensor<128xf32>
    %621 = chlo.square %348 : tensor<128xf32> -> tensor<128xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %622 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %623 = chlo.square %622 : tensor<128xf32> -> tensor<128xf32>
    %624 = stablehlo.add %621, %623 : tensor<128xf32>
    %625 = chlo.square %354 : tensor<128xf32> -> tensor<128xf32>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %626 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %627 = chlo.square %626 : tensor<128xf32> -> tensor<128xf32>
    %628 = stablehlo.add %625, %627 : tensor<128xf32>
    %629 = chlo.square %329 : tensor<512x128xf32> -> tensor<512x128xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %630 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %631 = chlo.square %630 : tensor<512x128xf32> -> tensor<512x128xf32>
    %632 = stablehlo.add %629, %631 : tensor<512x128xf32>
    %633 = chlo.square %382 : tensor<128x128xf32> -> tensor<128x128xf32>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %634 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %635 = chlo.square %634 : tensor<128x128xf32> -> tensor<128x128xf32>
    %636 = stablehlo.add %633, %635 : tensor<128x128xf32>
    %637 = chlo.square %412 : tensor<128x384xf32> -> tensor<128x384xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %638 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %639 = chlo.square %638 : tensor<128x384xf32> -> tensor<128x384xf32>
    %640 = stablehlo.add %637, %639 : tensor<128x384xf32>
    %641 = chlo.square %296 : tensor<128xf32> -> tensor<128xf32>
    %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %642 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %643 = chlo.square %642 : tensor<128xf32> -> tensor<128xf32>
    %644 = stablehlo.add %641, %643 : tensor<128xf32>
    %645 = chlo.square %302 : tensor<128xf32> -> tensor<128xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %646 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %647 = chlo.square %646 : tensor<128xf32> -> tensor<128xf32>
    %648 = stablehlo.add %645, %647 : tensor<128xf32>
    %649 = chlo.square %573 : tensor<10x128xf32> -> tensor<10x128xf32>
    %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %650 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %651 = chlo.square %650 : tensor<10x128xf32> -> tensor<10x128xf32>
    %652 = stablehlo.add %649, %651 : tensor<10x128xf32>
    %653 = chlo.square %576 : tensor<15x128xf32> -> tensor<15x128xf32>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %654 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %655 = chlo.square %654 : tensor<15x128xf32> -> tensor<15x128xf32>
    %656 = stablehlo.add %653, %655 : tensor<15x128xf32>
    %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %657 = stablehlo.reduce(%580 init: %cst_154) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x512xf32>, tensor<f32>) -> tensor<f32>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %658 = stablehlo.reduce(%584 init: %cst_155) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %659 = stablehlo.reduce(%588 init: %cst_156) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %660 = stablehlo.reduce(%592 init: %cst_157) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %661 = stablehlo.reduce(%596 init: %cst_158) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %662 = stablehlo.reduce(%600 init: %cst_159) applies stablehlo.add across dimensions = [0, 1] : (tensor<512x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %663 = stablehlo.reduce(%604 init: %cst_160) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %664 = stablehlo.reduce(%608 init: %cst_161) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<f32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %665 = stablehlo.reduce(%612 init: %cst_162) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x512xf32>, tensor<f32>) -> tensor<f32>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %666 = stablehlo.reduce(%616 init: %cst_163) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %667 = stablehlo.reduce(%620 init: %cst_164) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %668 = stablehlo.reduce(%624 init: %cst_165) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %669 = stablehlo.reduce(%628 init: %cst_166) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %670 = stablehlo.reduce(%632 init: %cst_167) applies stablehlo.add across dimensions = [0, 1] : (tensor<512x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %671 = stablehlo.reduce(%636 init: %cst_168) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %672 = stablehlo.reduce(%640 init: %cst_169) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<f32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %673 = stablehlo.reduce(%644 init: %cst_170) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %674 = stablehlo.reduce(%648 init: %cst_171) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %675 = stablehlo.reduce(%652 init: %cst_172) applies stablehlo.add across dimensions = [0, 1] : (tensor<10x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %676 = stablehlo.reduce(%656 init: %cst_173) applies stablehlo.add across dimensions = [0, 1] : (tensor<15x128xf32>, tensor<f32>) -> tensor<f32>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %677 = stablehlo.add %cst_174, %657 : tensor<f32>
    %678 = stablehlo.add %677, %658 : tensor<f32>
    %679 = stablehlo.add %678, %659 : tensor<f32>
    %680 = stablehlo.add %679, %660 : tensor<f32>
    %681 = stablehlo.add %680, %661 : tensor<f32>
    %682 = stablehlo.add %681, %662 : tensor<f32>
    %683 = stablehlo.add %682, %663 : tensor<f32>
    %684 = stablehlo.add %683, %664 : tensor<f32>
    %685 = stablehlo.add %684, %665 : tensor<f32>
    %686 = stablehlo.add %685, %666 : tensor<f32>
    %687 = stablehlo.add %686, %667 : tensor<f32>
    %688 = stablehlo.add %687, %668 : tensor<f32>
    %689 = stablehlo.add %688, %669 : tensor<f32>
    %690 = stablehlo.add %689, %670 : tensor<f32>
    %691 = stablehlo.add %690, %671 : tensor<f32>
    %692 = stablehlo.add %691, %672 : tensor<f32>
    %693 = stablehlo.add %692, %673 : tensor<f32>
    %694 = stablehlo.add %693, %674 : tensor<f32>
    %695 = stablehlo.add %694, %675 : tensor<f32>
    %696 = stablehlo.add %695, %676 : tensor<f32>
    %697 = stablehlo.sqrt %696 : tensor<f32>
    %cst_175 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %698 = stablehlo.compare LT, %697, %cst_175, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_176 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %699 = call @_where_3(%698, %cst_176, %696) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %700 = stablehlo.sqrt %699 : tensor<f32>
    %701 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %702 = stablehlo.divide %465, %701 : tensor<128x512xf32>
    %cst_177 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %703 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %704 = stablehlo.multiply %702, %703 : tensor<128x512xf32>
    %705 = stablehlo.select %698, %465, %704 : tensor<i1>, tensor<128x512xf32>
    %706 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %707 = stablehlo.divide %537, %706 : tensor<128xf32>
    %cst_178 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %708 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %709 = stablehlo.multiply %707, %708 : tensor<128xf32>
    %710 = stablehlo.select %698, %537, %709 : tensor<i1>, tensor<128xf32>
    %711 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %712 = stablehlo.divide %543, %711 : tensor<128xf32>
    %cst_179 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %713 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %714 = stablehlo.multiply %712, %713 : tensor<128xf32>
    %715 = stablehlo.select %698, %543, %714 : tensor<i1>, tensor<128xf32>
    %716 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %717 = stablehlo.divide %469, %716 : tensor<128xf32>
    %cst_180 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %718 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %719 = stablehlo.multiply %717, %718 : tensor<128xf32>
    %720 = stablehlo.select %698, %469, %719 : tensor<i1>, tensor<128xf32>
    %721 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %722 = stablehlo.divide %475, %721 : tensor<128xf32>
    %cst_181 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %723 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %724 = stablehlo.multiply %722, %723 : tensor<128xf32>
    %725 = stablehlo.select %698, %475, %724 : tensor<i1>, tensor<128xf32>
    %726 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %727 = stablehlo.divide %450, %726 : tensor<512x128xf32>
    %cst_182 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %728 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %729 = stablehlo.multiply %727, %728 : tensor<512x128xf32>
    %730 = stablehlo.select %698, %450, %729 : tensor<i1>, tensor<512x128xf32>
    %731 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %732 = stablehlo.divide %503, %731 : tensor<128x128xf32>
    %cst_183 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %733 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %734 = stablehlo.multiply %732, %733 : tensor<128x128xf32>
    %735 = stablehlo.select %698, %503, %734 : tensor<i1>, tensor<128x128xf32>
    %736 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %737 = stablehlo.divide %533, %736 : tensor<128x384xf32>
    %cst_184 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %738 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %739 = stablehlo.multiply %737, %738 : tensor<128x384xf32>
    %740 = stablehlo.select %698, %533, %739 : tensor<i1>, tensor<128x384xf32>
    %741 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %742 = stablehlo.divide %344, %741 : tensor<128x512xf32>
    %cst_185 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %743 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %744 = stablehlo.multiply %742, %743 : tensor<128x512xf32>
    %745 = stablehlo.select %698, %344, %744 : tensor<i1>, tensor<128x512xf32>
    %746 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %747 = stablehlo.divide %416, %746 : tensor<128xf32>
    %cst_186 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %748 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %749 = stablehlo.multiply %747, %748 : tensor<128xf32>
    %750 = stablehlo.select %698, %416, %749 : tensor<i1>, tensor<128xf32>
    %751 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %752 = stablehlo.divide %422, %751 : tensor<128xf32>
    %cst_187 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %753 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %754 = stablehlo.multiply %752, %753 : tensor<128xf32>
    %755 = stablehlo.select %698, %422, %754 : tensor<i1>, tensor<128xf32>
    %756 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %757 = stablehlo.divide %348, %756 : tensor<128xf32>
    %cst_188 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %758 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %759 = stablehlo.multiply %757, %758 : tensor<128xf32>
    %760 = stablehlo.select %698, %348, %759 : tensor<i1>, tensor<128xf32>
    %761 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %762 = stablehlo.divide %354, %761 : tensor<128xf32>
    %cst_189 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %763 = stablehlo.broadcast_in_dim %cst_189, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %764 = stablehlo.multiply %762, %763 : tensor<128xf32>
    %765 = stablehlo.select %698, %354, %764 : tensor<i1>, tensor<128xf32>
    %766 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %767 = stablehlo.divide %329, %766 : tensor<512x128xf32>
    %cst_190 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %768 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %769 = stablehlo.multiply %767, %768 : tensor<512x128xf32>
    %770 = stablehlo.select %698, %329, %769 : tensor<i1>, tensor<512x128xf32>
    %771 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %772 = stablehlo.divide %382, %771 : tensor<128x128xf32>
    %cst_191 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %773 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %774 = stablehlo.multiply %772, %773 : tensor<128x128xf32>
    %775 = stablehlo.select %698, %382, %774 : tensor<i1>, tensor<128x128xf32>
    %776 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %777 = stablehlo.divide %412, %776 : tensor<128x384xf32>
    %cst_192 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %778 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %779 = stablehlo.multiply %777, %778 : tensor<128x384xf32>
    %780 = stablehlo.select %698, %412, %779 : tensor<i1>, tensor<128x384xf32>
    %781 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %782 = stablehlo.divide %296, %781 : tensor<128xf32>
    %cst_193 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %783 = stablehlo.broadcast_in_dim %cst_193, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %784 = stablehlo.multiply %782, %783 : tensor<128xf32>
    %785 = stablehlo.select %698, %296, %784 : tensor<i1>, tensor<128xf32>
    %786 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %787 = stablehlo.divide %302, %786 : tensor<128xf32>
    %cst_194 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %788 = stablehlo.broadcast_in_dim %cst_194, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %789 = stablehlo.multiply %787, %788 : tensor<128xf32>
    %790 = stablehlo.select %698, %302, %789 : tensor<i1>, tensor<128xf32>
    %791 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %792 = stablehlo.divide %573, %791 : tensor<10x128xf32>
    %cst_195 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %793 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %794 = stablehlo.multiply %792, %793 : tensor<10x128xf32>
    %795 = stablehlo.select %698, %573, %794 : tensor<i1>, tensor<10x128xf32>
    %796 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %797 = stablehlo.divide %576, %796 : tensor<15x128xf32>
    %cst_196 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %798 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %799 = stablehlo.multiply %797, %798 : tensor<15x128xf32>
    %800 = stablehlo.select %698, %576, %799 : tensor<i1>, tensor<15x128xf32>
    %cst_197 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %801 = stablehlo.broadcast_in_dim %cst_197, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %802 = stablehlo.multiply %801, %705 : tensor<128x512xf32>
    %cst_198 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %803 = stablehlo.broadcast_in_dim %cst_198, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %804 = stablehlo.multiply %803, %arg21 : tensor<128x512xf32>
    %805 = stablehlo.add %802, %804 : tensor<128x512xf32>
    %cst_199 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %806 = stablehlo.broadcast_in_dim %cst_199, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %807 = stablehlo.multiply %806, %710 : tensor<128xf32>
    %cst_200 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %808 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %809 = stablehlo.multiply %808, %arg22 : tensor<128xf32>
    %810 = stablehlo.add %807, %809 : tensor<128xf32>
    %cst_201 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %811 = stablehlo.broadcast_in_dim %cst_201, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %812 = stablehlo.multiply %811, %715 : tensor<128xf32>
    %cst_202 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %813 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %814 = stablehlo.multiply %813, %arg23 : tensor<128xf32>
    %815 = stablehlo.add %812, %814 : tensor<128xf32>
    %cst_203 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %816 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %817 = stablehlo.multiply %816, %720 : tensor<128xf32>
    %cst_204 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %818 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %819 = stablehlo.multiply %818, %arg24 : tensor<128xf32>
    %820 = stablehlo.add %817, %819 : tensor<128xf32>
    %cst_205 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %821 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %822 = stablehlo.multiply %821, %725 : tensor<128xf32>
    %cst_206 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %823 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %824 = stablehlo.multiply %823, %arg25 : tensor<128xf32>
    %825 = stablehlo.add %822, %824 : tensor<128xf32>
    %cst_207 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %826 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %827 = stablehlo.multiply %826, %730 : tensor<512x128xf32>
    %cst_208 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %828 = stablehlo.broadcast_in_dim %cst_208, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %829 = stablehlo.multiply %828, %arg26 : tensor<512x128xf32>
    %830 = stablehlo.add %827, %829 : tensor<512x128xf32>
    %cst_209 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %831 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %832 = stablehlo.multiply %831, %735 : tensor<128x128xf32>
    %cst_210 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %833 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %834 = stablehlo.multiply %833, %arg27 : tensor<128x128xf32>
    %835 = stablehlo.add %832, %834 : tensor<128x128xf32>
    %cst_211 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %836 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %837 = stablehlo.multiply %836, %740 : tensor<128x384xf32>
    %cst_212 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %838 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %839 = stablehlo.multiply %838, %arg28 : tensor<128x384xf32>
    %840 = stablehlo.add %837, %839 : tensor<128x384xf32>
    %cst_213 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %841 = stablehlo.broadcast_in_dim %cst_213, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %842 = stablehlo.multiply %841, %745 : tensor<128x512xf32>
    %cst_214 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %843 = stablehlo.broadcast_in_dim %cst_214, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %844 = stablehlo.multiply %843, %arg29 : tensor<128x512xf32>
    %845 = stablehlo.add %842, %844 : tensor<128x512xf32>
    %cst_215 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %846 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %847 = stablehlo.multiply %846, %750 : tensor<128xf32>
    %cst_216 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %848 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %849 = stablehlo.multiply %848, %arg30 : tensor<128xf32>
    %850 = stablehlo.add %847, %849 : tensor<128xf32>
    %cst_217 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %851 = stablehlo.broadcast_in_dim %cst_217, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %852 = stablehlo.multiply %851, %755 : tensor<128xf32>
    %cst_218 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %853 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %854 = stablehlo.multiply %853, %arg31 : tensor<128xf32>
    %855 = stablehlo.add %852, %854 : tensor<128xf32>
    %cst_219 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %856 = stablehlo.broadcast_in_dim %cst_219, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %857 = stablehlo.multiply %856, %760 : tensor<128xf32>
    %cst_220 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %858 = stablehlo.broadcast_in_dim %cst_220, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %859 = stablehlo.multiply %858, %arg32 : tensor<128xf32>
    %860 = stablehlo.add %857, %859 : tensor<128xf32>
    %cst_221 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %861 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %862 = stablehlo.multiply %861, %765 : tensor<128xf32>
    %cst_222 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %863 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %864 = stablehlo.multiply %863, %arg33 : tensor<128xf32>
    %865 = stablehlo.add %862, %864 : tensor<128xf32>
    %cst_223 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %866 = stablehlo.broadcast_in_dim %cst_223, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %867 = stablehlo.multiply %866, %770 : tensor<512x128xf32>
    %cst_224 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %868 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %869 = stablehlo.multiply %868, %arg34 : tensor<512x128xf32>
    %870 = stablehlo.add %867, %869 : tensor<512x128xf32>
    %cst_225 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %871 = stablehlo.broadcast_in_dim %cst_225, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %872 = stablehlo.multiply %871, %775 : tensor<128x128xf32>
    %cst_226 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %873 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %874 = stablehlo.multiply %873, %arg35 : tensor<128x128xf32>
    %875 = stablehlo.add %872, %874 : tensor<128x128xf32>
    %cst_227 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %876 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %877 = stablehlo.multiply %876, %780 : tensor<128x384xf32>
    %cst_228 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %878 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %879 = stablehlo.multiply %878, %arg36 : tensor<128x384xf32>
    %880 = stablehlo.add %877, %879 : tensor<128x384xf32>
    %cst_229 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %881 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %882 = stablehlo.multiply %881, %785 : tensor<128xf32>
    %cst_230 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %883 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %884 = stablehlo.multiply %883, %arg37 : tensor<128xf32>
    %885 = stablehlo.add %882, %884 : tensor<128xf32>
    %cst_231 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %886 = stablehlo.broadcast_in_dim %cst_231, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %887 = stablehlo.multiply %886, %790 : tensor<128xf32>
    %cst_232 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %888 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %889 = stablehlo.multiply %888, %arg38 : tensor<128xf32>
    %890 = stablehlo.add %887, %889 : tensor<128xf32>
    %cst_233 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %891 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %892 = stablehlo.multiply %891, %795 : tensor<10x128xf32>
    %cst_234 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %893 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %894 = stablehlo.multiply %893, %arg39 : tensor<10x128xf32>
    %895 = stablehlo.add %892, %894 : tensor<10x128xf32>
    %cst_235 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %896 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %897 = stablehlo.multiply %896, %800 : tensor<15x128xf32>
    %cst_236 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %898 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %899 = stablehlo.multiply %898, %arg40 : tensor<15x128xf32>
    %900 = stablehlo.add %897, %899 : tensor<15x128xf32>
    %901 = stablehlo.multiply %705, %705 : tensor<128x512xf32>
    %cst_237 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %902 = stablehlo.broadcast_in_dim %cst_237, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %903 = stablehlo.multiply %902, %901 : tensor<128x512xf32>
    %cst_238 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %904 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %905 = stablehlo.multiply %904, %arg41 : tensor<128x512xf32>
    %906 = stablehlo.add %903, %905 : tensor<128x512xf32>
    %907 = stablehlo.multiply %710, %710 : tensor<128xf32>
    %cst_239 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %908 = stablehlo.broadcast_in_dim %cst_239, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %909 = stablehlo.multiply %908, %907 : tensor<128xf32>
    %cst_240 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %910 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %911 = stablehlo.multiply %910, %arg42 : tensor<128xf32>
    %912 = stablehlo.add %909, %911 : tensor<128xf32>
    %913 = stablehlo.multiply %715, %715 : tensor<128xf32>
    %cst_241 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %914 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %915 = stablehlo.multiply %914, %913 : tensor<128xf32>
    %cst_242 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %916 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %917 = stablehlo.multiply %916, %arg43 : tensor<128xf32>
    %918 = stablehlo.add %915, %917 : tensor<128xf32>
    %919 = stablehlo.multiply %720, %720 : tensor<128xf32>
    %cst_243 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %920 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %921 = stablehlo.multiply %920, %919 : tensor<128xf32>
    %cst_244 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %922 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %923 = stablehlo.multiply %922, %arg44 : tensor<128xf32>
    %924 = stablehlo.add %921, %923 : tensor<128xf32>
    %925 = stablehlo.multiply %725, %725 : tensor<128xf32>
    %cst_245 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %926 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %927 = stablehlo.multiply %926, %925 : tensor<128xf32>
    %cst_246 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %928 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %929 = stablehlo.multiply %928, %arg45 : tensor<128xf32>
    %930 = stablehlo.add %927, %929 : tensor<128xf32>
    %931 = stablehlo.multiply %730, %730 : tensor<512x128xf32>
    %cst_247 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %932 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %933 = stablehlo.multiply %932, %931 : tensor<512x128xf32>
    %cst_248 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %934 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %935 = stablehlo.multiply %934, %arg46 : tensor<512x128xf32>
    %936 = stablehlo.add %933, %935 : tensor<512x128xf32>
    %937 = stablehlo.multiply %735, %735 : tensor<128x128xf32>
    %cst_249 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %938 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %939 = stablehlo.multiply %938, %937 : tensor<128x128xf32>
    %cst_250 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %940 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %941 = stablehlo.multiply %940, %arg47 : tensor<128x128xf32>
    %942 = stablehlo.add %939, %941 : tensor<128x128xf32>
    %943 = stablehlo.multiply %740, %740 : tensor<128x384xf32>
    %cst_251 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %944 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %945 = stablehlo.multiply %944, %943 : tensor<128x384xf32>
    %cst_252 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %946 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %947 = stablehlo.multiply %946, %arg48 : tensor<128x384xf32>
    %948 = stablehlo.add %945, %947 : tensor<128x384xf32>
    %949 = stablehlo.multiply %745, %745 : tensor<128x512xf32>
    %cst_253 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %950 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %951 = stablehlo.multiply %950, %949 : tensor<128x512xf32>
    %cst_254 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %952 = stablehlo.broadcast_in_dim %cst_254, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %953 = stablehlo.multiply %952, %arg49 : tensor<128x512xf32>
    %954 = stablehlo.add %951, %953 : tensor<128x512xf32>
    %955 = stablehlo.multiply %750, %750 : tensor<128xf32>
    %cst_255 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %956 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %957 = stablehlo.multiply %956, %955 : tensor<128xf32>
    %cst_256 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %958 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %959 = stablehlo.multiply %958, %arg50 : tensor<128xf32>
    %960 = stablehlo.add %957, %959 : tensor<128xf32>
    %961 = stablehlo.multiply %755, %755 : tensor<128xf32>
    %cst_257 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %962 = stablehlo.broadcast_in_dim %cst_257, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %963 = stablehlo.multiply %962, %961 : tensor<128xf32>
    %cst_258 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %964 = stablehlo.broadcast_in_dim %cst_258, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %965 = stablehlo.multiply %964, %arg51 : tensor<128xf32>
    %966 = stablehlo.add %963, %965 : tensor<128xf32>
    %967 = stablehlo.multiply %760, %760 : tensor<128xf32>
    %cst_259 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %968 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %969 = stablehlo.multiply %968, %967 : tensor<128xf32>
    %cst_260 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %970 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %971 = stablehlo.multiply %970, %arg52 : tensor<128xf32>
    %972 = stablehlo.add %969, %971 : tensor<128xf32>
    %973 = stablehlo.multiply %765, %765 : tensor<128xf32>
    %cst_261 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %974 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %975 = stablehlo.multiply %974, %973 : tensor<128xf32>
    %cst_262 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %976 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %977 = stablehlo.multiply %976, %arg53 : tensor<128xf32>
    %978 = stablehlo.add %975, %977 : tensor<128xf32>
    %979 = stablehlo.multiply %770, %770 : tensor<512x128xf32>
    %cst_263 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %980 = stablehlo.broadcast_in_dim %cst_263, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %981 = stablehlo.multiply %980, %979 : tensor<512x128xf32>
    %cst_264 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %982 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %983 = stablehlo.multiply %982, %arg54 : tensor<512x128xf32>
    %984 = stablehlo.add %981, %983 : tensor<512x128xf32>
    %985 = stablehlo.multiply %775, %775 : tensor<128x128xf32>
    %cst_265 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %986 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %987 = stablehlo.multiply %986, %985 : tensor<128x128xf32>
    %cst_266 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %988 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %989 = stablehlo.multiply %988, %arg55 : tensor<128x128xf32>
    %990 = stablehlo.add %987, %989 : tensor<128x128xf32>
    %991 = stablehlo.multiply %780, %780 : tensor<128x384xf32>
    %cst_267 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %992 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %993 = stablehlo.multiply %992, %991 : tensor<128x384xf32>
    %cst_268 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %994 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %995 = stablehlo.multiply %994, %arg56 : tensor<128x384xf32>
    %996 = stablehlo.add %993, %995 : tensor<128x384xf32>
    %997 = stablehlo.multiply %785, %785 : tensor<128xf32>
    %cst_269 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %998 = stablehlo.broadcast_in_dim %cst_269, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %999 = stablehlo.multiply %998, %997 : tensor<128xf32>
    %cst_270 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %1000 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1001 = stablehlo.multiply %1000, %arg57 : tensor<128xf32>
    %1002 = stablehlo.add %999, %1001 : tensor<128xf32>
    %1003 = stablehlo.multiply %790, %790 : tensor<128xf32>
    %cst_271 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %1004 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1005 = stablehlo.multiply %1004, %1003 : tensor<128xf32>
    %cst_272 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %1006 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1007 = stablehlo.multiply %1006, %arg58 : tensor<128xf32>
    %1008 = stablehlo.add %1005, %1007 : tensor<128xf32>
    %1009 = stablehlo.multiply %795, %795 : tensor<10x128xf32>
    %cst_273 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %1010 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1011 = stablehlo.multiply %1010, %1009 : tensor<10x128xf32>
    %cst_274 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %1012 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1013 = stablehlo.multiply %1012, %arg59 : tensor<10x128xf32>
    %1014 = stablehlo.add %1011, %1013 : tensor<10x128xf32>
    %1015 = stablehlo.multiply %800, %800 : tensor<15x128xf32>
    %cst_275 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %1016 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1017 = stablehlo.multiply %1016, %1015 : tensor<15x128xf32>
    %cst_276 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %1018 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1019 = stablehlo.multiply %1018, %arg60 : tensor<15x128xf32>
    %1020 = stablehlo.add %1017, %1019 : tensor<15x128xf32>
    %c_277 = stablehlo.constant dense<2147483647> : tensor<i32>
    %1021 = stablehlo.compare LT, %arg20, %c_277, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_278 = stablehlo.constant dense<1> : tensor<i32>
    %1022 = stablehlo.add %arg20, %c_278 : tensor<i32>
    %c_279 = stablehlo.constant dense<2147483647> : tensor<i32>
    %1023 = call @_where_4(%1021, %1022, %c_279) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %cst_280 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %1024 = stablehlo.convert %1023 : (tensor<i32>) -> tensor<f32>
    %1025 = stablehlo.power %cst_280, %1024 : tensor<f32>
    %cst_281 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1026 = stablehlo.subtract %cst_281, %1025 : tensor<f32>
    %1027 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1028 = stablehlo.divide %805, %1027 : tensor<128x512xf32>
    %1029 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1030 = stablehlo.divide %810, %1029 : tensor<128xf32>
    %1031 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1032 = stablehlo.divide %815, %1031 : tensor<128xf32>
    %1033 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1034 = stablehlo.divide %820, %1033 : tensor<128xf32>
    %1035 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1036 = stablehlo.divide %825, %1035 : tensor<128xf32>
    %1037 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1038 = stablehlo.divide %830, %1037 : tensor<512x128xf32>
    %1039 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1040 = stablehlo.divide %835, %1039 : tensor<128x128xf32>
    %1041 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1042 = stablehlo.divide %840, %1041 : tensor<128x384xf32>
    %1043 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1044 = stablehlo.divide %845, %1043 : tensor<128x512xf32>
    %1045 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1046 = stablehlo.divide %850, %1045 : tensor<128xf32>
    %1047 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1048 = stablehlo.divide %855, %1047 : tensor<128xf32>
    %1049 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1050 = stablehlo.divide %860, %1049 : tensor<128xf32>
    %1051 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1052 = stablehlo.divide %865, %1051 : tensor<128xf32>
    %1053 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1054 = stablehlo.divide %870, %1053 : tensor<512x128xf32>
    %1055 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1056 = stablehlo.divide %875, %1055 : tensor<128x128xf32>
    %1057 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1058 = stablehlo.divide %880, %1057 : tensor<128x384xf32>
    %1059 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1060 = stablehlo.divide %885, %1059 : tensor<128xf32>
    %1061 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1062 = stablehlo.divide %890, %1061 : tensor<128xf32>
    %1063 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1064 = stablehlo.divide %895, %1063 : tensor<10x128xf32>
    %1065 = stablehlo.broadcast_in_dim %1026, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1066 = stablehlo.divide %900, %1065 : tensor<15x128xf32>
    %cst_282 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %1067 = stablehlo.convert %1023 : (tensor<i32>) -> tensor<f32>
    %1068 = stablehlo.power %cst_282, %1067 : tensor<f32>
    %cst_283 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1069 = stablehlo.subtract %cst_283, %1068 : tensor<f32>
    %1070 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1071 = stablehlo.divide %906, %1070 : tensor<128x512xf32>
    %1072 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1073 = stablehlo.divide %912, %1072 : tensor<128xf32>
    %1074 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1075 = stablehlo.divide %918, %1074 : tensor<128xf32>
    %1076 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1077 = stablehlo.divide %924, %1076 : tensor<128xf32>
    %1078 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1079 = stablehlo.divide %930, %1078 : tensor<128xf32>
    %1080 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1081 = stablehlo.divide %936, %1080 : tensor<512x128xf32>
    %1082 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1083 = stablehlo.divide %942, %1082 : tensor<128x128xf32>
    %1084 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1085 = stablehlo.divide %948, %1084 : tensor<128x384xf32>
    %1086 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1087 = stablehlo.divide %954, %1086 : tensor<128x512xf32>
    %1088 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1089 = stablehlo.divide %960, %1088 : tensor<128xf32>
    %1090 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1091 = stablehlo.divide %966, %1090 : tensor<128xf32>
    %1092 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1093 = stablehlo.divide %972, %1092 : tensor<128xf32>
    %1094 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1095 = stablehlo.divide %978, %1094 : tensor<128xf32>
    %1096 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1097 = stablehlo.divide %984, %1096 : tensor<512x128xf32>
    %1098 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1099 = stablehlo.divide %990, %1098 : tensor<128x128xf32>
    %1100 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1101 = stablehlo.divide %996, %1100 : tensor<128x384xf32>
    %1102 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1103 = stablehlo.divide %1002, %1102 : tensor<128xf32>
    %1104 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1105 = stablehlo.divide %1008, %1104 : tensor<128xf32>
    %1106 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1107 = stablehlo.divide %1014, %1106 : tensor<10x128xf32>
    %1108 = stablehlo.broadcast_in_dim %1069, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1109 = stablehlo.divide %1020, %1108 : tensor<15x128xf32>
    %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1110 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1111 = stablehlo.add %1071, %1110 : tensor<128x512xf32>
    %1112 = stablehlo.sqrt %1111 : tensor<128x512xf32>
    %cst_285 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1113 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1114 = stablehlo.add %1112, %1113 : tensor<128x512xf32>
    %1115 = stablehlo.divide %1028, %1114 : tensor<128x512xf32>
    %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1116 = stablehlo.broadcast_in_dim %cst_286, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1117 = stablehlo.add %1073, %1116 : tensor<128xf32>
    %1118 = stablehlo.sqrt %1117 : tensor<128xf32>
    %cst_287 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1119 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1120 = stablehlo.add %1118, %1119 : tensor<128xf32>
    %1121 = stablehlo.divide %1030, %1120 : tensor<128xf32>
    %cst_288 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1122 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1123 = stablehlo.add %1075, %1122 : tensor<128xf32>
    %1124 = stablehlo.sqrt %1123 : tensor<128xf32>
    %cst_289 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1125 = stablehlo.broadcast_in_dim %cst_289, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1126 = stablehlo.add %1124, %1125 : tensor<128xf32>
    %1127 = stablehlo.divide %1032, %1126 : tensor<128xf32>
    %cst_290 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1128 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1129 = stablehlo.add %1077, %1128 : tensor<128xf32>
    %1130 = stablehlo.sqrt %1129 : tensor<128xf32>
    %cst_291 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1131 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1132 = stablehlo.add %1130, %1131 : tensor<128xf32>
    %1133 = stablehlo.divide %1034, %1132 : tensor<128xf32>
    %cst_292 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1134 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1135 = stablehlo.add %1079, %1134 : tensor<128xf32>
    %1136 = stablehlo.sqrt %1135 : tensor<128xf32>
    %cst_293 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1137 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1138 = stablehlo.add %1136, %1137 : tensor<128xf32>
    %1139 = stablehlo.divide %1036, %1138 : tensor<128xf32>
    %cst_294 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1140 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1141 = stablehlo.add %1081, %1140 : tensor<512x128xf32>
    %1142 = stablehlo.sqrt %1141 : tensor<512x128xf32>
    %cst_295 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1143 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1144 = stablehlo.add %1142, %1143 : tensor<512x128xf32>
    %1145 = stablehlo.divide %1038, %1144 : tensor<512x128xf32>
    %cst_296 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1146 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1147 = stablehlo.add %1083, %1146 : tensor<128x128xf32>
    %1148 = stablehlo.sqrt %1147 : tensor<128x128xf32>
    %cst_297 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1149 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1150 = stablehlo.add %1148, %1149 : tensor<128x128xf32>
    %1151 = stablehlo.divide %1040, %1150 : tensor<128x128xf32>
    %cst_298 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1152 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1153 = stablehlo.add %1085, %1152 : tensor<128x384xf32>
    %1154 = stablehlo.sqrt %1153 : tensor<128x384xf32>
    %cst_299 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1155 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1156 = stablehlo.add %1154, %1155 : tensor<128x384xf32>
    %1157 = stablehlo.divide %1042, %1156 : tensor<128x384xf32>
    %cst_300 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1158 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1159 = stablehlo.add %1087, %1158 : tensor<128x512xf32>
    %1160 = stablehlo.sqrt %1159 : tensor<128x512xf32>
    %cst_301 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1161 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1162 = stablehlo.add %1160, %1161 : tensor<128x512xf32>
    %1163 = stablehlo.divide %1044, %1162 : tensor<128x512xf32>
    %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1164 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1165 = stablehlo.add %1089, %1164 : tensor<128xf32>
    %1166 = stablehlo.sqrt %1165 : tensor<128xf32>
    %cst_303 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1167 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1168 = stablehlo.add %1166, %1167 : tensor<128xf32>
    %1169 = stablehlo.divide %1046, %1168 : tensor<128xf32>
    %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1170 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1171 = stablehlo.add %1091, %1170 : tensor<128xf32>
    %1172 = stablehlo.sqrt %1171 : tensor<128xf32>
    %cst_305 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1173 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1174 = stablehlo.add %1172, %1173 : tensor<128xf32>
    %1175 = stablehlo.divide %1048, %1174 : tensor<128xf32>
    %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1176 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1177 = stablehlo.add %1093, %1176 : tensor<128xf32>
    %1178 = stablehlo.sqrt %1177 : tensor<128xf32>
    %cst_307 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1179 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1180 = stablehlo.add %1178, %1179 : tensor<128xf32>
    %1181 = stablehlo.divide %1050, %1180 : tensor<128xf32>
    %cst_308 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1182 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1183 = stablehlo.add %1095, %1182 : tensor<128xf32>
    %1184 = stablehlo.sqrt %1183 : tensor<128xf32>
    %cst_309 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1185 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1186 = stablehlo.add %1184, %1185 : tensor<128xf32>
    %1187 = stablehlo.divide %1052, %1186 : tensor<128xf32>
    %cst_310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1188 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1189 = stablehlo.add %1097, %1188 : tensor<512x128xf32>
    %1190 = stablehlo.sqrt %1189 : tensor<512x128xf32>
    %cst_311 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1191 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1192 = stablehlo.add %1190, %1191 : tensor<512x128xf32>
    %1193 = stablehlo.divide %1054, %1192 : tensor<512x128xf32>
    %cst_312 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1194 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1195 = stablehlo.add %1099, %1194 : tensor<128x128xf32>
    %1196 = stablehlo.sqrt %1195 : tensor<128x128xf32>
    %cst_313 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1197 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1198 = stablehlo.add %1196, %1197 : tensor<128x128xf32>
    %1199 = stablehlo.divide %1056, %1198 : tensor<128x128xf32>
    %cst_314 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1200 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1201 = stablehlo.add %1101, %1200 : tensor<128x384xf32>
    %1202 = stablehlo.sqrt %1201 : tensor<128x384xf32>
    %cst_315 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1203 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1204 = stablehlo.add %1202, %1203 : tensor<128x384xf32>
    %1205 = stablehlo.divide %1058, %1204 : tensor<128x384xf32>
    %cst_316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1206 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1207 = stablehlo.add %1103, %1206 : tensor<128xf32>
    %1208 = stablehlo.sqrt %1207 : tensor<128xf32>
    %cst_317 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1209 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1210 = stablehlo.add %1208, %1209 : tensor<128xf32>
    %1211 = stablehlo.divide %1060, %1210 : tensor<128xf32>
    %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1212 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1213 = stablehlo.add %1105, %1212 : tensor<128xf32>
    %1214 = stablehlo.sqrt %1213 : tensor<128xf32>
    %cst_319 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1215 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1216 = stablehlo.add %1214, %1215 : tensor<128xf32>
    %1217 = stablehlo.divide %1062, %1216 : tensor<128xf32>
    %cst_320 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1218 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1219 = stablehlo.add %1107, %1218 : tensor<10x128xf32>
    %1220 = stablehlo.sqrt %1219 : tensor<10x128xf32>
    %cst_321 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1221 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1222 = stablehlo.add %1220, %1221 : tensor<10x128xf32>
    %1223 = stablehlo.divide %1064, %1222 : tensor<10x128xf32>
    %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1224 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1225 = stablehlo.add %1109, %1224 : tensor<15x128xf32>
    %1226 = stablehlo.sqrt %1225 : tensor<15x128xf32>
    %cst_323 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1227 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1228 = stablehlo.add %1226, %1227 : tensor<15x128xf32>
    %1229 = stablehlo.divide %1066, %1228 : tensor<15x128xf32>
    %cst_324 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1230 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1231 = stablehlo.multiply %1230, %arg0 : tensor<128x512xf32>
    %1232 = stablehlo.add %1115, %1231 : tensor<128x512xf32>
    %cst_325 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1233 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1234 = stablehlo.multiply %1233, %arg1 : tensor<128xf32>
    %1235 = stablehlo.add %1121, %1234 : tensor<128xf32>
    %cst_326 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1236 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1237 = stablehlo.multiply %1236, %arg2 : tensor<128xf32>
    %1238 = stablehlo.add %1127, %1237 : tensor<128xf32>
    %cst_327 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1239 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1240 = stablehlo.multiply %1239, %arg3 : tensor<128xf32>
    %1241 = stablehlo.add %1133, %1240 : tensor<128xf32>
    %cst_328 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1242 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1243 = stablehlo.multiply %1242, %arg4 : tensor<128xf32>
    %1244 = stablehlo.add %1139, %1243 : tensor<128xf32>
    %cst_329 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1245 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1246 = stablehlo.multiply %1245, %arg5 : tensor<512x128xf32>
    %1247 = stablehlo.add %1145, %1246 : tensor<512x128xf32>
    %cst_330 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1248 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1249 = stablehlo.multiply %1248, %arg6 : tensor<128x128xf32>
    %1250 = stablehlo.add %1151, %1249 : tensor<128x128xf32>
    %cst_331 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1251 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1252 = stablehlo.multiply %1251, %arg7 : tensor<128x384xf32>
    %1253 = stablehlo.add %1157, %1252 : tensor<128x384xf32>
    %cst_332 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1254 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1255 = stablehlo.multiply %1254, %arg8 : tensor<128x512xf32>
    %1256 = stablehlo.add %1163, %1255 : tensor<128x512xf32>
    %cst_333 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1257 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1258 = stablehlo.multiply %1257, %arg9 : tensor<128xf32>
    %1259 = stablehlo.add %1169, %1258 : tensor<128xf32>
    %cst_334 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1260 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1261 = stablehlo.multiply %1260, %arg10 : tensor<128xf32>
    %1262 = stablehlo.add %1175, %1261 : tensor<128xf32>
    %cst_335 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1263 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1264 = stablehlo.multiply %1263, %arg11 : tensor<128xf32>
    %1265 = stablehlo.add %1181, %1264 : tensor<128xf32>
    %cst_336 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1266 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1267 = stablehlo.multiply %1266, %arg12 : tensor<128xf32>
    %1268 = stablehlo.add %1187, %1267 : tensor<128xf32>
    %cst_337 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1269 = stablehlo.broadcast_in_dim %cst_337, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1270 = stablehlo.multiply %1269, %arg13 : tensor<512x128xf32>
    %1271 = stablehlo.add %1193, %1270 : tensor<512x128xf32>
    %cst_338 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1272 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1273 = stablehlo.multiply %1272, %arg14 : tensor<128x128xf32>
    %1274 = stablehlo.add %1199, %1273 : tensor<128x128xf32>
    %cst_339 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1275 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1276 = stablehlo.multiply %1275, %arg15 : tensor<128x384xf32>
    %1277 = stablehlo.add %1205, %1276 : tensor<128x384xf32>
    %cst_340 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1278 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1279 = stablehlo.multiply %1278, %arg16 : tensor<128xf32>
    %1280 = stablehlo.add %1211, %1279 : tensor<128xf32>
    %cst_341 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1281 = stablehlo.broadcast_in_dim %cst_341, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1282 = stablehlo.multiply %1281, %arg17 : tensor<128xf32>
    %1283 = stablehlo.add %1217, %1282 : tensor<128xf32>
    %cst_342 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1284 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1285 = stablehlo.multiply %1284, %arg18 : tensor<10x128xf32>
    %1286 = stablehlo.add %1223, %1285 : tensor<10x128xf32>
    %cst_343 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %1287 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1288 = stablehlo.multiply %1287, %arg19 : tensor<15x128xf32>
    %1289 = stablehlo.add %1229, %1288 : tensor<15x128xf32>
    %c_344 = stablehlo.constant dense<0> : tensor<i32>
    %1290 = stablehlo.subtract %arg61, %c_344 : tensor<i32>
    %c_345 = stablehlo.constant dense<0> : tensor<i32>
    %c_346 = stablehlo.constant dense<75> : tensor<i32>
    %1291 = call @clip(%1290, %c_345, %c_346) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1292 = stablehlo.convert %1291 : (tensor<i32>) -> tensor<f32>
    %cst_347 = stablehlo.constant dense<7.500000e+01> : tensor<f32>
    %1293 = stablehlo.divide %1292, %cst_347 : tensor<f32>
    %cst_348 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1294 = stablehlo.subtract %cst_348, %1293 : tensor<f32>
    %cst_349 = stablehlo.constant dense<-3.000000e-03> : tensor<f32>
    %1295 = stablehlo.multiply %cst_349, %1294 : tensor<f32>
    %cst_350 = stablehlo.constant dense<3.000000e-03> : tensor<f32>
    %1296 = stablehlo.add %1295, %cst_350 : tensor<f32>
    %c_351 = stablehlo.constant dense<75> : tensor<i32>
    %1297 = stablehlo.compare LT, %arg61, %c_351, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_352 = stablehlo.constant dense<75> : tensor<i32>
    %1298 = stablehlo.subtract %arg61, %c_352 : tensor<i32>
    %1299 = stablehlo.convert %1298 : (tensor<i32>) -> tensor<f32>
    %cst_353 = stablehlo.constant dense<1.425000e+03> : tensor<f32>
    %1300 = stablehlo.minimum %1299, %cst_353 : tensor<f32>
    %cst_354 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %1301 = stablehlo.multiply %cst_354, %1300 : tensor<f32>
    %cst_355 = stablehlo.constant dense<1.425000e+03> : tensor<f32>
    %1302 = stablehlo.divide %1301, %cst_355 : tensor<f32>
    %1303 = stablehlo.cosine %1302 : tensor<f32>
    %cst_356 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1304 = stablehlo.add %cst_356, %1303 : tensor<f32>
    %cst_357 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1305 = stablehlo.multiply %cst_357, %1304 : tensor<f32>
    %cst_358 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1306 = stablehlo.power %1305, %cst_358 : tensor<f32>
    %cst_359 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %1307 = stablehlo.multiply %cst_359, %1306 : tensor<f32>
    %cst_360 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %1308 = stablehlo.add %1307, %cst_360 : tensor<f32>
    %cst_361 = stablehlo.constant dense<3.000000e-03> : tensor<f32>
    %1309 = stablehlo.multiply %cst_361, %1308 : tensor<f32>
    %1310 = call @_where_5(%1297, %1296, %1309) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %cst_362 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %1311 = stablehlo.multiply %cst_362, %1310 : tensor<f32>
    %1312 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1313 = stablehlo.multiply %1312, %1232 : tensor<128x512xf32>
    %1314 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1315 = stablehlo.multiply %1314, %1235 : tensor<128xf32>
    %1316 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1317 = stablehlo.multiply %1316, %1238 : tensor<128xf32>
    %1318 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1319 = stablehlo.multiply %1318, %1241 : tensor<128xf32>
    %1320 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1321 = stablehlo.multiply %1320, %1244 : tensor<128xf32>
    %1322 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1323 = stablehlo.multiply %1322, %1247 : tensor<512x128xf32>
    %1324 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1325 = stablehlo.multiply %1324, %1250 : tensor<128x128xf32>
    %1326 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1327 = stablehlo.multiply %1326, %1253 : tensor<128x384xf32>
    %1328 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %1329 = stablehlo.multiply %1328, %1256 : tensor<128x512xf32>
    %1330 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1331 = stablehlo.multiply %1330, %1259 : tensor<128xf32>
    %1332 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1333 = stablehlo.multiply %1332, %1262 : tensor<128xf32>
    %1334 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1335 = stablehlo.multiply %1334, %1265 : tensor<128xf32>
    %1336 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1337 = stablehlo.multiply %1336, %1268 : tensor<128xf32>
    %1338 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<512x128xf32>
    %1339 = stablehlo.multiply %1338, %1271 : tensor<512x128xf32>
    %1340 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %1341 = stablehlo.multiply %1340, %1274 : tensor<128x128xf32>
    %1342 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128x384xf32>
    %1343 = stablehlo.multiply %1342, %1277 : tensor<128x384xf32>
    %1344 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1345 = stablehlo.multiply %1344, %1280 : tensor<128xf32>
    %1346 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1347 = stablehlo.multiply %1346, %1283 : tensor<128xf32>
    %1348 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<10x128xf32>
    %1349 = stablehlo.multiply %1348, %1286 : tensor<10x128xf32>
    %1350 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<15x128xf32>
    %1351 = stablehlo.multiply %1350, %1289 : tensor<15x128xf32>
    %c_363 = stablehlo.constant dense<2147483647> : tensor<i32>
    %1352 = stablehlo.compare LT, %arg61, %c_363, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_364 = stablehlo.constant dense<1> : tensor<i32>
    %1353 = stablehlo.add %arg61, %c_364 : tensor<i32>
    %c_365 = stablehlo.constant dense<2147483647> : tensor<i32>
    %1354 = call @_where_4(%1352, %1353, %c_365) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1355 = stablehlo.add %arg0, %1313 : tensor<128x512xf32>
    %1356 = stablehlo.add %arg1, %1315 : tensor<128xf32>
    %1357 = stablehlo.add %arg2, %1317 : tensor<128xf32>
    %1358 = stablehlo.add %arg3, %1319 : tensor<128xf32>
    %1359 = stablehlo.add %arg4, %1321 : tensor<128xf32>
    %1360 = stablehlo.add %arg5, %1323 : tensor<512x128xf32>
    %1361 = stablehlo.add %arg6, %1325 : tensor<128x128xf32>
    %1362 = stablehlo.add %arg7, %1327 : tensor<128x384xf32>
    %1363 = stablehlo.add %arg8, %1329 : tensor<128x512xf32>
    %1364 = stablehlo.add %arg9, %1331 : tensor<128xf32>
    %1365 = stablehlo.add %arg10, %1333 : tensor<128xf32>
    %1366 = stablehlo.add %arg11, %1335 : tensor<128xf32>
    %1367 = stablehlo.add %arg12, %1337 : tensor<128xf32>
    %1368 = stablehlo.add %arg13, %1339 : tensor<512x128xf32>
    %1369 = stablehlo.add %arg14, %1341 : tensor<128x128xf32>
    %1370 = stablehlo.add %arg15, %1343 : tensor<128x384xf32>
    %1371 = stablehlo.add %arg16, %1345 : tensor<128xf32>
    %1372 = stablehlo.add %arg17, %1347 : tensor<128xf32>
    %1373 = stablehlo.add %arg18, %1349 : tensor<10x128xf32>
    %1374 = stablehlo.add %arg19, %1351 : tensor<15x128xf32>
    return %1355, %1356, %1357, %1358, %1359, %1360, %1361, %1362, %1363, %1364, %1365, %1366, %1367, %1368, %1369, %1370, %1371, %1372, %1373, %1374, %1023, %805, %810, %815, %820, %825, %830, %835, %840, %845, %850, %855, %860, %865, %870, %875, %880, %885, %890, %895, %900, %906, %912, %918, %924, %930, %936, %942, %948, %954, %960, %966, %972, %978, %984, %990, %996, %1002, %1008, %1014, %1020, %1354, %282 : tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<128x384xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<128x384xf32>, tensor<128xf32>, tensor<128xf32>, tensor<10x128xf32>, tensor<15x128xf32>, tensor<i32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<128x384xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<128x384xf32>, tensor<128xf32>, tensor<128xf32>, tensor<10x128xf32>, tensor<15x128xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<128x384xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<128x128xf32>, tensor<128x384xf32>, tensor<128xf32>, tensor<128xf32>, tensor<10x128xf32>, tensor<15x128xf32>, tensor<i32>, tensor<f32>
  }
  func.func private @tril(%arg0: tensor<9x9xi1>) -> tensor<9x9xi1> {
    %0 = stablehlo.iota dim = 0 : tensor<9x9xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<9x9xi32>
    %2 = stablehlo.add %0, %1 : tensor<9x9xi32>
    %3 = stablehlo.iota dim = 1 : tensor<9x9xi32>
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<9x9xi32>, tensor<9x9xi32>) -> tensor<9x9xi1>
    %c_0 = stablehlo.constant dense<false> : tensor<i1>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i1>) -> tensor<9x9xi1>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<9x9xi1>, tensor<9x9xi1>
    return %6 : tensor<9x9xi1>
  }
  func.func private @_where(%arg0: tensor<9x9xi1>, %arg1: tensor<128x4x9x9xf32>, %arg2: tensor<f32>) -> (tensor<128x4x9x9xf32>, tensor<128x4x9x9xi1>) {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [2, 3] : (tensor<9x9xi1>) -> tensor<128x4x9x9xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<128x4x9x9xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<128x4x9x9xi1>, tensor<128x4x9x9xf32>
    return %3, %1 : tensor<128x4x9x9xf32>, tensor<128x4x9x9xi1>
  }
  func.func private @log_softmax(%arg0: tensor<128x9x15xf32>) -> (tensor<128x9x15xf32>, tensor<128x9x15xf32>, tensor<128x9x1xf32>) {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x9x15xf32>, tensor<f32>) -> tensor<128x9xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x9xf32>
    %2 = stablehlo.maximum %1, %0 : tensor<128x9xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x15xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<128x9x15xf32>
    %6 = stablehlo.exponential %5 : tensor<128x9x15xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<128x9x15xf32>, tensor<f32>) -> tensor<128x9xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %9 = stablehlo.log %8 : tensor<128x9x1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<128x9x1xf32>) -> tensor<128x9x15xf32>
    %11 = stablehlo.subtract %5, %10 : tensor<128x9x15xf32>
    return %11, %6, %8 : tensor<128x9x15xf32>, tensor<128x9x15xf32>, tensor<128x9x1xf32>
  }
  func.func private @take_along_axis(%arg0: tensor<128x9x15xf32>, %arg1: tensor<128x9x1xi32>) -> (tensor<128x9x1xf32>, tensor<128x9x1x1xi32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x9x1xi32>
    %1 = stablehlo.compare LT, %arg1, %0, SIGNED : (tensor<128x9x1xi32>, tensor<128x9x1xi32>) -> tensor<128x9x1xi1>
    %c_0 = stablehlo.constant dense<15> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128x9x1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<128x9x1xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<128x9x1xi1>, tensor<128x9x1xi32>
    %5 = stablehlo.reshape %4 : (tensor<128x9x1xi32>) -> tensor<128x9x1x1xi32>
    %c_1 = stablehlo.constant dense<14> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x9x1x1xi32>
    %7 = stablehlo.compare GE, %5, %6, SIGNED : (tensor<128x9x1x1xi32>, tensor<128x9x1x1xi32>) -> tensor<128x9x1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [3] : (tensor<1xi32>) -> tensor<1x1x1x1xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi32>) -> tensor<128x9x1x1xi32>
    %10 = stablehlo.compare LE, %5, %9, SIGNED : (tensor<128x9x1x1xi32>, tensor<128x9x1x1xi32>) -> tensor<128x9x1x1xi1>
    %11 = stablehlo.and %7, %10 : tensor<128x9x1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.reduce(%11 init: %c_3) applies stablehlo.and across dimensions = [3] : (tensor<128x9x1x1xi1>, tensor<i1>) -> tensor<128x9x1xi1>
    %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<128x9x15xf32>, tensor<128x9x1x1xi32>) -> tensor<128x9x1xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x9x1xf32>
    %15 = stablehlo.select %12, %13, %14 : tensor<128x9x1xi1>, tensor<128x9x1xf32>
    return %15, %5 : tensor<128x9x1xf32>, tensor<128x9x1x1xi32>
  }
  func.func private @take_along_axis_0(%arg0: tensor<128x9x1x1xi32>, %arg1: tensor<128x9x1xf32>) -> tensor<128x9x15xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x9x15xf32>
    %1 = "stablehlo.scatter"(%0, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<128x9x15xf32>, tensor<128x9x1x1xi32>, tensor<128x9x1xf32>) -> tensor<128x9x15xf32>
    return %1 : tensor<128x9x15xf32>
  }
  func.func private @log_softmax_1(%arg0: tensor<128x9x15xf32>, %arg1: tensor<128x9x1xf32>, %arg2: tensor<128x9x15xf32>) -> tensor<128x9x15xf32> {
    %0 = stablehlo.negate %arg2 : tensor<128x9x15xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x9x15xf32>, tensor<f32>) -> tensor<128x9xf32>
    %2 = stablehlo.reshape %1 : (tensor<128x9xf32>) -> tensor<128x9x1xf32>
    %3 = stablehlo.divide %2, %arg1 : tensor<128x9x1xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<128x9x1xf32>, tensor<f32>) -> tensor<128x9xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<128x9xf32>) -> tensor<128x9x15xf32>
    %6 = stablehlo.multiply %5, %arg0 : tensor<128x9x15xf32>
    %7 = stablehlo.add %arg2, %6 : tensor<128x9x15xf32>
    return %7 : tensor<128x9x15xf32>
  }
  func.func private @_where_2(%arg0: tensor<128x4x9x9xi1>, %arg1: tensor<128x4x9x9xf32>) -> tensor<128x4x9x9xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4x9x9xf32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<128x4x9x9xi1>, tensor<128x4x9x9xf32>
    return %1 : tensor<128x4x9x9xf32>
  }
  func.func private @_where_3(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_4(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @_where_5(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
}
