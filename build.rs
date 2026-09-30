// With the `tensorrt` feature, compile the C++ shim over TensorRT and link it, CUDA and the ONNX parser.
// CUDA_HOME (default /usr/local/cuda) and TENSORRT_ROOT (default: the system paths JetPack uses) locate them.

fn main() {
    println!("cargo:rerun-if-changed=src/tensorrt.cpp");
    if std::env::var_os("CARGO_FEATURE_TENSORRT").is_none() {
        return;
    }
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").file("src/tensorrt.cpp").include(format!("{cuda}/include")).warnings(false);
    if let Ok(root) = std::env::var("TENSORRT_ROOT") {
        build.include(format!("{root}/include"));
        println!("cargo:rustc-link-search=native={root}/lib");
    }
    build.compile("d2d_tensorrt");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    for library in ["nvinfer", "nvonnxparser", "cudart"] {
        println!("cargo:rustc-link-lib={library}");
    }
    println!("cargo:rustc-link-lib=stdc++");
}
