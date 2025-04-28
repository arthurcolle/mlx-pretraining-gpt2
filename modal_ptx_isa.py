import modal
import cupy as cp

# Modal setup
stub = modal.Stub()

# Define CUDA kernel for simple forward pass
MLP_KERNEL = r'''
extern "C" __global__ void mlp_forward(
    const float* x, const float* w1, const float* b1,
    const float* w2, const float* b2,
    float* out, int input_dim, int hidden_dim, int output_dim
) {
    int idx = threadIdx.x;
    if (idx >= output_dim) return;

    // Hidden layer computation
    float hidden_val = 0.0f;
    for (int i = 0; i < input_dim; ++i) {
        hidden_val += x[i] * w1[i * hidden_dim + idx];
    }
    hidden_val += b1[idx];
    hidden_val = fmaxf(0.0f, hidden_val); // ReLU

    // Output layer computation
    float output_val = 0.0f;
    for (int j = 0; j < hidden_dim; ++j) {
        output_val += hidden_val * w2[j * output_dim + idx];
    }
    output_val += b2[idx];

    out[idx] = output_val;
}
'''

@stub.function(gpu="any")
def run_mlp():
    # Dimensions
    input_dim = 4
    hidden_dim = 8
    output_dim = 2

    # Inputs and weights
    x = cp.random.randn(input_dim).astype(cp.float32)
    w1 = cp.random.randn(input_dim, hidden_dim).astype(cp.float32)
    b1 = cp.random.randn(hidden_dim).astype(cp.float32)
    w2 = cp.random.randn(hidden_dim, output_dim).astype(cp.float32)
    b2 = cp.random.randn(output_dim).astype(cp.float32)

    # Allocate output
    out = cp.zeros(output_dim, dtype=cp.float32)

    # Compile the CUDA kernel
    module = cp.RawModule(code=MLP_KERNEL)
    kernel = module.get_function("mlp_forward")

    # Launch the kernel
    kernel((1,), (output_dim,),
           (x, w1, b1, w2, b2, out,
            cp.int32(input_dim), cp.int32(hidden_dim), cp.int32(output_dim)))

    # Fetch result
    print("Input:", x)
    print("Output:", out)

if __name__ == "__main__":
    with stub.run():
        run_mlp.remote()

