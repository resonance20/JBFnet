#include "C:\Users\z003zv1a\Downloads\libtorch\include\torch\extension.h"

at::Tensor nlmeans_forward(
    torch::Tensor input,
    int p,
    int s
){
    int or_p = p;
    p *= 2;
    p++;
    int w = s;
    s *= 2;
    
    auto h = torch::ones({1, 1, p, p, p}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0));
    auto c = 0.5 * torch::sum(h);

    auto x_stat = torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({w, w, w, w, w, w}).
        mode(torch::kReplicate));
    auto z_stat = torch::zeros_like(x_stat, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0));

    auto w_sum = torch::zeros_like(z_stat);

    for(int i = 0; i < (s/2); ++i){
        for(int j = 0; j < (s/2); ++j){
            for(int k = 0; k < (s/2); ++k){

                auto x_disp = torch::nn::functional::pad(input, 
                    torch::nn::functional::PadFuncOptions({k, s - k, j, s - j, i, s - i}).mode(torch::kReplicate));

                auto u_n = torch::square(torch::sub(x_stat, x_disp));

                auto v_n = torch::nn::functional::conv3d(u_n, h, 
                    torch::nn::functional::Conv3dFuncOptions().stride(1).padding({or_p, or_p, or_p}));

                std::vector<float> dist_math{(float)(pow((w - i), 2) + pow((w - j), 2) + pow((w - k), 2))/2};
                auto dist = torch::from_blob(dist_math.data(), {1}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(torch::kCUDA, 0);

                auto w_n = torch::exp( torch::add(- dist, torch::div(v_n, c)) );

                using namespace torch::indexing;
                z_stat = torch::add(z_stat, torch::mul(x_disp, w_n));
                w_sum = torch::add(w_sum, w_n);
            }
        }
    }

    z_stat = torch::div(z_stat, w_sum);
    using namespace torch::indexing;
    return z_stat.index({Slice(), Slice(), Slice(w, -w), Slice(w, -w), Slice(w, -w)});

}

torch::Tensor nlmeans_backward(
    torch::Tensor gradient
){
    return gradient;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nlmeans_forward, "NLmeans forward");
  m.def("backward", &nlmeans_backward, "NLmeans backward");
}