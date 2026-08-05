/*
 * GEMM gRPC server — backed by TensorFlow C++ Session / MatMul op kernel.
 *
 * The graph is built once at startup with dynamic-shape Placeholders,
 * then reused for every call (mutex-protected).
 *
 * Build:
 *   /home/wanglimin/bazel-7.4.1 build -c opt <common-flags> \
 *     //tensorflow/tools/gemm_server:gemm_server
 *
 * Run:
 *   ./bazel-bin/tensorflow/tools/gemm_server/gemm_server [--addr=0.0.0.0:50052]
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// TF C++ API
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"   // generated: MatMul, Placeholder
#include "tensorflow/core/framework/tensor.h"

// gRPC + generated stubs
#include "grpcpp/grpcpp.h"
#include "tensorflow/tools/gemm_server/proto/gemm.grpc.pb.h"
#include "tensorflow/tools/gemm_server/proto/gemm.pb.h"

namespace tf    = tensorflow;
namespace tfops = tensorflow::ops;
using Clock = std::chrono::steady_clock;

// ── helpers ───────────────────────────────────────────────────────────────────

struct Stats { double avg, p50, p99; };

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    return {s / v.size(), v[v.size() / 2], v[v.size() * 99 / 100]};
}

static tf::Tensor make_random(int rows, int cols) {
    tf::Tensor t(tf::DT_FLOAT, tf::TensorShape({rows, cols}));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;
    auto flat = t.flat<float>();
    for (int i = 0; i < flat.size(); ++i) flat.data()[i] = dist(rng);
    return t;
}

// ── GEMMRunner ────────────────────────────────────────────────────────────────

// Wraps a single TF graph: Placeholder(A) -> MatMul -> Placeholder(B) -> C.
// Dynamic shapes: accepts any [M,K] x [K,N] without rebuilding the graph.
class GEMMRunner {
public:
    GEMMRunner() : scope_(tf::Scope::NewRootScope()) {
        // Build graph once.  A_ph_, B_ph_, C_op_ are tf::Output (implicitly
        // converted from ops::Placeholder / ops::MatMul via operator Output()).
        auto ph_a = tfops::Placeholder(scope_.WithOpName("A"), tf::DT_FLOAT);
        auto ph_b = tfops::Placeholder(scope_.WithOpName("B"), tf::DT_FLOAT);
        auto mm   = tfops::MatMul(scope_.WithOpName("C"), ph_a, ph_b);
        TF_CHECK_OK(scope_.status());
        A_ph_ = ph_a;  // implicit Output conversion
        B_ph_ = ph_b;
        C_op_ = mm;
        session_ = std::make_unique<tf::ClientSession>(scope_);
    }

    // C = MatMul(A, B) via TF kernel. Returns compute-only time (ms).
    double Run(const tf::Tensor& A, const tf::Tensor& B, tf::Tensor* C_out) {
        std::vector<tf::Tensor> outputs;
        std::lock_guard<std::mutex> lk(mu_);
        auto t0 = Clock::now();
        TF_CHECK_OK(session_->Run({{A_ph_, A}, {B_ph_, B}}, {C_op_}, &outputs));
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
        *C_out = std::move(outputs[0]);
        return ms;
    }

private:
    tf::Scope  scope_;
    tf::Output A_ph_, B_ph_, C_op_;  // Output is default-constructible
    std::unique_ptr<tf::ClientSession> session_;
    std::mutex mu_;
};

// ── gRPC service ──────────────────────────────────────────────────────────────

class GEMMServiceImpl final : public gemm::GEMMService::Service {
public:
    // Single call: client supplies matrix data, receives result + compute time.
    grpc::Status Compute(grpc::ServerContext*,
                          const gemm::ComputeRequest* req,
                          gemm::ComputeResponse* resp) override {
        int M = req->m(), K = req->k(), N = req->n();

        if (req->a_data_size() != M * K || req->b_data_size() != K * N)
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "a_data / b_data size mismatch with M, K, N"};

        tf::Tensor A(tf::DT_FLOAT, tf::TensorShape({M, K}));
        tf::Tensor B(tf::DT_FLOAT, tf::TensorShape({K, N}));
        std::copy(req->a_data().begin(), req->a_data().end(),
                  A.flat<float>().data());
        std::copy(req->b_data().begin(), req->b_data().end(),
                  B.flat<float>().data());

        tf::Tensor C;
        double ms = runner_.Run(A, B, &C);

        auto flat = C.flat<float>();
        resp->mutable_c_data()->Assign(flat.data(), flat.data() + M * N);
        resp->set_server_compute_ms(ms);
        return grpc::Status::OK;
    }

    // Sweep: server generates random matrices, benchmarks each size with TF.
    grpc::Status Sweep(grpc::ServerContext*,
                        const gemm::SweepRequest* req,
                        gemm::SweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 50;
        int warmup = req->warmup() > 0 ? req->warmup() : 10;

        for (int sz : req->sizes()) {
            tf::Tensor A = make_random(sz, sz);
            tf::Tensor B = make_random(sz, sz);

            for (int i = 0; i < warmup; ++i) {
                tf::Tensor C;
                runner_.Run(A, B, &C);
            }

            std::vector<double> lats(iters);
            for (int i = 0; i < iters; ++i) {
                tf::Tensor C;
                lats[i] = runner_.Run(A, B, &C);
            }

            auto st     = compute_stats(lats);
            double fl   = 2.0 * sz * sz * sz;
            double gfl  = fl / (st.avg / 1e3) / 1e9;

            std::cout << "[sweep] " << sz << "x" << sz
                      << "  avg=" << st.avg << " ms"
                      << "  GFLOPS=" << gfl
                      << " (TF MatMul kernel)\n" << std::flush;

            auto* r = resp->add_results();
            r->set_m(sz);  r->set_k(sz);  r->set_n(sz);
            r->set_avg_ms(st.avg);
            r->set_p50_ms(st.p50);
            r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }

private:
    GEMMRunner runner_;
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string addr = "0.0.0.0:50052";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--addr=", 0) == 0) addr = a.substr(7);
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    GEMMServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(256 << 20);
    builder.SetMaxSendMessageSize(256 << 20);

    auto server = builder.BuildAndStart();
    std::cout << "TF GEMM server on " << addr
              << "  (backend: TF C++ MatMul kernel)\n";
    server->Wait();
    return 0;
}
