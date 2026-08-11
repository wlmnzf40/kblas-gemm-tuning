/*
 * GEMM gRPC server — direct BLAS / Eigen dispatch, no TF Session in hot path.
 *
 * Flags:
 *   --addr=host:port     listen address           (default 0.0.0.0:50052)
 *   --backend=kblas      cblas_sgemm from libkblas.so  (requires --config=kml_kblas build)
 *   --backend=eigen      Eigen matrix multiply          (always available)
 *
 * Build:
 *   --config=kml_kblas   links libkblas.so; both --backend values work; default=kblas
 *   (no config)          Eigen only; --backend=kblas prints an error; default=eigen
 *
 * Examples:
 *   ./gemm_server --backend=kblas               # KBLAS, port 50052
 *   ./gemm_server --backend=eigen --addr=:50053 # Eigen, port 50053
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "grpcpp/grpcpp.h"
#include "tf_serving_gemm/tf_gemm_server/proto/gemm.grpc.pb.h"
#include "tf_serving_gemm/tf_gemm_server/proto/gemm.pb.h"

// ── cblas forward declarations (inlined — no -I flag needed) ─────────────────
#ifdef TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
extern "C" {
enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans  = 111, CblasTrans    = 112,
                       CblasConjTrans = 113 };
void cblas_sgemm(CBLAS_ORDER Order,
                 CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                 int M, int N, int K,
                 float alpha, const float* A, int lda,
                              const float* B, int ldb,
                 float beta,        float* C, int ldc);
}
#endif

using Clock = std::chrono::steady_clock;
using MatRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ── runtime backend selection ─────────────────────────────────────────────────

enum class Backend { kKBLAS, kEigen };
static Backend g_backend;

// ── helpers ───────────────────────────────────────────────────────────────────

struct Stats { double avg, p50, p99; };

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    return {s / v.size(), v[v.size() / 2], v[v.size() * 99 / 100]};
}

static void fill_random(float* data, int n) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
}

// ── core dispatch: C[M,N] = A[M,K] × B[K,N]  (row-major, fp32) ──────────────

static double direct_sgemm(const float* A, const float* B, float* C,
                            int M, int K, int N) {
    auto t0 = Clock::now();
#ifdef TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
    if (g_backend == Backend::kKBLAS) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    } else
#endif
    {
        Eigen::Map<const MatRM> eA(A, M, K);
        Eigen::Map<const MatRM> eB(B, K, N);
        Eigen::Map<MatRM>       eC(C, M, N);
        eC.noalias() = eA * eB;
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// C[b,M,N] = A[b,M,K] × B[b,K,N]
static double direct_batch_sgemm(const float* A, const float* B, float* C,
                                  int b, int M, int K, int N) {
    auto t0 = Clock::now();
    for (int i = 0; i < b; ++i) {
        const float* ai = A + i * M * K;
        const float* bi = B + i * K * N;
        float*       ci = C + i * M * N;
#ifdef TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
        if (g_backend == Backend::kKBLAS) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, ai, K, bi, N, 0.0f, ci, N);
        } else
#endif
        {
            Eigen::Map<const MatRM> eA(ai, M, K);
            Eigen::Map<const MatRM> eB(bi, K, N);
            Eigen::Map<MatRM>       eC(ci, M, N);
            eC.noalias() = eA * eB;
        }
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── gRPC service ──────────────────────────────────────────────────────────────

class GEMMServiceImpl final : public gemm::GEMMService::Service {
public:
    // ── Compute ───────────────────────────────────────────────────────────────
    grpc::Status Compute(grpc::ServerContext*,
                          const gemm::ComputeRequest* req,
                          gemm::ComputeResponse* resp) override {
        int M = req->m(), K = req->k(), N = req->n();
        if (req->a_data_size() != M * K || req->b_data_size() != K * N)
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "a_data / b_data size mismatch with M, K, N"};

        std::vector<float> C(M * N);
        double ms = direct_sgemm(req->a_data().data(), req->b_data().data(),
                                  C.data(), M, K, N);
        resp->mutable_c_data()->Assign(C.begin(), C.end());
        resp->set_server_compute_ms(ms);
        return grpc::Status::OK;
    }

    // ── Sweep (square sizes) ──────────────────────────────────────────────────
    grpc::Status Sweep(grpc::ServerContext*,
                        const gemm::SweepRequest* req,
                        gemm::SweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 50;
        int warmup = req->warmup() > 0 ? req->warmup() : 10;

        for (int sz : req->sizes()) {
            std::vector<float> A(sz*sz), B(sz*sz), C(sz*sz);
            fill_random(A.data(), sz*sz);
            fill_random(B.data(), sz*sz);

            for (int i = 0; i < warmup; ++i)
                direct_sgemm(A.data(), B.data(), C.data(), sz, sz, sz);

            std::vector<double> lats(iters);
            for (int i = 0; i < iters; ++i)
                lats[i] = direct_sgemm(A.data(), B.data(), C.data(), sz, sz, sz);

            auto   st  = compute_stats(lats);
            double gfl = 2.0 * sz * sz * sz / (st.avg / 1e3) / 1e9;
            std::cout << "[sweep] " << sz << "x" << sz
                      << "  avg=" << st.avg << " ms  GFLOPS=" << gfl << "\n"
                      << std::flush;

            auto* r = resp->add_results();
            r->set_m(sz); r->set_k(sz); r->set_n(sz);
            r->set_avg_ms(st.avg); r->set_p50_ms(st.p50); r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }

    // ── ShapeSweep (production shapes) ───────────────────────────────────────
    grpc::Status ShapeSweep(grpc::ServerContext*,
                              const gemm::ShapeSweepRequest* req,
                              gemm::ShapeSweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 100;
        int warmup = req->warmup() > 0 ? req->warmup() : 20;

        for (const auto& shape : req->shapes()) {
            int  b = shape.batch(), M = shape.m(), K = shape.k(), N = shape.n();
            bool batched = (b > 1);

            std::vector<float> A(b*M*K), B(b*K*N), C(b*M*N);
            fill_random(A.data(), b*M*K);
            fill_random(B.data(), b*K*N);

            for (int i = 0; i < warmup; ++i) {
                if (batched) direct_batch_sgemm(A.data(), B.data(), C.data(), b, M, K, N);
                else         direct_sgemm(A.data(), B.data(), C.data(), M, K, N);
            }
            std::vector<double> lats(iters);
            for (int i = 0; i < iters; ++i) {
                lats[i] = batched
                    ? direct_batch_sgemm(A.data(), B.data(), C.data(), b, M, K, N)
                    : direct_sgemm(A.data(), B.data(), C.data(), M, K, N);
            }

            auto   st  = compute_stats(lats);
            double gfl = 2.0 * b * M * K * N / (st.avg / 1e3) / 1e9;
            std::cout << "[shape] " << shape.model()
                      << " b=" << b << " [" << M << "x" << K << "x" << N << "]"
                      << "  avg=" << st.avg << " ms  GFLOPS=" << gfl << "\n"
                      << std::flush;

            auto* r = resp->add_results();
            *r->mutable_shape() = shape;
            r->set_avg_ms(st.avg);
            r->set_p50_ms(st.p50);
            r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string addr = "0.0.0.0:50052";
#ifdef TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
    std::string backend = "kblas";
#else
    std::string backend = "eigen";
#endif

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--addr=",    0) == 0) addr    = a.substr(7);
        else if (a.rfind("--backend=", 0) == 0) backend = a.substr(10);
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    if (backend == "kblas") {
#ifndef TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
        std::cerr << "ERROR: --backend=kblas requires building with --config=kml_kblas\n";
        return 1;
#endif
        g_backend = Backend::kKBLAS;
    } else if (backend == "eigen") {
        g_backend = Backend::kEigen;
    } else {
        std::cerr << "Unknown backend '" << backend << "'. Use kblas|eigen\n";
        return 1;
    }

    std::cout << "TF GEMM server  addr=" << addr
              << "  backend=" << backend << "\n";

    GEMMServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(256 << 20);
    builder.SetMaxSendMessageSize(256 << 20);

    auto server = builder.BuildAndStart();
    server->Wait();
    return 0;
}
