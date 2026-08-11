/*
 * GEMM gRPC client
 *
 * Modes:
 *   --mode=shape_sweep  benchmark the production shapes from the CSV (default)
 *   --mode=sweep        benchmark square sizes
 *   --mode=compute      single round-trip with client-supplied data
 *
 * Build:
 *   //tf_serving_gemm/tf_gemm_server:gemm_client
 *
 * Examples:
 *   ./gemm_client                                        # shape_sweep, default shapes
 *   ./gemm_client --mode=sweep --sizes=128,256,512,1024
 *   ./gemm_client --mode=compute --M=512 --K=512 --N=512
 *   ./gemm_client --host=localhost:50051 --mode=shape_sweep  # hit Eigen server
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "tf_serving_gemm/tf_gemm_server/proto/gemm.grpc.pb.h"
#include "tf_serving_gemm/tf_gemm_server/proto/gemm.pb.h"

using Clock = std::chrono::steady_clock;
using Stub  = gemm::GEMMService::Stub;

// ── production shapes from the profiled CSV ───────────────────────────────────
// Columns: model, batch, M, K, N, count
// batch=1  → plain MatMul  [M,K] × [K,N]
// batch>1  → BatchMatMul   [b,M,K] × [b,K,N]
// count    = how many times this GEMM appears per inference forward pass

struct ShapeSpec {
    const char* model;
    int batch, M, K, N, count;
};

static const std::vector<ShapeSpec> kProductionShapes = {
    // ── adx (batch=1) ─────────────────────────────────────────────────────────
    {"adx",  1,  35,  800,   1,  1},
    {"adx",  1,  35,  400, 400,  4},
    {"adx",  1,  39,    5,   1,  1},
    {"adx",  1,  39,  492,   1,  3},
    {"adx",  1,  39,  800,   1,  1},
    {"adx",  1,  39,  400, 400,  4},
    {"adx",  1,  46,    5,   1,  1},
    {"adx",  1,  46,  492,   1,  3},
    // ── cvr (batch=1) ─────────────────────────────────────────────────────────
    {"cvr",  1,   1,   16,  16,  1},
    {"cvr",  1,   1,   24,  16,  2},
    {"cvr",  1,  14,   28,  16,  2},
    {"cvr",  1,  16,    8,   4,  1},
    {"cvr",  1,  19,    8,   4,  1},
    {"cvr",  1,  48,   28,  16,  2},
    {"cvr",  1,  59,   64,   1,  4},
    {"cvr",  1,  59, 1316,   2,  5},
    {"cvr",  1,  59,   64,   6,  1},
    {"cvr",  1,  59,   48,  48,  2},
    {"cvr",  1,  59,  128,  64,  5},
    {"cvr",  1,  59, 1540,  64,  1},
    {"cvr",  1,  59,   48,  80,  2},
    {"cvr",  1,  59,  256, 128,  3},
    {"cvr",  1,  59, 1316, 256,  3},
    {"cvr",  1,  77,   20,  16,  2},
    {"cvr",  1,  80,   36,  16,  2},
    // ── cvr (batch=50 / 59) ───────────────────────────────────────────────────
    {"cvr", 50,   1,   32,  16,  1},
    {"cvr", 50,  32,    8,   4,  1},
    {"cvr", 59,   1,    2, 128,  5},
    // ── hmv (batch=1) ─────────────────────────────────────────────────────────
    {"hmv",  1,   59,    8,   1,  1},
    {"hmv",  1,   59,   32,   2,  2},
    {"hmv",  1,   59,   64,   2,  2},
    {"hmv",  1,   59,   16,   8,  1},
    {"hmv",  1,   59,   30,  16,  1},
    {"hmv",  1,   59,   64,  32,  4},
    {"hmv",  1,   59,   32,  64,  2},
    {"hmv",  1,   59,  128,  64,  4},
    {"hmv",  1,   59,  256, 128,  3},
    {"hmv",  1,   59, 1570, 128,  2},
    {"hmv",  1,   59, 1570, 256,  3},
    {"hmv",  1, 4130,   16,   1,  2},
    {"hmv",  1, 4130,   32,  16,  2},
    {"hmv",  1, 4130,  104,  32,  1},
    {"hmv",  1, 4130,  116,  32,  1},
    // ── presort (batch=1) ─────────────────────────────────────────────────────
    {"presort",  1,  32,    8,   4,  1},
    {"presort",  1,  32,   16,   8,  3},
    {"presort",  1,  32,   32,   8,  1},
    {"presort",  1,  50,  100,   1,  2},
    {"presort",  1,  50,  250, 100,  2},
    {"presort",  1,  50,  250, 250,  2},
    {"presort",  1,  50, 1298, 250,  2},
    // ── presort (batch=50) ────────────────────────────────────────────────────
    {"presort", 50,   1,   32,  16,  2},
    {"presort", 50,  32,    4,   1,  2},
    {"presort", 50,  32,    8,   4,  1},
    {"presort", 50,  32,   16,   8,  2},
    {"presort", 50,  32,   32,   8,  1},
    {"presort", 50,  32,   32,  16,  1},
    {"presort", 50,  32,   64,  16,  2},
};

// ── helpers ───────────────────────────────────────────────────────────────────

static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
    return v;
}

struct Stats { double avg, p50, p99; };
static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0; for (double x : v) s += x;
    return {s / v.size(), v[v.size() / 2], v[v.size() * 99 / 100]};
}

// ── shape_sweep mode ──────────────────────────────────────────────────────────

static void run_shape_sweep(Stub& stub, int iters, int warmup) {
    gemm::ShapeSweepRequest req;
    req.set_iters(iters);
    req.set_warmup(warmup);
    for (const auto& s : kProductionShapes) {
        auto* g = req.add_shapes();
        g->set_model(s.model); g->set_batch(s.batch);
        g->set_m(s.M);         g->set_k(s.K);
        g->set_n(s.N);         g->set_count(s.count);
    }

    gemm::ShapeSweepResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));

    std::cout << "ShapeSweep: " << kProductionShapes.size() << " shapes, "
              << warmup << " warmup + " << iters << " iters each\n\n";

    if (auto st = stub.ShapeSweep(&ctx, req, &resp); !st.ok()) {
        std::cerr << "[ERROR] " << st.error_message() << "\n";
        return;
    }

    // Print grouped by (model, batch)
    const int W1=14, W2=5, W3=10, W4=9, W5=9, W6=9, W7=10;
    std::string cur_group;
    for (const auto& r : resp.results()) {
        const auto& s  = r.shape();
        std::string group = std::string(s.model()) +
                            (s.batch() > 1 ? " (batch=" + std::to_string(s.batch()) + ")" : "");
        if (group != cur_group) {
            cur_group = group;
            std::cout << "\n=== " << cur_group << " ===\n"
                      << std::left
                      << std::setw(W1) << "MxKxN"
                      << std::setw(W2) << "cnt"
                      << std::setw(W3) << "avg_ms"
                      << std::setw(W4) << "p50_ms"
                      << std::setw(W5) << "p99_ms"
                      << std::setw(W6) << "GFLOPS"
                      << "\n" << std::string(W1+W2+W3+W4+W5+W6, '-') << "\n";
        }
        std::string shape_str =
            std::to_string(s.m()) + "x" + std::to_string(s.k()) + "x" + std::to_string(s.n());
        std::cout << std::left  << std::setw(W1) << shape_str
                  << std::setw(W2) << s.count()
                  << std::fixed << std::setprecision(4)
                  << std::setw(W3) << r.avg_ms()
                  << std::setw(W4) << r.p50_ms()
                  << std::setw(W5) << r.p99_ms()
                  << std::setprecision(3)
                  << std::setw(W6) << r.gflops()
                  << "\n";
    }
    std::cout << "\n";
}

// ── sweep mode ────────────────────────────────────────────────────────────────

static void run_sweep(Stub& stub, const std::vector<int>& sizes,
                       int iters, int warmup) {
    gemm::SweepRequest req;
    for (int s : sizes) req.add_sizes(s);
    req.set_iters(iters); req.set_warmup(warmup);

    gemm::SweepResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));

    if (auto st = stub.Sweep(&ctx, req, &resp); !st.ok()) {
        std::cerr << "[ERROR] " << st.error_message() << "\n";
        return;
    }

    const int W = 12;
    std::cout << "\n"
              << std::left
              << std::setw(W) << "Size"
              << std::setw(W) << "avg_ms"
              << std::setw(W) << "p50_ms"
              << std::setw(W) << "p99_ms"
              << std::setw(W) << "GFLOPS"
              << "\n" << std::string(W * 5, '-') << "\n";
    for (const auto& r : resp.results()) {
        std::string sz = std::to_string(r.m()) + "x" + std::to_string(r.n());
        std::cout << std::left  << std::setw(W) << sz
                  << std::fixed << std::setprecision(3)
                  << std::setw(W) << r.avg_ms()
                  << std::setw(W) << r.p50_ms()
                  << std::setw(W) << r.p99_ms()
                  << std::setw(W) << r.gflops() << "\n";
    }
}

// ── compute mode ──────────────────────────────────────────────────────────────

static void run_compute(Stub& stub, int M, int K, int N, int iters, int warmup) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;
    gemm::ComputeRequest req;
    req.set_m(M); req.set_k(K); req.set_n(N);
    for (int i = 0; i < M * K; ++i) req.add_a_data(dist(rng));
    for (int i = 0; i < K * N; ++i) req.add_b_data(dist(rng));

    auto call = [&]() -> std::pair<double, double> {
        gemm::ComputeResponse resp;
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(1));
        auto t0 = Clock::now();
        auto st = stub.Compute(&ctx, req, &resp);
        double rtt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (!st.ok()) throw std::runtime_error(st.error_message());
        return {rtt, resp.server_compute_ms()};
    };

    for (int i = 0; i < warmup; ++i) call();

    std::vector<double> rtt_v(iters), srv_v(iters);
    for (int i = 0; i < iters; ++i) {
        auto [r, s] = call();
        rtt_v[i] = r; srv_v[i] = s;
    }

    auto rs = compute_stats(rtt_v), ss = compute_stats(srv_v);
    double gfl = 2.0 * M * K * N / (ss.avg / 1e3) / 1e9;
    std::cout << "\nMatrix: (" << M << "x" << K << ") x (" << K << "x" << N << ")\n"
              << "  Client RTT  : avg=" << rs.avg << " p50=" << rs.p50 << " p99=" << rs.p99 << " ms\n"
              << "  Server pure : avg=" << ss.avg << " p50=" << ss.p50 << " p99=" << ss.p99 << " ms\n"
              << "  GFLOPS(srv) : " << gfl << "\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string host      = "localhost:50052";
    std::string mode      = "shape_sweep";  // default: production shapes
    std::string sizes_str = "128,256,512,1024,2048";
    int M = 512, K = 512, N = 512, iters = 100, warmup = 20;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](size_t n){ return a.substr(n); };
        if      (a.rfind("--host=",   0)==0) host      = val(7);
        else if (a.rfind("--mode=",   0)==0) mode      = val(7);
        else if (a.rfind("--sizes=",  0)==0) sizes_str = val(8);
        else if (a.rfind("--M=",      0)==0) M         = std::stoi(val(4));
        else if (a.rfind("--K=",      0)==0) K         = std::stoi(val(4));
        else if (a.rfind("--N=",      0)==0) N         = std::stoi(val(4));
        else if (a.rfind("--iters=",  0)==0) iters     = std::stoi(val(8));
        else if (a.rfind("--warmup=", 0)==0) warmup    = std::stoi(val(9));
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    auto stub = gemm::GEMMService::NewStub(
        grpc::CreateChannel(host, grpc::InsecureChannelCredentials()));

    try {
        if      (mode == "shape_sweep") run_shape_sweep(*stub, iters, warmup);
        else if (mode == "sweep")       run_sweep(*stub, parse_ints(sizes_str), iters, warmup);
        else if (mode == "compute")     run_compute(*stub, M, K, N, iters, warmup);
        else { std::cerr << "Unknown mode: " << mode << "\n"; return 1; }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
