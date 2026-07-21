#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────
// HELPER TYPES
// ─────────────────────────────────────────────────────────────────

// transition_matrix as a 2D numpy array: columns [sn, a, sn1, prob, reward]
// transition_index as vector<vector<pair<int,int>>>: transition_index[s][a] = {start, end}
using TransitionIndex = std::vector<std::vector<std::pair<int,int>>>;


// ─────────────────────────────────────────────────────────────────
// NORMS
// ─────────────────────────────────────────────────────────────────

double sup_norm(const std::vector<double>& v1, const std::vector<double>& v0) {
    double mx = 0.0;
    for (size_t i = 0; i < v0.size(); i++)
        mx = std::max(mx, std::abs(v1[i] - v0[i]));
    return mx;
}

double relative_sup_norm(const std::vector<double>& v1, const std::vector<double>& v0) {
    double mx = 0.0;
    for (size_t i = 0; i < v0.size(); i++) {
        double d;
        if (v0[i] == 0.0 && v1[i] == 0.0)
            d = 0.0;
        else if (v0[i] == 0.0)
            d = 1000.0;
        else
            d = std::abs((v1[i] - v0[i]) / v0[i]);
        mx = std::max(mx, d);
    }
    return mx;
}

double euclidean_norm(const std::vector<double>& v1, const std::vector<double>& v0) {
    double s = 0.0;
    for (size_t i = 0; i < v0.size(); i++)
        s += (v1[i] - v0[i]) * (v1[i] - v0[i]);
    return std::sqrt(s);
}


// ─────────────────────────────────────────────────────────────────
// BUILD THE transition_index
// ─────────────────────────────────────────────────────────────────

TransitionIndex build_transition_index(py::array_t<double> mt_arr) {
    auto mt = mt_arr.unchecked<2>();
    int nrows = mt_arr.shape(0);

    TransitionIndex index;
    std::vector<std::pair<int,int>> action_index;
    int start = 0;

    for (int i = 0; i < nrows - 1; i++) {
        bool state_changed  = (int)mt(i,0) != (int)mt(i+1,0);
        bool action_changed = (int)mt(i,1) != (int)mt(i+1,1);

        if (state_changed) {
            action_index.push_back({start, i+1});
            index.push_back(action_index);
            action_index.clear();
            start = i+1;
        } else if (action_changed) {
            action_index.push_back({start, i+1});
            start = i+1;
        }
    }
    action_index.push_back({start, nrows});
    index.push_back(action_index);
    return index;
}


// ─────────────────────────────────────────────────────────────────
// Q(s,a) — pure C++ inner product
// ─────────────────────────────────────────────────────────────────

inline double calc_qa(
    const double* mt_data, int ncols,
    int start, int end,
    const std::vector<double>& vn,
    double gamma
) {
    double va = 0.0;
    for (int l = start; l < end; l++) {
        double prob  = mt_data[l * ncols + 3];
        double rt    = mt_data[l * ncols + 4];
        int    nS1   = (int)mt_data[l * ncols + 2];
        va += prob * (rt + gamma * vn[nS1]);
    }
    return va;
}


// ─────────────────────────────────────────────────────────────────
// VALUE ITERATION (Jacobi)
// ─────────────────────────────────────────────────────────────────

py::dict vi_cpp(
    py::array_t<double> mt_arr,
    const TransitionIndex& index,
    int dimS,
    double gamma,
    double tol,
    std::vector<double> v0,
    int max_iter,
    int norm_type
) {
    auto mt   = mt_arr.unchecked<2>();
    const double* mt_data = mt_arr.data();
    int ncols = mt_arr.shape(1);

    std::vector<double> vn  = v0;
    std::vector<double> vn1(dimS, 0.0);
    std::vector<int>    an1(dimS, 0);

    int n = 0;
    bool converged = false;

    while (!converged) {
        n++;

        for (int sn = 0; sn < dimS; sn++) {
            double vamax = -std::numeric_limits<double>::infinity();
            int    amax  = 0;

            for (int aj = 0; aj < (int)index[sn].size(); aj++) {
                int start = index[sn][aj].first;
                int end   = index[sn][aj].second;
                double va = calc_qa(mt_data, ncols, start, end, vn, gamma);

                if (va > vamax) {
                    vamax = va;
                    amax  = (int)mt_data[start * ncols + 1];
                }
            }
            vn1[sn] = vamax;
            an1[sn] = amax;
        }

        double norm;
        if      (norm_type == 0) norm = relative_sup_norm(vn1, vn);
        else if (norm_type == 1) norm = sup_norm(vn1, vn);
        else                     norm = euclidean_norm(vn1, vn);

        if (n >= max_iter || norm < tol)
            converged = true;
        else
            vn = vn1;
    }

    py::dict result;
    result["value"]      = vn1;
    result["policy_idx"] = an1;
    result["n_iter"]     = n;
    return result;
}


// ─────────────────────────────────────────────────────────────────
// VALUE ITERATION GAUSS-SEIDEL
// ─────────────────────────────────────────────────────────────────

py::dict vigs_cpp(
    py::array_t<double> mt_arr,
    const TransitionIndex& index,
    int dimS,
    double gamma,
    double tol,
    std::vector<double> v0,
    int max_iter,
    int norm_type
) {
    const double* mt_data = mt_arr.data();
    int ncols = mt_arr.shape(1);

    std::vector<double> vn  = v0;
    std::vector<double> vn1 = v0;   // Gauss-Seidel: updates in-place
    std::vector<int>    an1(dimS, 0);

    int n = 0;
    bool converged = false;

    while (!converged) {
        n++;

        for (int sn = 0; sn < dimS; sn++) {
            double vamax = -std::numeric_limits<double>::infinity();
            int    amax  = 0;

            for (int aj = 0; aj < (int)index[sn].size(); aj++) {
                int start = index[sn][aj].first;
                int end   = index[sn][aj].second;
                // uses vn1 (already updated) — Gauss-Seidel
                double va = calc_qa(mt_data, ncols, start, end, vn1, gamma);

                if (va > vamax) {
                    vamax = va;
                    amax  = (int)mt_data[start * ncols + 1];
                }
            }
            vn1[sn] = vamax;
            an1[sn] = amax;
        }

        double norm;
        if      (norm_type == 0) norm = relative_sup_norm(vn1, vn);
        else if (norm_type == 1) norm = sup_norm(vn1, vn);
        else                     norm = euclidean_norm(vn1, vn);

        if (n >= max_iter || norm < tol)
            converged = true;
        else
            vn = vn1;
    }

    py::dict result;
    result["value"]      = vn1;
    result["policy_idx"] = an1;
    result["n_iter"]     = n;
    return result;
}


// ─────────────────────────────────────────────────────────────────
// POLICY ITERATION
// ─────────────────────────────────────────────────────────────────

py::dict pi_cpp(
    py::array_t<double> mt_arr,
    const TransitionIndex& index,
    int dimS,
    double gamma,
    double tol,
    std::vector<double> v0,
    std::vector<int> a0,
    int max_iter,
    int norm_type
) {
    const double* mt_data = mt_arr.data();
    int ncols = mt_arr.shape(1);

    std::vector<double> vn  = v0;
    std::vector<double> vn1 = v0;
    std::vector<int>    an1 = a0;

    // range in the transition matrix for the current policy
    std::vector<std::pair<int,int>> policy_range(dimS);
    for (int s = 0; s < dimS; s++)
        policy_range[s] = index[s][a0[s]];

    int n           = 0;
    int n_eval       = 0;
    bool outer_done = false;

    while (!outer_done) {

        // ── policy evaluation ──
        bool inner_done = false;
        while (!inner_done) {
            n_eval++;

            for (int sn = 0; sn < dimS; sn++) {
                int start = policy_range[sn].first;
                int end   = policy_range[sn].second;
                vn1[sn] = calc_qa(mt_data, ncols, start, end, vn1, gamma);
            }

            double norm;
            if      (norm_type == 0) norm = relative_sup_norm(vn1, vn);
            else if (norm_type == 1) norm = sup_norm(vn1, vn);
            else                     norm = euclidean_norm(vn1, vn);

            if (n_eval >= max_iter || norm < tol)
                inner_done = true;
            else
                vn = vn1;
        }

        // ── policy improvement ──
        n++;
        for (int sn = 0; sn < dimS; sn++) {
            double vamax = -std::numeric_limits<double>::infinity();
            int    amax  = 0;
            std::pair<int,int> rmax = {0,0};

            for (int aj = 0; aj < (int)index[sn].size(); aj++) {
                int start = index[sn][aj].first;
                int end   = index[sn][aj].second;
                double va = calc_qa(mt_data, ncols, start, end, vn1, gamma);

                if (va > vamax) {
                    vamax = va;
                    amax  = (int)mt_data[start * ncols + 1];
                    rmax  = {start, end};
                }
            }
            vn1[sn]          = vamax;
            an1[sn]          = amax;
            policy_range[sn] = rmax;
        }

        double norm;
        if      (norm_type == 0) norm = relative_sup_norm(vn1, vn);
        else if (norm_type == 1) norm = sup_norm(vn1, vn);
        else                     norm = euclidean_norm(vn1, vn);

        if (n >= max_iter || norm < tol)
            outer_done = true;
        else {
            vn     = vn1;
            n_eval = 0;
        }
    }

    py::dict result;
    result["value"]      = vn1;
    result["policy_idx"] = an1;
    result["n_iter"]     = n;
    return result;
}


// ─────────────────────────────────────────────────────────────────
// Q-LEARNING
// ─────────────────────────────────────────────────────────────────

py::dict qlearning_cpp(
    py::array_t<double> mt_arr,
    const TransitionIndex& index,
    int dimS,
    double gamma,
    double epsilon_final,
    double alpha,
    int s0,
    int max_iter,
    unsigned int seed
) {
    const double* mt_data = mt_arr.data();
    int ncols = mt_arr.shape(1);

    // initialize Q with zeros
    std::vector<std::vector<double>> Q(dimS);
    for (int s = 0; s < dimS; s++)
        Q[s].assign(index[s].size(), 0.0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int sn = s0;
    double epsilon_init = 1.0;

    for (int n = 1; n <= max_iter; n++) {
        double epsilon = epsilon_init - (double)n * (epsilon_init - epsilon_final) / max_iter;

        int a;
        if (dist(rng) >= epsilon) {
            // greedy
            a = (int)(std::max_element(Q[sn].begin(), Q[sn].end()) - Q[sn].begin());
        } else {
            std::uniform_int_distribution<int> da(0, (int)Q[sn].size()-1);
            a = da(rng);
        }

        int start = index[sn][a].first;
        int end   = index[sn][a].second;

        // draws the next state from the CDF
        double roll     = dist(rng);
        double cum_prob = 0.0;
        int row         = start;
        for (int l = start; l < end; l++) {
            cum_prob += mt_data[l * ncols + 3];
            row = l;
            if (cum_prob >= roll) break;
        }

        double rt  = mt_data[row * ncols + 4];
        int    sn1 = (int)mt_data[row * ncols + 2];

        double max_q_sn1 = *std::max_element(Q[sn1].begin(), Q[sn1].end());
        Q[sn][a] += alpha * (rt + gamma * max_q_sn1 - Q[sn][a]);

        sn = sn1;
    }

    // extract policy and value
    std::vector<double> value(dimS);
    std::vector<int>    policy(dimS);
    for (int s = 0; s < dimS; s++) {
        int maxj = (int)(std::max_element(Q[s].begin(), Q[s].end()) - Q[s].begin());
        policy[s] = (int)mt_data[index[s][maxj].first * ncols + 1];
        value[s]  = Q[s][maxj];
    }

    py::dict result;
    result["value"]      = value;
    result["policy_idx"] = policy;
    result["n_iter"]     = max_iter;
    return result;
}


// ─────────────────────────────────────────────────────────────────
// SARSA
// ─────────────────────────────────────────────────────────────────

py::dict sarsa_cpp(
    py::array_t<double> mt_arr,
    const TransitionIndex& index,
    int dimS,
    double gamma,
    double epsilon_final,
    double alpha,
    int s0,
    int max_iter,
    unsigned int seed
) {
    const double* mt_data = mt_arr.data();
    int ncols = mt_arr.shape(1);

    std::vector<std::vector<double>> Q(dimS);
    for (int s = 0; s < dimS; s++)
        Q[s].assign(index[s].size(), 0.0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int sn       = s0;
    double epsilon_init = 1.0;

    // choose initial action
    int an;
    {
        std::uniform_int_distribution<int> da(0, (int)Q[sn].size()-1);
        an = da(rng);
    }

    for (int n = 1; n <= max_iter; n++) {
        double epsilon = epsilon_init - (double)n * (epsilon_init - epsilon_final) / max_iter;

        int start = index[sn][an].first;
        int end   = index[sn][an].second;

        // draws the next state
        double roll     = dist(rng);
        double cum_prob = 0.0;
        int row         = start;
        for (int l = start; l < end; l++) {
            cum_prob += mt_data[l * ncols + 3];
            row = l;
            if (cum_prob >= roll) break;
        }

        double rt  = mt_data[row * ncols + 4];
        int    sn1 = (int)mt_data[row * ncols + 2];

        // choose next action (on-policy)
        int an1;
        if (dist(rng) >= epsilon) {
            an1 = (int)(std::max_element(Q[sn1].begin(), Q[sn1].end()) - Q[sn1].begin());
        } else {
            std::uniform_int_distribution<int> da(0, (int)Q[sn1].size()-1);
            an1 = da(rng);
        }

        // SARSA update (on-policy: uses Q[sn1][an1])
        Q[sn][an] += alpha * (rt + gamma * Q[sn1][an1] - Q[sn][an]);

        sn = sn1;
        an = an1;
    }

    std::vector<double> value(dimS);
    std::vector<int>    policy(dimS);
    for (int s = 0; s < dimS; s++) {
        int maxj = (int)(std::max_element(Q[s].begin(), Q[s].end()) - Q[s].begin());
        policy[s] = (int)mt_data[index[s][maxj].first * ncols + 1];
        value[s]  = Q[s][maxj];
    }

    py::dict result;
    result["value"]      = value;
    result["policy_idx"] = policy;
    result["n_iter"]     = max_iter;
    return result;
}


// ─────────────────────────────────────────────────────────────────
// PYBIND11 MODULE
// ─────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_mdp_core, m) {
    m.doc() = "mdp-solver C++ core";

    m.def("build_transition_index", &build_transition_index,
          "Build the transition_index from the transition_matrix numpy array.");

    m.def("vi_cpp",   &vi_cpp,
          "Value Iteration (Jacobi) in C++.");

    m.def("vigs_cpp", &vigs_cpp,
          "Value Iteration Gauss-Seidel in C++.");

    m.def("pi_cpp",   &pi_cpp,
          "Policy Iteration in C++.");

    m.def("qlearning_cpp", &qlearning_cpp,
          "Q-Learning in C++.");

    m.def("sarsa_cpp", &sarsa_cpp,
          "SARSA in C++.");

    m.def("sup_norm",          &sup_norm);
    m.def("relative_sup_norm", &relative_sup_norm);
    m.def("euclidean_norm",    &euclidean_norm);
}
