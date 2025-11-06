#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <random>
#include <dirent.h>
#include <sys/stat.h>

#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ---------------- utilities ----------------
void createDirectoryIfMissing(const string &path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
}
bool isDirectory(const string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return S_ISDIR(info.st_mode);
}
string filenameOnly(const string &p) {
    size_t pos = p.find_last_of("/\\");
    return (pos == string::npos) ? p : p.substr(pos + 1);
}
vector<string> getCSVFiles(const string &folder) {
    vector<string> out;
    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        cerr << "Error: Cannot open directory " << folder << endl;
        return out;
    }
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        string name = ent->d_name;
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
            out.push_back(folder + "/" + name);
        }
    }
    closedir(dir);
    sort(out.begin(), out.end());
    return out;
}

// ---------------- Haar DWT (single level) ----------------
void dwtHaarSingleLevel(const vector<double>& x, vector<double>& approx, vector<double>& detail) {
    size_t n = x.size();
    size_t half = n / 2;
    approx.assign(half, 0.0);
    detail.assign(half, 0.0);
    double s = 1.0 / std::sqrt(2.0);
    for (size_t k = 0; k < half; ++k) {
        approx[k] = s * (x[2 * k] + x[2 * k + 1]);
        detail[k] = s * (x[2 * k] - x[2 * k + 1]);
    }
}
void idwtHaarSingleLevel(const vector<double>& approx, const vector<double>& detail, vector<double>& out) {
    size_t half = approx.size();
    out.assign(half * 2, 0.0);
    double s = 1.0 / std::sqrt(2.0);
    for (size_t k = 0; k < half; ++k) {
        out[2 * k] = s * (approx[k] + detail[k]);
        out[2 * k + 1] = s * (approx[k] - detail[k]);
    }
}
inline double soft(double v, double T) {
    if (v > T) return v - T;
    if (v < -T) return v + T;
    return 0.0;
}

// ---------------- multi-level Haar denoise (safe) ----------------
vector<double> waveletDenoiseHaarSafe(const vector<double>& signal, int levels = 3, double safetyFactor = 0.75) {
    int n = (int)signal.size();
    if (n <= 0) return {};
    int target = n;
    while (target % (1 << levels) != 0) ++target;
    vector<double> work = signal;
    if ((int)work.size() < target) work.resize(target, 0.0);

    vector<vector<double>> details(levels);
    int curLen = target;
    for (int L = 0; L < levels; ++L) {
        if (curLen % 2 != 0) { work.push_back(0.0); ++curLen; }
        vector<double> slice(work.begin(), work.begin() + curLen);
        vector<double> approx, detail;
        dwtHaarSingleLevel(slice, approx, detail);
        details[L] = std::move(detail);
        for (size_t i = 0; i < approx.size(); ++i) work[i] = approx[i];
        curLen = (int)approx.size();
    }
    vector<double> approx(work.begin(), work.begin() + curLen);

    if (details.empty() || details[0].empty()) {
        approx.resize(n);
        return approx;
    }

    vector<double> finest = details[0];
    vector<double> absd(finest.size());
    for (size_t i = 0; i < finest.size(); ++i) absd[i] = fabs(finest[i]);
    double med;
    {
        auto tmp = absd;
        size_t mid = tmp.size() / 2;
        nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
        med = tmp[mid];
    }
    double sigma = med / 0.6745;
    if (!(sigma > 0)) sigma = 1e-8;

    for (int L = 0; L < levels; ++L) {
        int len = (int)details[L].size();
        double T = sigma * sqrt(2.0 * log(max(2, len))) * safetyFactor;
        for (int k = 0; k < len; ++k) details[L][k] = soft(details[L][k], T);
    }

    vector<double> rec = approx;
    for (int L = levels - 1; L >= 0; --L) {
        vector<double> tmp;
        idwtHaarSingleLevel(rec, details[L], tmp);
        rec.swap(tmp);
    }
    rec.resize(n);
    return rec;
}

// ---------------- FastICA (symmetric) with explicit .eval() usage ----------------
struct ICAResult { MatrixXd S; MatrixXd unmix; }; // S: m x nSamp, unmix: m x nChan

ICAResult fastICA_symmetric_explicit(const MatrixXd& X_in, int nComp, int maxIter = 300, double tol = 1e-6) {
    int nChan = (int)X_in.rows();
    int nSamp = (int)X_in.cols();
    int m = std::min(nComp, nChan);
    if (nSamp <= 1) throw runtime_error("Not enough samples.");

    MatrixXd X = X_in; // copy
    // center rows
    for (int r = 0; r < nChan; ++r) {
        double mu = X.row(r).mean();
        X.row(r).array() -= mu;
    }

    // covariance
    MatrixXd cov = (X * X.transpose()).eval() / double(max(1, nSamp));
    // eigen-decomp
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(cov);
    if (es.info() != Eigen::Success) throw runtime_error("Eigen decomposition failed.");
    VectorXd eigvals_full = es.eigenvalues().eval();
    MatrixXd eigvecs_full = es.eigenvectors().eval();

    // sort eigenvalues descending and select top-m
    vector<pair<double, int>> ev;
    ev.reserve(nChan);
    for (int i = 0; i < nChan; ++i) ev.emplace_back(eigvals_full(i), i);
    sort(ev.rbegin(), ev.rend());

    MatrixXd Esel(nChan, m);
    VectorXd Dsel(m);
    for (int i = 0; i < m; ++i) {
        Dsel(i) = max(ev[i].first, 0.0);
        Esel.col(i) = eigvecs_full.col(ev[i].second);
    }

    VectorXd invSqrt = Dsel.unaryExpr([](double v) { return (v > 1e-12) ? 1.0 / sqrt(v) : 0.0; });
    MatrixXd DinvSqrt = invSqrt.asDiagonal();
    MatrixXd whitening = (DinvSqrt * Esel.transpose()).eval(); // m x nChan

    MatrixXd Xw = (whitening * X).eval(); // m x nSamp

    // init W random orthonormal
    std::mt19937_64 rng(1234567);
    std::normal_distribution<double> nd(0.0, 1.0);
    MatrixXd W = MatrixXd::Zero(m, m);
    for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) W(i, j) = nd(rng);
    {
        Eigen::JacobiSVD<MatrixXd> svd0(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        W = (svd0.matrixU() * svd0.matrixV().transpose()).eval();
    }

    MatrixXd WX(m, nSamp);
    MatrixXd gw(m, nSamp);
    MatrixXd gwp(m, nSamp);

    for (int it = 0; it < maxIter; ++it) {
        WX.noalias() = (W * Xw).eval();
        for (int i = 0; i < m; ++i) for (int j = 0; j < nSamp; ++j) {
            double v = WX(i, j);
            double t = tanh(v);
            gw(i, j) = t;
            gwp(i, j) = 1.0 - t * t;
        }

        MatrixXd term1 = (gw * Xw.transpose()).eval() / double(max(1, nSamp));
        VectorXd gwp_mean(m);
        for (int i = 0; i < m; ++i) gwp_mean(i) = gwp.row(i).mean();

        MatrixXd Wnew = term1;
        for (int i = 0; i < m; ++i) Wnew.row(i).noalias() -= gwp_mean(i) * W.row(i);

        MatrixXd K = (Wnew * Wnew.transpose()).eval();
        K = ((K + K.transpose()) * 0.5).eval();
        Eigen::SelfAdjointEigenSolver<MatrixXd> es2(K);
        if (es2.info() != Eigen::Success) throw runtime_error("Eigen failed in decorrelation.");
        VectorXd lam = es2.eigenvalues().eval();
        MatrixXd P = es2.eigenvectors().eval();
        MatrixXd invS = MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i) { double v = lam(i); invS(i, i) = (v > 1e-12) ? 1.0 / sqrt(v) : 0.0; }
        MatrixXd decor = (P * invS * P.transpose()).eval();
        Wnew = (decor * Wnew).eval();

        double maxChange = 0.0;
        for (int i = 0; i < m; ++i) {
            double dot = fabs(Wnew.row(i).dot(W.row(i)));
            double ch = 1.0 - dot;
            if (ch > maxChange) maxChange = ch;
        }
        W = Wnew;
        if (maxChange < tol) break;
    }

    MatrixXd S = (W * Xw).eval();            // m x nSamp
    MatrixXd unmix = (W * whitening).eval(); // m x nChan

    return {S, unmix};
}

// ---------------- kurtosis ----------------
double kurtosisExcess(const Eigen::VectorXd &v) {
    int n = v.size();
    if (n < 2) return 0.0;
    double mean = v.mean();
    VectorXd z = v.array() - mean;
    double s2 = (z.squaredNorm()) / double(n);
    if (s2 <= 0) return 0.0;
    double s4 = z.array().pow(4).sum() / double(n);
    return s4 / (s2 * s2) - 3.0;
}

// ---------------- CSV helpers ----------------
MatrixXd loadCSVmatrix(const string &path) {
    ifstream f(path);
    if (!f.is_open()) throw runtime_error("Cannot open " + path);
    vector<vector<double>> rows;
    string line;
    while (getline(f, line)) {
        if (line.size() == 0) continue;
        stringstream ss(line);
        string tok;
        vector<double> row;
        while (getline(ss, tok, ',')) {
            try { row.push_back(stod(tok)); } catch (...) { row.push_back(0.0); }
        }
        rows.push_back(row);
    }
    if (rows.empty()) throw runtime_error("Empty CSV: " + path);
    size_t R = rows.size(), C = rows[0].size();
    MatrixXd M((int)R, (int)C);
    for (size_t i = 0; i < R; ++i) {
        if (rows[i].size() != C) rows[i].resize(C, 0.0);
        for (size_t j = 0; j < C; ++j) M((int)i, (int)j) = rows[i][j];
    }
    return M;
}
void saveCSV_full(const string &path, const MatrixXd &M) {
    ofstream f(path);
    f << fixed << setprecision(10);
    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            f << M(r, c);
            if (c < M.cols() - 1) f << ",";
        }
        f << "\n";
    }
}

// ---------------- main file processing ----------------
void processOneFile(const string &inPath, const string &outFolder) {
    cout << "Processing: " << inPath << "\n";
    MatrixXd raw = loadCSVmatrix(inPath);
    cout << "Loaded shape: " << raw.rows() << " x " << raw.cols() << "\n";

    // Adapt orientation to 60x1280 (channels x samples) and remove first channel
    if (raw.rows() == 1280 && raw.cols() == 60) {
        raw = raw.transpose().eval();
        cout << "Transposed input to 60 x 1280 (channels x samples).\n";
    } else if (raw.rows() != 60 || raw.cols() != 1280) {
        cerr << "Warning: Expected 60x1280 or 1280x60, got " << raw.rows() << "x" << raw.cols() << ". Proceeding with current shape.\n";
    }
    if (raw.rows() > 1) { // Ensure at least 2 channels remain
        raw = raw.bottomRows(raw.rows() - 1).eval(); // Remove first channel
        cout << "Removed first channel. New shape: " << raw.rows() << " x " << raw.cols() << "\n";
    } else {
        cerr << "Error: Not enough channels after removing the first one.\n";
        return;
    }
    int nChan = raw.rows(); // Updated number of channels (59)
    int nSamp = raw.cols();
    cout << "Using shape " << nChan << " x " << nSamp << "\n";

    // Jitter tiny-variance channels
    for (int r = 0; r < nChan; ++r) {
        double mean = raw.row(r).mean();
        double s2 = (raw.row(r).array() - mean).square().sum() / double(max(1, nSamp));
        if (!(s2 > 1e-16)) {
            for (int c = 0; c < nSamp; ++c) raw(r, c) += 1e-9 * ((c % 2) ? 1.0 : -1.0);
        }
    }

    // Per-channel Haar denoise
    MatrixXd den(nChan, nSamp);
    for (int r = 0; r < nChan; ++r) {
        vector<double> row(nSamp);
        for (int c = 0; c < nSamp; ++c) row[c] = raw(r, c);
        vector<double> out = waveletDenoiseHaarSafe(row, 3, 0.75);
        for (int c = 0; c < nSamp; ++c) den(r, c) = out[c];
    }

    cout << "After denoise ch0 first 8: ";
    for (int i = 0; i < min(8, nSamp); ++i) cout << den(0, i) << " "; cout << "\n";

    int m = nChan; // Use all remaining channels for ICA
    ICAResult ica = fastICA_symmetric_explicit(den, m);

    cout << "ICA S shape: " << ica.S.rows() << " x " << ica.S.cols() << "\n";

    // Detect artifact ICs by kurtosis
    vector<int> bad;
    for (int ic = 0; ic < ica.S.rows(); ++ic) {
        double k = kurtosisExcess(ica.S.row(ic).transpose());
        if (std::isfinite(k) && fabs(k) > 8.0) bad.push_back(ic); // Adjustable threshold
    }
    cout << "Artifact ICs: ";
    for (int b : bad) cout << b << " "; cout << "\n";

    // Reconstruct cleaned signal
    Eigen::JacobiSVD<MatrixXd> svdA(ica.unmix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd svals = svdA.singularValues();
    double tol = 1e-8;
    VectorXd invs(svals.size());
    for (int i = 0; i < svals.size(); ++i) invs(i) = (svals(i) > tol) ? 1.0 / svals(i) : 0.0;
    MatrixXd A = (svdA.matrixV() * invs.asDiagonal() * svdA.matrixU().transpose()).eval(); // nChan x m

    MatrixXd Sclean = ica.S;
    for (int idx : bad) if (idx >= 0 && idx < Sclean.rows()) Sclean.row(idx).setZero();
    MatrixXd Xclean = (A * Sclean).eval(); // 59 x 1280 cleaned signal

    cout << "After reconstruction ch0 first 8: ";
    for (int i = 0; i < min(8, nSamp); ++i) cout << Xclean(0, i) << " "; cout << "\n";

    string fname = filenameOnly(inPath);
    string outPath = outFolder + "/" + fname;
    saveCSV_full(outPath, Xclean);
    cout << "Saved cleaned file: " << outPath << "\n";
}

// ---------------- main ----------------
int main() {
    string inFolder = "C:/Users/vishw/Documents/ASUfeature/ASUdata";
    string outFolder = "C:/Users/vishw/Documents/ASUfeature/WICA";

    if (!isDirectory(inFolder)) {
        cerr << "Input folder missing: " << inFolder << "\n";
        return 1;
    }
    createDirectoryIfMissing(outFolder);

    vector<string> files = getCSVFiles(inFolder);
    if (files.empty()) {
        cerr << "No CSV files found in " << inFolder << "\n";
        return 1;
    }

    for (const string &f : files) {
        try {
            processOneFile(f, outFolder);
        } catch (exception &e) {
            cerr << "Error processing " << f << ": " << e.what() << "\n";
        }
    }

    cout << "All done.\n";
    return 0;
}