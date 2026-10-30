// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "nnice_perspecies.h"
#include "hdf5.h"

SnglInferencePerSpecies::~SnglInferencePerSpecies()
{
    for (size_t i = 0; i < species_models.size(); i++) delete species_models[i];
    species_models.clear();
}

bool SnglInferencePerSpecies::is_perspecies_file(const string& model_path)
{
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t h5Model = H5Fopen(model_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5Model < 0) return false;
    bool has_species = (H5Lexists(h5Model, "species_0", H5P_DEFAULT) > 0);
    H5Fclose(h5Model);
    return has_species;
}

void SnglInferencePerSpecies::ImportNN_pt_custom_perspecies(const string& model_path)
{
    ImportNN_pt_custom_perspecies(model_path, "dense_layer_", "hidden_residual_block_", "output_layer");
}

void SnglInferencePerSpecies::ImportNN_pt_custom_perspecies(const string& model_path,
                                                            const string& hidden_layers_prefix,
                                                            const string& resblock_prefix,
                                                            const string& output_layer_name)
{
    //
    // Count the "species_i" groups in the file
    //
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t h5Model = H5Fopen(model_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5Model < 0) {
        string msg("Could not open HDF5 file containing per-species NN : " + model_path + "\n");
        throw msg;
    }
    int n_sp = 0;
    while (H5Lexists(h5Model, ("species_" + std::to_string(n_sp)).c_str(), H5P_DEFAULT) > 0) n_sp++;
    H5Fclose(h5Model);
    if (n_sp == 0) {
        string msg("No \"species_i\" group found in per-species NN file : " + model_path + "\n");
        throw msg;
    }
    //
    // Build one sub-network per species. Each sub-network is a regular
    // ImportNN_pt_custom import rooted inside the "species_i/" group.
    //
    species_models.reserve(n_sp);
    for (int s = 0; s < n_sp; s++) {
        string grp = "species_" + std::to_string(s) + "/";
        SnglInference* net = new SnglInference();
        net->ImportNN_pt_custom(model_path, "",
                                grp + hidden_layers_prefix,
                                grp + resblock_prefix,
                                grp + output_layer_name);
        species_models.push_back(net);
    }
    //
    // Global metadata: input size taken from the sub-networks (they all share it),
    // output size is the number of species sub-networks.
    //
    n_input_ai  = species_models[0]->n_input_ai;
    n_output_ai = n_sp;
    output_buffer.assign(n_output_ai, 0.0);

    log_threshold = 1.0; // SetLogThreshold and SetBctCst TBD (kept consistent with Inference)
    bct_constant  = 1.0;
    rbct = 1.0 / bct_constant;

    build_fused();
}

void SnglInferencePerSpecies::build_fused()
{
    fused = false;
    const int S = static_cast<int>(species_models.size());
    const SnglInference* ref = species_models[0];
    const size_t nl = ref->topology.size(); // n_in, H, ..., H, 1
    //
    // Eligibility : Dense-only, >= 1 hidden layer, constant hidden width, scalar output,
    // identical topology / activations for every species.
    //
    if (nl < 3 || ref->topology.back() != 1) return;
    const int H = static_cast<int>(ref->topology[1]);
    for (size_t l = 1; l + 1 < nl; l++) if (static_cast<int>(ref->topology[l]) != H) return;
    for (int s = 0; s < S; s++) {
        const SnglInference* net = species_models[s];
        if (net->topology != ref->topology) return;
        if (net->activationFunctions != ref->activationFunctions) return;
        for (size_t l = 0; l < net->layers.size(); l++)
            if (net->layers[l] != &SnglInference::Dense) return;
    }
    //
    // Pack weights
    //
    const int n_in = static_cast<int>(ref->topology[0]);
    const size_t n_layers = nl - 1;         // number of weight matrices
    n_hidden = H;
    fused_first.assign(1, Matrix(n_in, S * H));
    fused_hidden.assign(n_layers - 2, std::vector<Matrix, Eigen::aligned_allocator<Matrix>>(S));
    fused_bias.assign(n_layers - 1, RowVector(S * H));
    fused_act.assign(n_layers - 1, RowVector(S * H));
    fused_out_w.resize(H, S);
    fused_out_b.resize(S);
    fused_y.resize(S);
    for (int s = 0; s < S; s++) {
        const SnglInference* net = species_models[s];
        fused_first[0].middleCols(s * H, H) = net->weights[0];
        for (size_t l = 0; l + 1 < n_layers; l++) fused_bias[l].segment(s * H, H) = net->bias[l];
        for (size_t l = 1; l + 1 < n_layers; l++) fused_hidden[l - 1][s] = net->weights[l];
        fused_out_w.col(s) = net->weights[n_layers - 1].col(0);
        fused_out_b(s) = net->bias[n_layers - 1](0);
    }
    fused_activations = ref->activationFunctions;
    fused = true;
}

double* SnglInferencePerSpecies::run_fused(double* input_ai)
{
    const int S = n_output_ai;
    const int H = n_hidden;
    Eigen::Map<const RowVector> x(input_ai, n_input_ai);
    //
    // First layer : one GEMV for all species
    //
    fused_act[0].noalias() = x * fused_first[0];
    fused_act[0] += fused_bias[0];
    fused_activations[0](fused_act[0]);
    //
    // Hidden layers : block-diagonal
    //
    for (size_t l = 0; l < fused_hidden.size(); l++) {
        const RowVector& h_in = fused_act[l];
        RowVector& h_out = fused_act[l + 1];
        for (int s = 0; s < S; s++)
            h_out.segment(s * H, H).noalias() = h_in.segment(s * H, H) * fused_hidden[l][s];
        h_out += fused_bias[l + 1];
        fused_activations[l + 1](h_out);
    }
    //
    // Output layer : one dot product per species
    //
    const RowVector& h_last = fused_act.back();
    RowVector& out = fused_y;
    for (int s = 0; s < S; s++)
        out(s) = h_last.segment(s * H, H).dot(fused_out_w.col(s).transpose()) + fused_out_b(s);
    fused_activations.back()(out);
    for (int s = 0; s < S; s++) output_buffer[s] = out(s);
    return output_buffer.data();
}

double* SnglInferencePerSpecies::run_ai(double* input_ai)
{
    if (fused) return run_fused(input_ai);
    for (int s = 0; s < n_output_ai; s++) {
        double* y_s = species_models[s]->run_ai(input_ai);
        output_buffer[s] = y_s[0];
    }
    return output_buffer.data();
}

double* SnglInferencePerSpecies::run_ai(double* input_ai, const size_t& size)
{
    // TBD : true vectorized path (size ignored, same as SnglInference)
    return run_ai(input_ai);
}
