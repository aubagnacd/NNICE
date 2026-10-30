// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//
// Cross-validation driver: reproduces the ARF (PyTorch) per-species
// prediction pipeline (log_transform_X=1, log_transform_Y=1, threshold=1e-14,
// output_omegas=true) using NNICE's C++ per-species inference, on the exact
// same trained weights and the exact same fixed input state, so its output
// can be diffed against ai_reacting_flows/.idea/CFD_REDUCED_MECH_A/
// compare_arf_nnice/output_arf.txt.
//
// usage: ./mainPerSpeciesArfCompare model.h5 input_raw.dat norm_param_X.dat norm_param_Y.dat species_names.txt [output.txt]
//
//   input_raw.dat      : whitespace-separated raw state, T then Yk (18 species),
//                        same order as norm_param_X/Y and species_names.txt.
//   species_names.txt  : one species name per line, in the same order as the
//                        Yk in input_raw.dat (used only to label the output).
//   output.txt          : optional; if given, writes "name value" lines like
//                          output_arf.txt for an easy diff.
//
#include "nnice_perspecies.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7) {
        std::cout << "usage: " << argv[0]
                  << " model.h5 input_raw.dat norm_param_X.dat norm_param_Y.dat species_names.txt [output.txt]\n";
        return 1;
    }

    const std::string model_path   = argv[1];
    const std::string input_path   = argv[2];
    const std::string normX_path   = argv[3];
    const std::string normY_path   = argv[4];
    const std::string species_path = argv[5];
    const bool write_output = (argc == 7);

    if (!SnglInferencePerSpecies::is_perspecies_file(model_path)) {
        std::cerr << "ERROR: " << model_path << " is not a per-species file "
                  << "(no \"species_0\" group).\n";
        return 1;
    }

    SnglInferencePerSpecies net;
    try {
        net.ImportNN_pt_custom_perspecies(model_path);
    } catch (const std::string& msg) {
        std::cerr << "ERROR: " << msg;
        return 1;
    }

    std::cout << "n_input_ai  = " << net.n_input_ai  << "\n";
    std::cout << "n_output_ai = " << net.n_output_ai << "\n";
    std::cout << "n_species   = " << net.n_species() << "\n";

    // Read raw input vector: T, Y_1..Y_18
    std::vector<double> input;
    {
        std::ifstream fin(input_path);
        if (!fin.is_open()) { std::cerr << "cannot open " << input_path << "\n"; return 1; }
        double v;
        while (fin >> v) input.push_back(v);
    }
    if ((int)input.size() != net.n_input_ai) {
        std::cerr << "input length " << input.size() << " != n_input_ai " << net.n_input_ai << "\n";
        return 1;
    }

    // Read species names (for labeling only)
    std::vector<std::string> species_names;
    {
        std::ifstream fsp(species_path);
        if (!fsp.is_open()) { std::cerr << "cannot open " << species_path << "\n"; return 1; }
        std::string name;
        while (fsp >> name) species_names.push_back(name);
    }
    if ((int)species_names.size() != net.n_output_ai) {
        std::cerr << "species_names length " << species_names.size()
                   << " != n_output_ai " << net.n_output_ai << "\n";
        return 1;
    }

    net.read_input_norm(normX_path);
    net.read_output_norm(normY_path);

    // --- Step 1: X_species_t[k] = log(clip(Y0[k], log_threshold)), T untouched ---
    net.log_threshold = 1.0e-14;
    std::vector<double> state(input);           // working copy, will be overwritten in place
    net.apply_log_transform(state.data());      // touches state[1..n_input_ai-1] only

    // Keep a copy of the log-transformed species part (X_species_t), since
    // normalize_input() below overwrites `state` in place with the scaled input.
    std::vector<double> X_species_t(net.n_output_ai);
    for (int k = 0; k < net.n_output_ai; k++) X_species_t[k] = state[k + 1];

    // --- Steps 3-4: scale input, run network ---
    net.normalize_input(state.data());
    double* raw_out = net.run_ai(state.data());

    // --- Step 5: denormalize (still in transformed/log space) -> omega_t ---
    std::vector<double> omega_t(raw_out, raw_out + net.n_output_ai);
    net.denormalize_output(omega_t.data());

    // --- Step 6: combined = X_species_t + omega_t (residual in log space) ---
    std::vector<double> combined(net.n_output_ai);
    for (int k = 0; k < net.n_output_ai; k++) combined[k] = X_species_t[k] + omega_t[k];

    // --- Step 7: Y_new = exp(combined) ---
    net.inverse_log_transform(combined.data());
    const std::vector<double>& Y_new = combined;

    std::cout.precision(17);
    std::cout << "\nNNICE prediction (Y_new):\n";
    for (int k = 0; k < net.n_output_ai; k++) {
        std::cout << "  " << species_names[k] << " " << Y_new[k] << "\n";
    }

    if (write_output) {
        std::ofstream fout(argv[6]);
        fout.precision(17);
        for (int k = 0; k < net.n_output_ai; k++) {
            fout << species_names[k] << " " << Y_new[k] << "\n";
        }
        std::cout << "\nWrote " << argv[6] << "\n";
    }

    return 0;
}
