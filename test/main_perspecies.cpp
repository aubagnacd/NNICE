// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//
// Standalone checker for the per-species model format.
//
// usage: ./mainPerSpecies model_weights_clusterX.h5 input.dat [norm_param_X.dat norm_param_Y.dat]
//
//   input.dat        : whitespace-separated raw input vector (temperature + Yk),
//                      length must equal the network input size.
//   norm_param_X/Y   : optional. If both are given, the full pipeline
//                      normalize -> run -> denormalize is exercised; otherwise
//                      the raw network output is printed.
//
#include "nnice_perspecies.h"
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 5) {
        std::cout << "usage: " << argv[0]
                  << " model_weights_clusterX.h5 input.dat [norm_param_X.dat norm_param_Y.dat]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const bool do_norm = (argc == 5);

    if (!SnglInferencePerSpecies::is_perspecies_file(model_path)) {
        std::cerr << "ERROR: " << model_path << " is not a per-species file "
                  << "(no \"species_0\" group). Use mainProgramPt for reduced models.\n";
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

    // Read input vector
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

    if (do_norm) {
        net.read_input_norm(argv[3]);
        net.read_output_norm(argv[4]);
        net.normalize_input(input.data());
    }

    double* out = net.run_ai(input.data());

    if (do_norm) net.denormalize_output(out);

    std::cout.precision(17);
    std::cout << "output:";
    for (int i = 0; i < net.n_output_ai; i++) std::cout << " " << out[i];
    std::cout << std::endl;

    return 0;
}
