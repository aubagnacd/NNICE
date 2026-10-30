// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef SNGL_INF_PERSPECIES_H
#define SNGL_INF_PERSPECIES_H
#include "nnice_single.h"

// Per-species inference.
//
// In the "per-species" model format (see ai_reacting_flows EXEMPLE_MODEL_PERSPECIES),
// a single HDF5 file holds one independent sub-network per species, stored in groups
// named "species_0", "species_1", ... Every sub-network takes the *same* input vector
// (temperature + species mass fractions, size n_input_ai) and produces a single scalar
// output. The global network output is the concatenation of the sub-network outputs,
// in species order (size n_output_ai = number of species groups).
//
// Input/output normalization and the optional log/BCT data transforms are applied
// globally, exactly as for the single-network "reduced" model, using the
// norm_param_X_cluster*.dat (n_input_ai lines) and norm_param_Y_cluster*.dat
// (n_output_ai lines) files. Those inherited methods (from SnglInference) are reused
// as-is: only network import and forward evaluation are specialized here.
class SnglInferencePerSpecies : public SnglInference
{
    public:
        SnglInferencePerSpecies() {};
        ~SnglInferencePerSpecies();

        // Imports every "species_i" sub-network from a per-species HDF5 file.
        // The layer names passed are the ones *local* to each "species_i" group;
        // the default values match the ai_reacting_flows pytorch export.
        void ImportNN_pt_custom_perspecies(const string& model_path);
        void ImportNN_pt_custom_perspecies(const string& model_path,
                                           const string& hidden_layers_prefix,
                                           const string& resblock_prefix,
                                           const string& output_layer_name);

        double* run_ai(double* input_ai);
        double* run_ai(double* input_ai, const size_t& size);

        int n_species() const { return static_cast<int>(species_models.size()); };

        // Returns true if model_path is a per-species file (contains a "species_0" group).
        static bool is_perspecies_file(const string& model_path);

        // True when the sub-networks were fused into a single forward pass (see build_fused)
        bool is_fused() const { return fused; };

    protected:
        std::vector<SnglInference*> species_models;
        std::vector<double> output_buffer;

        // Fused forward pass. When every sub-network is a plain Dense MLP with the same
        // topology and activations, they are evaluated together instead of one by one:
        //   - first layer : all S kernels stacked into one (n_in x S*H) matrix -> one GEMV
        //   - hidden layers: block-diagonal, S small (H x H) GEMVs on contiguous segments
        //   - output layer : S dot products (H x S matrix, one column per species)
        // Activations are applied once on the whole S*H vector. Mathematically identical
        // to the per-sub-network loop; it just removes per-call overhead and lets the
        // activation run on long vectors.
        void build_fused();
        double* run_fused(double* input_ai);

        bool fused = false;
        int n_hidden = 0;                                                     // H
        std::vector<Matrix, Eigen::aligned_allocator<Matrix>> fused_first;    // [0]: n_in x S*H
        std::vector<std::vector<Matrix, Eigen::aligned_allocator<Matrix>>> fused_hidden; // [layer][s]: H x H
        Matrix fused_out_w;                                                   // H x S
        RowVector fused_out_b;                                                // S
        RowVector fused_y;                                                    // S (work buffer)
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> fused_bias; // per hidden layer, S*H
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> fused_act;  // work buffers, S*H
        std::vector<activ_ptr> fused_activations;                             // one per layer (incl. output)
};
#endif
