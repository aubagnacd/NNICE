// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "nnice_single.h"
#include "vectorized.h"
#include "data.h"
#include <chrono>
#include <set>

int main(int argc, char** argv) {
    string c;
    size_t n_repeat;
    bool show_results = false;
    std::vector<string> casenames = {"ReLU","tanh","swish"};
    if ((argc > 2)&&(argc < 5)) {
        if (std::find(casenames.begin(), casenames.end(), argv[1])!= casenames.end()) {
            c = argv[1];
        } else {
            cout << "case should be \"" << casenames[0] << "\", \"" << casenames[1] << "\" or \"" << casenames[2] <<"\"" << endl;
            exit(1);
        }
        //
        try {
            string arg = argv[2];
            size_t pos;
            n_repeat = std::stoi(arg, &pos);
            if (pos < arg.size()) {
                std::cerr << "Trailing characters after number: " << arg << '\n';
            }
        } catch (std::invalid_argument const &ex) {
                std::cerr << "Invalid number: " << argv[2] << '\n';
                exit(1);
        } catch (std::out_of_range const &ex) {
                std::cerr << "Number out of range: " << argv[2] << '\n';
                exit(1);
        }
        //
        if (argc>3) {
            string arg = argv[3];
            if (arg == "True" || arg == "true" || arg == "T") {
                show_results = true;
                n_repeat = 1;
            }
        }
        //
        // if (argc==5) {
        //     try {
        //     string arg = argv[4];
        //     size_t pos;
        //     size_t n_thread = std::stoi(arg, &pos);
        //     omp_set_num_threads(n_thread);
        //     if (pos < arg.size()) {
        //         std::cerr << "Trailing characters after number: " << arg << '\n';
        //     }
        //     } catch (std::invalid_argument const &ex) {
        //             std::cerr << "Invalid number: " << argv[4] << '\n';
        //             exit(1);
        //     } catch (std::out_of_range const &ex) {
        //             std::cerr << "Number out of range: " << argv[4] << '\n';
        //             exit(1);
        //     }
        // }
    } else {
        cout << "usage - mpirun -np 1 ./mainProgram case n_repeat l_show(optional) n_thread(optional)" << endl;
        cout << "case : case to test (string : \"" << casenames[0] << "\", \"" << casenames[1] << "\" or \"" << casenames[2] <<"\")" << endl;
        cout << "n_repeat : number of time case should be run (int)" << endl;
        cout << "l_show : write results in console (optional bool, default value : False, \"True\", \"true\" and \"T\" set to True) - forces n_repeat to 1" << endl;
        // cout << "n_thread : (int)" << endl;
        exit(1);
    }
	auto end = std::chrono::steady_clock::now();
    auto mid = std::chrono::steady_clock::now();
    Inference* net;
    Inference* vect;
    std::vector<std::vector<double>> inputs;
    size_t n_cases, input_size;
    //
    if (c == "ReLU") {
        net = new SnglInference();
        net->ImportNN("NNs/5x20_ReLU/model.h5", "NNs/5x20_ReLU/architecture.json");
        vect = new VectInference();
        vect->ImportNN("NNs/5x20_ReLU/model.h5", "NNs/5x20_ReLU/architecture.json");
        n_cases = sizeof(inputs_ReLU) / sizeof(inputs_ReLU[0]);
        input_size = sizeof(inputs_ReLU[0]) / sizeof(double);
        inputs.resize(n_cases);
        for(size_t i = 0; i < n_cases; i++) {
            inputs[i].resize(input_size);
            memcpy(&inputs[i][0], &inputs_ReLU[i][0], input_size * sizeof(double));    
        }
    } else if (c == "tanh") {
        net = new SnglInference();
        net->ImportNN("NNs/2x80_tanh/model.h5", "NNs/2x80_tanh/architecture.json", "", "dense_layer_", "", "output_layer");
        vect = new VectInference();
        vect->ImportNN("NNs/2x80_tanh/model.h5", "NNs/2x80_tanh/architecture.json", "", "dense_layer_", "", "output_layer");
        n_cases = sizeof(inputs_tanh) / sizeof(inputs_tanh[0]);
        input_size = sizeof(inputs_tanh[0]) / sizeof(double);
        inputs.resize(n_cases);
        for(size_t i = 0; i < n_cases; i++) {
            inputs[i].resize(input_size);
            memcpy(&inputs[i][0], &inputs_tanh[i][0], input_size * sizeof(double));
        }
    } else if (c == "swish") {
        net = new SnglInference();
        net->ImportNN("NNs/250ResBlock_swish/model.h5", "NNs/250ResBlock_swish/architecture.json", "backbone", "", "hidden_residual_block_", "output_layer");
        vect = new VectInference();
        vect->ImportNN("NNs/250ResBlock_swish/model.h5", "NNs/250ResBlock_swish/architecture.json", "backbone", "", "hidden_residual_block_", "output_layer");
        n_cases = sizeof(inputs_swish) / sizeof(inputs_swish[0]);
        input_size = sizeof(inputs_swish[0]) / sizeof(double);
        inputs.resize(n_cases);
        for(size_t i = 0; i < n_cases; i++) {
            inputs[i].resize(input_size);
            memcpy(&inputs[i][0], &inputs_swish[i][0], input_size * sizeof(double));
        }
    }
    //
    double* data = new double[input_size];
    double* result;
    size_t output_size = net->n_output_ai;
    //
    // Non-vectorized runs with Inference
    //
    uint j = 0;
    ulong sum_inf = 0;
    auto init = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n_repeat*n_cases; i++) {
        for (size_t k = 0; k < input_size; k++) data[k] = inputs[j][k];
        mid = std::chrono::steady_clock::now();
        result = net->run_ai(data);
        end = std::chrono::steady_clock::now();
        if (show_results) {
            cout << "results :" << result[0];
            for (size_t ires = 1; ires < output_size; ires++) cout << " - " << result[ires];
            cout << endl;
        }
        j++;
        if (j == n_cases) j = 0;
        sum_inf += std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count();
    }
    end = std::chrono::steady_clock::now();
    sum_inf /= 1000000;
    cout << "time (total/inference): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - init).count() << "ms / " << sum_inf << "ms" << endl << endl;
    //
    //
    // Non-vectorized runs with VectInference
    //
    j = 0;
    sum_inf = 0;
    init = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n_repeat*n_cases; i++) {
        for (size_t k = 0; k < input_size; k++) data[k] = inputs[j][k];
        mid = std::chrono::steady_clock::now();
        result = vect->run_ai(data,1);
        end = std::chrono::steady_clock::now();
        if (show_results) {
            cout << "results :" << result[0];
            for (size_t ires = 1; ires < output_size; ires++) cout << " - " << result[ires];
            cout << endl;
        }
        j++;
        if (j == n_cases) j = 0;
        sum_inf += std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count();
    }
    end = std::chrono::steady_clock::now();
    sum_inf /= 1000000;
    cout << "time (total/inference): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - init).count() << "ms / " << sum_inf << "ms" << endl << endl;
    delete[] data;
    //
    // Vectorized run
    //
    j = 0;
    data = new double[input_size*n_cases*n_repeat];
    double* result_vect;
    size_t n_vect_runs = (show_results)? 1 : 5; // hard-coded 5 runs for now
    for (size_t irun = 0; irun < n_vect_runs; irun++) { // First run is longer because of memory allocation --> several runs show actual performances
        init = std::chrono::steady_clock::now();
        for (size_t k = 0; k < input_size*n_repeat; k++) {
            for (size_t i = k*n_cases; i < k*n_cases+n_cases;i++) data[i] = inputs[i-k*n_cases][j];
            j++;
            if (j == input_size) j = 0;
        }
        mid = std::chrono::steady_clock::now();
        result_vect = vect->run_ai(data,n_repeat*n_cases);
        end = std::chrono::steady_clock::now();
        //
        if (show_results) {
            for (size_t i = 0; i < n_repeat*n_cases; i++) {
                cout << "results :" << result_vect[i];
                for (size_t ires = 1; ires < output_size; ires++) cout << " - " << result_vect[i+ires*n_repeat*n_cases];
                cout << endl;
            }
        }
        //
        cout << "time (total/inference): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - init).count() << "ms / " << std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count() << "ms"  << endl;
    }
    delete[] data;
    //
    return 0;
}