// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "inference.h"

Inference::~Inference() {
}

void Inference::ImportNN(const string& model_path, const string& activation_path)
{
    input_layer = "";
    hidden_prefix = "dense_";
    res2_prefix = "";
    output_layer = "";
    //
    read_kernel_bias(model_path);
    read_activations(activation_path);
    //
    n_input_ai = topology[0];
    n_output_ai = topology.back();
    log_threshold = 1.0; // SetLogThreshold and SetBctCst TBD
    bct_constant = 1.0;
    rbct = 1.0 / bct_constant;
    // for debug purpose
    // for (size_t i = 0; i < topology.size(); i++) {
    //     cout << "topology[" << i << "] = " << topology[i] << endl;
    //     //
    //     if (i > 0) {
    //         cout << "bias[" << i - 1 << "] = " << *bias[i-1] << endl;
    //         cout << "weights[" << i - 1 << "] = " << (*weights[i-1]).transpose() << endl;
    //     }
    //     cout << endl << endl;
    // }
}

void Inference::ImportNN(const string& model_path, const string& activation_path, const string& input_layer_name, const string& hidden_layers_prefix, const string& resblock_prefix, const string& output_layer_name)
{
    input_layer = input_layer_name;
    hidden_prefix = hidden_layers_prefix;
    res2_prefix = resblock_prefix;
    output_layer = output_layer_name;
    //
    read_kernel_bias(model_path);
    read_activations(activation_path);
    //
    n_input_ai = topology[0];
    n_output_ai = topology.back();
    log_threshold = 1.0; // SetLogThreshold and SetBctCst TBD
    bct_constant = 1.0;
    rbct = 1.0 / bct_constant;
}

void Inference::read_kernel_bias(const string& path) {
    hid_t h5Model, grp, dset;
    //
    // Open .h5 file
    //
	try {
        h5Model = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    }
    catch (...) {
        string msg("Could not open HDF5 file containing NN\n");
        throw msg;
    }
    //
    // Input layer
    //
    size_t nlayer = 1;
    string grp_path;
    string res2_path;
    if (input_layer!="") { // Hard-coded as dense layer for now
        grp_path = input_layer+"/"+input_layer;
        grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

        AddDense();

        dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
        add_weight_from_dataset(dset);
        H5Dclose(dset);

        dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
        add_bias_from_dataset(dset);
        H5Dclose(dset);
        H5Gclose(grp);
    }
    //
    // Loop over layers
    //
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    string layer_str = std::to_string(nlayer);
    grp_path = hidden_prefix+layer_str+"/"+hidden_prefix+layer_str;
    res2_path = res2_prefix+layer_str+"/"+res2_prefix+layer_str;
    bool has_more = true;
    while (has_more) {
        if (H5Lexists(h5Model,grp_path.c_str(),H5P_DEFAULT) > 0) { // Dense layer
            grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

            dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
            add_weight_from_dataset(dset);
            H5Dclose(dset);

            dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
            add_bias_from_dataset(dset);
            H5Dclose(dset);
            H5Gclose(grp);

            AddDense();
        } else if (H5Lexists(h5Model,res2_path.c_str(),H5P_DEFAULT) > 0) { // ResBlock (1 dense + 1 res2)
            string local_path = res2_path + "/hidden_unit_0"; // add a dense layer
            grp = H5Gopen(h5Model,local_path.c_str(),H5P_DEFAULT);

            dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
            add_weight_from_dataset(dset);
            H5Dclose(dset);

            dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
            add_bias_from_dataset(dset);
            H5Dclose(dset);
            H5Gclose(grp);
            AddDense();

            local_path = res2_path + "/hidden_unit_1"; // add a res2 layer
            grp = H5Gopen(h5Model,local_path.c_str(),H5P_DEFAULT);

            dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
            add_weight_from_dataset(dset);
            H5Dclose(dset);

            dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
            add_bias_from_dataset(dset);
            H5Dclose(dset);
            H5Gclose(grp);
            AddResBlock2();
        } else { // neither Dense nor ResBlock --> end search
            has_more = false;
        }

        nlayer+=1;
        layer_str = std::to_string(nlayer);
        grp_path = hidden_prefix+layer_str+"/"+hidden_prefix+layer_str;
        res2_path = res2_prefix+layer_str+"/"+res2_prefix+layer_str;
    }
    //
    // Output layer
    //
    if (output_layer != "") { // Hard-coded as dense layer for now
        string grp_path = output_layer+"/"+output_layer;
        grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

        dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
        add_weight_from_dataset(dset);
        H5Dclose(dset);

        AddDense();

        dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
        add_bias_from_dataset(dset);
        H5Dclose(dset);
        H5Gclose(grp);
    }
}

void Inference::read_activations(const string& path) {
    std::ifstream jFile(path);
    if(!jFile.is_open()) 
        throw std::runtime_error("Inference - Unable to open json data file : " + path);

    std::stringstream contents;
    contents << jFile.rdbuf();

    rapidjson::Document doc;
    doc.Parse(contents.str().c_str());
    //
    try {
        bool check = doc["config"].HasMember("layers");
    }
    catch (...) {
        string msg("Could not find \"layers\" object in NN json attributes \n");
        throw msg;
    }
    rapidjson::Value& layers = doc["config"]["layers"];
    assert(layers.IsArray());
    for(size_t i = 0; i < layers.Size(); i++) {
        string class_name(layers[i]["class_name"].GetString());
        // cout << layers[i]["class_name"].GetString() << endl; // for debug purpose
        //
        if (class_name == "Dense") { // to be checked (might depend on NN generation method)
            string act_func(layers[i]["config"]["activation"].GetString());
            // cout << act_func << endl; // for debug purpose
            //
            if (act_func == "tanh") {
                AddTanh();
            } else if (act_func == "relu") {
                AddReLU();
            } else if (act_func == "swish") {
                AddSwish();
            } else {
                AddId();
            }
        } else if (class_name == "ResidualBlock") {
            string act_func(layers[i]["config"]["activations"].GetString());
            // cout << act_func << endl; // for debug purpose
            //
            if (act_func == "tanh") {
                AddTanh();
                AddTanh();
            } else if (act_func == "relu") {
                AddReLU();
                AddReLU();
            } else if (act_func == "swish") {
                AddSwish();
                AddSwish();
            } else {
                AddId();
                AddId();
            }
        }
    }
}

void Inference::ImportNN_pt_custom(const string& model_path)
{
    input_layer = "";
    hidden_prefix = "dense_";
    res2_prefix = "";
    output_layer = "";
    //
    read_kernel_bias_act_pt_custom(model_path);
    //
    n_input_ai = topology[0];
    n_output_ai = topology.back();
    log_threshold = 1.0; // SetLogThreshold and SetBctCst TBD
    bct_constant = 1.0;
    rbct = 1.0 / bct_constant;
    // for debug purpose
    // for (size_t i = 0; i < topology.size(); i++) {
    //     cout << "topology[" << i << "] = " << topology[i] << endl;
    //     //
    //     if (i > 0) {
    //         cout << "bias[" << i - 1 << "] = " << *bias[i-1] << endl;
    //         cout << "weights[" << i - 1 << "] = " << (*weights[i-1]).transpose() << endl;
    //     }
    //     cout << endl << endl;
    // }
}

void Inference::ImportNN_pt_custom(const string& model_path, const string& input_layer_name, const string& hidden_layers_prefix, const string& resblock_prefix, const string& output_layer_name)
{
    input_layer = input_layer_name;
    hidden_prefix = hidden_layers_prefix;
    res2_prefix = resblock_prefix;
    output_layer = output_layer_name;
    //
    read_kernel_bias_act_pt_custom(model_path);
    //
    n_input_ai = topology[0];
    n_output_ai = topology.back();
    log_threshold = 1.0; // SetLogThreshold and SetBctCst TBD
    bct_constant = 1.0;
    rbct = 1.0 / bct_constant;
}

void Inference::read_kernel_bias_act_pt_custom(const string& path) {
    hid_t h5Model, grp, dset;
    //
    // Open .h5 file
    //
	try {
        h5Model = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    }
    catch (...) {
        string msg("Could not open HDF5 file containing NN\n");
        throw msg;
    }
    //
    // Input layer
    //
    size_t nlayer = 1;
    string grp_path;
    string res2_path;
    if (input_layer!="") { // Hard-coded as dense layer for now
        grp_path = input_layer;
        grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

        AddDense();

        dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
        add_weight_from_dataset(dset);
        H5Dclose(dset);

        dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
        add_bias_from_dataset(dset);
        H5Dclose(dset);
        H5Gclose(grp);
    }
    //
    // Loop over layers
    //
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    string layer_str = std::to_string(nlayer);
    grp_path = hidden_prefix+layer_str;
    res2_path = res2_prefix+layer_str;
    bool has_more = true;
    while (has_more) {
        if (H5Lexists(h5Model,grp_path.c_str(),H5P_DEFAULT) > 0) { // Dense layer
            grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

            // Read activation
            std::string activation = read_string_attribute(grp, "activation");

            dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
            add_weight_from_dataset(dset);
            H5Dclose(dset);

            dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
            add_bias_from_dataset(dset);
            H5Dclose(dset);
            H5Gclose(grp);

            AddDense();
            set_activation(activation);
        } else if (H5Lexists(h5Model,res2_path.c_str(),H5P_DEFAULT) > 0) { // ResBlock (1 dense + 1 res2)
            string local_path = res2_path + "/hidden_unit_0"; // add a dense layer
            grp = H5Gopen(h5Model,local_path.c_str(),H5P_DEFAULT);

            // Read activation
            std::string activation = read_string_attribute(grp, "activation");

            dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
            add_weight_from_dataset(dset);
            H5Dclose(dset);

            dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
            add_bias_from_dataset(dset);
            H5Dclose(dset);
            H5Gclose(grp);
            AddDense();

            local_path = res2_path + "/hidden_unit_1"; // add a res2 layer
            grp = H5Gopen(h5Model,local_path.c_str(),H5P_DEFAULT);

            dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
            add_weight_from_dataset(dset);
            H5Dclose(dset);

            dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
            add_bias_from_dataset(dset);
            H5Dclose(dset);
            H5Gclose(grp);
            AddResBlock2();

            set_activation(activation);
            set_activation(activation);
        } else { // neither Dense nor ResBlock --> end search
            has_more = false;
        }

        nlayer+=1;
        layer_str = std::to_string(nlayer);
        grp_path = hidden_prefix+layer_str;
        res2_path = res2_prefix+layer_str;
    }
    //
    // Output layer
    //
    if (output_layer != "") { // Hard-coded as dense layer for now
        string grp_path = output_layer;
        grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

        // Read activation
        std::string activation = read_string_attribute(grp, "activation");

        dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
        add_weight_from_dataset(dset);
        H5Dclose(dset);

        AddDense();
        set_activation(activation);

        dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
        add_bias_from_dataset(dset);
        H5Dclose(dset);
        H5Gclose(grp);
    }
}

std::string Inference::read_string_attribute(hid_t group, const std::string& attr_name)
{
    if (H5Aexists(group, attr_name.c_str()) <= 0)
        return "identity";  // default fallback

    hid_t attr = H5Aopen(group, attr_name.c_str(), H5P_DEFAULT);
    hid_t type = H5Aget_type(attr);
    hid_t space = H5Aget_space(attr);

    std::string value;
    
    // Check if it's a variable-length string
    if (H5Tis_variable_str(type)) {
        char* rdata = nullptr;
        H5Aread(attr, type, &rdata);
        if (rdata) {
            value = std::string(rdata);
            H5free_memory(rdata);  // Must free variable-length string
        }
    } else {
        // Fixed-length string
        size_t size = H5Tget_size(type);
        std::vector<char> buffer(size + 1, '\0');  // +1 for safety
        H5Aread(attr, type, buffer.data());
        value = std::string(buffer.data());
        
        // Clean up: remove trailing nulls and spaces
        value.erase(std::find_if(value.rbegin(), value.rend(), 
                    [](unsigned char ch) { return ch != '\0' && ch != ' '; }).base(), 
                    value.end());
    }

    H5Sclose(space);
    H5Tclose(type);
    H5Aclose(attr);
    
    return value;
}


void Inference::set_activation(const std::string& act)
{
    std::string act_lower = act;
    std::transform(act_lower.begin(), act_lower.end(), act_lower.begin(), ::tolower);

    if (act_lower == "relu")       AddReLU();
    else if (act_lower == "tanh")  AddTanh();
    else if (act_lower == "swish") AddSwish();
    else                     AddId();
}

void Inference::add_bias_from_dataset(hid_t& h5_dset) {
    size_t i = topology.size();
    hid_t dspace = H5Dget_space(h5_dset);
    double* tmp = new double[topology[i-1]];
    H5Dread(h5_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,tmp);
    H5Sclose(dspace);
    
    bias.push_back(RowVector(topology[i-1]));
    bias[i-2] = Eigen::Map<RowVector>(tmp,topology[i-1]);
    
    delete[] tmp;
}

void Inference::add_weight_from_dataset(hid_t& h5_dset) {
    hid_t dspace = H5Dget_space(h5_dset);
    hsize_t ndim = H5Sget_simple_extent_ndims(dspace);
    hsize_t* dims = new hsize_t[ndim];
    H5Sget_simple_extent_dims(dspace,dims,NULL);

    size_t i = topology.size();
    if (i == 0) {
        topology.push_back(dims[0]);
        AddLayer(topology[i]);
        i++;
    }
    topology.push_back(dims[1]);
    AddLayer(topology[i]);

    double* tmp = new double[dims[0]*dims[1]];
    H5Dread(h5_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,tmp);
    H5Sclose(dspace);
    weights.push_back(Matrix(topology[i - 1], topology[i]));
    Matrix mat = Eigen::Map<Matrix>(tmp, topology[i], topology[i - 1]);
    weights[i-1] = mat.transpose();
    //
    delete[] tmp;
    delete[] dims;
}