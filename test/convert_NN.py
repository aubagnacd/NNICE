

import h5py

def flatten_and_add_activation(input_path, output_path, activation):
    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        
        for layer_name in f_in.keys():
            print(f"Processing: {layer_name}")
            
            # 1. Determine Activation Attribute
            if layer_name == "output_layer":
                act_attr = "Id"
            else:
                act_attr = activation

            # 2. Check for Redundancy structure: layer_X/layer_X/...
            # We check if a child exists with the same name as the parent
            if isinstance(f_in[layer_name], h5py.Group) and layer_name in f_in[layer_name]:
                
                # Create the clean parent group in the new file
                new_group = f_out.create_group(layer_name)
                
                # Get the redundant inner group (e.g., dense_1/dense_1)
                redundant_inner = f_in[layer_name][layer_name]
                
                # Loop through contents of the redundant layer and move them up
                for item_name in redundant_inner.keys():
                    
                    # CASE 1: dense/dense/bias
                    # item_name is 'bias', it is copied to new_group/bias
                    
                    # CASE 2: dense/dense/hidden_unit/bias
                    # item_name is 'hidden_unit' (which is a Group).
                    # h5py copies the Group and ALL its children automatically.
                    # Result: new_group/hidden_unit/bias
                    
                    source_item = redundant_inner[item_name]
                    f_in.copy(source_item, new_group, name=item_name)
                
                # Add attribute to the newly created group
                new_group.attrs['activation'] = act_attr
                
            else:
                # Structure is already flat or unknown, copy as-is
                print(f"  -> Structure OK, copying directly.")
                f_in.copy(f_in[layer_name], f_out, name=layer_name)
                
                # Add attribute
                if isinstance(f_out[layer_name], h5py.Group):
                    f_out[layer_name].attrs['activation'] = act_attr

    print(f"\nSuccess. Cleaned file saved to: {output_path}")


# --- Usage ---
flatten_and_add_activation('NNs/2x80_tanh/model.h5', 'NNs/2x80_tanh/model_pt_custom.h5', activation="Tanh")
flatten_and_add_activation('NNs/5x20_ReLU/model.h5', 'NNs/5x20_ReLU/model_pt_custom.h5', activation="ReLU")
flatten_and_add_activation('NNs/250ResBlock_swish/model.h5', 'NNs/250ResBlock_swish/model_pt_custom.h5', activation="Swish")










