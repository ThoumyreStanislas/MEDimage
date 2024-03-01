from string import Template

ngtdm_kernel = Template("""
#include <stdio.h>
#include <math.h>
#include <iostream>
                        
# define MAX_SIZE ${max_vol}
# define FILTER_SIZE ${filter_size}
                        
// Function flatten a 3D matrix into a 1D vector
__device__ float * reshape(float(*matrix)[FILTER_SIZE][FILTER_SIZE]) {
    //size of array
    const int size = FILTER_SIZE* FILTER_SIZE* FILTER_SIZE;
    float flattened[size];
    int index = 0;
    for (int i = 0; i < FILTER_SIZE; ++i) {
		for (int j = 0; j < FILTER_SIZE; ++j) {
			for (int k = 0; k < FILTER_SIZE; ++k) {
				flattened[index] = matrix[i][j][k];
                index++;
			}
		}
	}
    return flattened;
}

// Function to perform histogram equalisation of the ROI imaging intensities
__device__ void discretize(float vol_quant_re[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE], 
                            float max_val, float min_val=${min_val}) {
    
    // PARSING ARGUMENTS
    float n_q = ${n_q};
    const char* discr_type = "${discr_type}";

    // DISCRETISATION
    if (discr_type == "FBS") {
        float w_b = n_q;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                for (int k = 0; k < FILTER_SIZE; k++) {
                    float value = vol_quant_re[i][j][k];
                    if (!isnan(value)) {
                        vol_quant_re[i][j][k] = floorf((value - min_val) / w_b) + 1.0;
                    }
                }
            }
        }
    }
    else if (discr_type == "FBN") {
        float w_b = (max_val - min_val) / n_q;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                for (int k = 0; k < FILTER_SIZE; k++) {
                    float value = vol_quant_re[i][j][k];
                    if (!isnan(value)) {
                        vol_quant_re[i][j][k] = floorf(n_q * ((value - min_val) / (max_val - min_val))) + 1.0;
                        if (value == max_val) {
							vol_quant_re[i][j][k] = n_q;
						}
                    }
                }
            }
        }
    }
    else {
        printf("ERROR: discretization type not supported");
        assert(false);
    }
}
                        

extern "C"
__global__ void ngtdm_filter_global(
    float vol[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}][5],
    float vol_copy[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    bool distCorrection = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ${shape_volume_0} && j < ${shape_volume_1} && k < ${shape_volume_2} && i >= 0 && j >= 0 && k >= 0) {
        // pad size
        const int padd_size = (FILTER_SIZE - 1) / 2;

        // size vol
        const int size_x = ${shape_volume_0};
        const int size_y = ${shape_volume_1};
        const int size_z = ${shape_volume_2};

        // skip all calculations if vol at position i,j,k is nan
        if (!isnan(vol_copy[i][j][k])) {
            // get submatrix
            float sub_matrix[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] = {NAN};
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                    if ((i - padd_size + idx_i) >= 0 && (i - padd_size + idx_i) < size_x && 
                        (j - padd_size + idx_j) >= 0 && (j - padd_size + idx_j) < size_y &&
                        (k - padd_size + idx_k) >= 0 && (k - padd_size + idx_k) < size_z) {
                            sub_matrix[idx_i][idx_j][idx_k] = vol_copy[i - padd_size + idx_i][j - padd_size + idx_j][k - padd_size + idx_k];
                        }
                    }
                }
            }

            // get the maximum value of the submatrix
            float max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }
            
            // compute NGTDM features
            float features[5] = { 0.0 };
            computeNGTDMFeatures(sub_matrix, features, max_vol, false);
            
            // Copy NGTDM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                for (int idx = 0; idx < 25; ++idx) {
                    vol[i][j][k][idx] = features[idx];
                }
            }
        }
    }
}

extern "C"
__global__ void ngtdm_filter_local(
    float vol[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}][5],
    float vol_copy[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    bool distCorrection = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ${shape_volume_0} && j < ${shape_volume_1} && k < ${shape_volume_2} && i >= 0 && j >= 0 && k >= 0) {
        // pad size
        const int padd_size = (FILTER_SIZE - 1) / 2;

        // size vol
        const int size_x = ${shape_volume_0};
        const int size_y = ${shape_volume_1};
        const int size_z = ${shape_volume_2};

        // skip all calculations if vol at position i,j,k is nan
        if (!isnan(vol_copy[i][j][k])) {
            // get submatrix
            float sub_matrix[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] = {NAN};
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                    if ((i - padd_size + idx_i) >= 0 && (i - padd_size + idx_i) < size_x && 
                        (j - padd_size + idx_j) >= 0 && (j - padd_size + idx_j) < size_y &&
                        (k - padd_size + idx_k) >= 0 && (k - padd_size + idx_k) < size_z) {
                            sub_matrix[idx_i][idx_j][idx_k] = vol_copy[i - padd_size + idx_i][j - padd_size + idx_j][k - padd_size + idx_k];
                        }
                    }
                }
            }

            // get the maximum value of the submatrix
            float max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }
            // get the minimum value of the submatrix if discr_type is FBN
            float min_val = 3.40282e+38;
            if ("${discr_type}" == "FBN") {
                for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                    for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                        for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                            min_val = min(min_val, sub_matrix[idx_i][idx_j][idx_k]);
                        }
                    }
                }
                discretize(sub_matrix, max_vol, min_val);
            }

            // If FBS discretize the submatrix with user set minimum value
            else{
                discretize(sub_matrix, max_vol);
            }
                                                          
            // get the maximum value of the submatrix after discretization
            max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }

            // compute NGTDM features
            float features[5] = { 0.0 };
            computeNGTDMFeatures(sub_matrix, features, max_vol, false);
            
            // Copy NGTDM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                for (int idx = 0; idx < 25; ++idx) {
                    vol[i][j][k][idx] = features[idx];
                }
            }
        }
    }
}
                        """)