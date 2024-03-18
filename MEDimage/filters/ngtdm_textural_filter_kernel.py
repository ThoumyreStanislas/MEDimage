from string import Template

ngtdm_kernel = Template("""
#include <stdio.h>
#include <math.h>
#include <cassert>


# define MAX_SIZE ${max_vol}
# define FILTER_SIZE ${filter_size}
const int padd_size = (FILTER_SIZE - 1) / 2;
const int final_size = FILTER_SIZE + padd_size;
                                               
// Function to flatten a 3D matrix into a 1D vector
__device__ void reshape_3D_to_1D(float matrix[final_size][final_size][final_size], float flattend_1d[(final_size)*(final_size)*(final_size)]) {
    //size of array
    int index = 0;
    for (int i = 0; i < final_size; ++i) {
        for (int j = 0; j < final_size; ++j) {
            for (int k = 0; k < final_size; ++k) {
                flattend_1d[index] = matrix[i][j][k];
                index++;
            }
        }
    }
}
                        
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


// Function to convert a linear index into multidimensional indices based on the provided shapes.
__device__ void unravel_index(int index, int* shape, int ndim, int* indices) {
    int prod = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        indices[i] = index / prod % (shape[i]);
        prod *= shape[i];
    }
}

// Padding function 
__device__ void pad_array(float array[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE], int rows, int cols, int depth, int pad_width, float pad_value, float result[FILTER_SIZE + padd_size][FILTER_SIZE + padd_size][FILTER_SIZE + padd_size]) {
    int padded_rows = rows + 2 * pad_width;
    int padded_cols = cols + 2 * pad_width;
    int padded_depth = depth + 2 * pad_width;

    for (int i = 0; i < padded_rows; i++) {
        for (int j = 0; j < padded_cols; j++) {
            for (int k = 0; k < padded_depth; k++) {
                if (i < pad_width || i >= rows + pad_width || j < pad_width || j >= cols + pad_width || k < pad_width || k >= depth + pad_width) {
                    // Remplir les zones de padding avec la valeur de padding
                    result[i][j][k] = pad_value;
                }
                else {
                    // Copier les valeurs du tableau d'origine dans la zone non rembourrée
                    result[i][j][k] = array[i - pad_width][j - pad_width][k - pad_width];
                }
            }
        }
    }
}

// Converts a 3D array into a 1D array, storing indices of non-NaN elements.
__device__ void indices_elements_non_nan(float tableau[final_size][final_size][final_size], float indices[final_size*final_size*final_size], int index_return[1]) {
    // Initialiser la taille maximale du tableau d'indices
    float tableau_1d [final_size*final_size*final_size];

    int index = 0;

    reshape_3D_to_1D(tableau, tableau_1d);

    // Parcourir le tableau et stocker les indices des éléments non nuls
    for (int i = 0; i < final_size*final_size*final_size; ++i) {
        if (!isnan(tableau_1d[i])) {
            indices[index] = i;
            index += 1;
        }
    }

    index_return[0] = index;
}
                       

__device__ void getNGTDMmatrix(
        float ROIonly[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE],
        float ngtdm[MAX_SIZE],
        float count_valid[MAX_SIZE],
        float proba[MAX_SIZE],
        bool distCorrection = true){

    float paddedROI[final_size][final_size][final_size];

    pad_array(ROIonly, FILTER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, NAN ,paddedROI);

    float indices_valid[final_size*final_size*final_size] = {NAN};

    int n_valid_temp[1];
    int n_valid = {0};


    indices_elements_non_nan(paddedROI, indices_valid, n_valid_temp);
    n_valid = n_valid_temp[0];

    int pos_valid[3][FILTER_SIZE*FILTER_SIZE*FILTER_SIZE];
    int pos_valid_i[3];



    for (int i = 0; i < n_valid+1; i++){
        if (indices_valid[i] != 0){
            int indice_i = indices_valid[i];
            int shape[] = {final_size , final_size , final_size};
            unravel_index(indice_i ,shape, 3, pos_valid_i);
            pos_valid[0][i] = pos_valid_i[0];
            pos_valid[1][i] = pos_valid_i[1];
            pos_valid[2][i] = pos_valid_i[2];
        }
    }

    
    for (int i = 0; i < n_valid; i++) {
        int n = 0;
        int value;
        float neighbours[26] ={NAN};
        int count = 0;
        int x_i = pos_valid[0][i];
        int y_i = pos_valid[1][i];
        int z_i = pos_valid[2][i];
        value = paddedROI[x_i][y_i][z_i];
        for (int x = x_i - 1; x < x_i + 2; x++) {
            for (int y = y_i - 1; y < y_i + 2; y++) {
                for (int z = z_i - 1; z < z_i + 2; z++) {
                    if (!(x==x_i && y==y_i && z==z_i)){
                        neighbours[n] = paddedROI[x][y][z];
                        if (!isnan(neighbours[n])){
                            count++ ;
                        }
                        n++;
                    }
                }
            }
        }
            if (count !=0 ){
                int somme = 0;
                for (int n = 0; n < 27; n++){
                    if ((!isnan(neighbours[n]))){
                        somme += neighbours[n];
                    }
                }
                ngtdm[value-1] += fabs(float(value) - float(somme) / float(count));
                count_valid[value-1]++;
            }
        }


    float n_tot = 0;
    for (int i = 0; i < MAX_SIZE; i++){
        n_tot += count_valid[i];
    }
    for (int i = 0; i < MAX_SIZE; i++){
        proba[i] = count_valid[i]/n_tot;
    }
}
                        
__device__ void coarseness(float probability[MAX_SIZE], float ngtdm[MAX_SIZE], float coarseness[1]){
    float coarseness_temp = powf(10, 6);
    float val = 0.0;
    for (int i = 1; i < int(MAX_SIZE+1); i++) {
        val += probability[i-1] * ngtdm[i-1];
    }
    if (!(val == 0.0)) {
        coarseness_temp = 1/val;
    }
    coarseness[0] = coarseness_temp;
}

__device__ void contrast(float probability[MAX_SIZE], float ngtdm[MAX_SIZE], float count_valid[MAX_SIZE], float contrast[1]){
    float contrast_temp = 0.0;
    float n_tot = 0.0;
    float ngtdm_total = 0.0;
    int n_valid = 0;
    for (int h = 0; h < int(MAX_SIZE); h++){
        n_tot += count_valid[h];
        ngtdm_total += ngtdm[h];
        if (count_valid[h] != 0) {
            n_valid++;
        }}
    if (n_valid == 1) {
         contrast_temp = 0.0;
    } else{
        for (int i = 0; i < MAX_SIZE; ++i) {
            for (int j = 0; j < MAX_SIZE; ++j) {
                contrast_temp +=  probability[i] * probability[j] * powf( i-j, 2);
            }
        }

        contrast_temp = contrast_temp * ngtdm_total/(n_valid*(n_valid-1)* n_tot);
    }
    contrast[0] = contrast_temp;
}

__device__ void busyness(float probability[MAX_SIZE], float ngtdm[MAX_SIZE], float count_valid[MAX_SIZE], float busyness[1]){
    float busyness_temp = 0.0;
    float busyness_temp_1 = 0.0;
    float n_valid = 0.0;
    float val = 0.0;
    for (int h = 0; h < int(MAX_SIZE); h++){
        if (count_valid[h] != 0) {
            n_valid++;
        }
        val += probability[h] * ngtdm[h];
    }
    if (n_valid == 1) {
        busyness[0] = 0.0;
    }
    else {
        for (int i = 0; i < int(MAX_SIZE); ++i) {
            for (int j = 0; j < int(MAX_SIZE); ++j) {
                if(!( probability[i]==0 || probability[j] == 0)) {
                    busyness_temp += fabsf(float(i+1) * probability[i] - float(j+1) * probability[j]);
                }
            }
        }
        busyness_temp_1 = val / busyness_temp;
        busyness[0] = busyness_temp_1;
    }
}

__device__ void complexity(float probability[MAX_SIZE], float ngtdm[MAX_SIZE], float count_valid[MAX_SIZE], float complexity[1]){
    float complexity_temp = 0.0;
    float n_tot = 0.0;
    for (int h = 0; h < int(MAX_SIZE); h++) {
        n_tot += count_valid[h];
    }
    for (int i = 0; i < MAX_SIZE; ++i) {
        for (int j = 0; j < MAX_SIZE; ++j) {
            if(!( probability[i]==0 || probability[j] == 0)) {
            complexity_temp += fabsf(float(i-j)) *(probability[i] * ngtdm[i] + probability[j] * ngtdm[j]) / (n_tot*(probability[i] + probability[j]));
        }}
    }
    complexity[0] = complexity_temp;
}

__device__ void strength(float probability[MAX_SIZE], float ngtdm[MAX_SIZE], float strength[1]) {
    float strength_temp = 0.0;
    float ngtdm_total = 0.0;
    for (int h = 0; h < int(MAX_SIZE); h++) {
        ngtdm_total += ngtdm[h];
    }
    if (ngtdm_total == 0) {
        strength[0] = 0;
    } else {
        for (int i = 1; i < MAX_SIZE; ++i) {
            for (int j = 1; j < MAX_SIZE; ++j) {
                if (!(probability[i] == 0 || probability[j] == 0)) {
                    strength_temp += (probability[i] + probability[j]) * powf(float(i - j), 2);
                }
            }
            strength[0] = strength_temp / ngtdm_total;
        }
    }
}

__device__ void computeNGTDMFeatures(float ROIonly[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE], float features[5]){
    float NGTDM[MAX_SIZE] = {0};
    float count_valid[MAX_SIZE] = {0};
    float probability[MAX_SIZE] = {0};
    getNGTDMmatrix(ROIonly, NGTDM, count_valid, probability, true);

    float coarseness_result[1]= {0.0};
    coarseness(probability, NGTDM, coarseness_result);
    features[0] = coarseness_result[0];

    float contrast_result[1] = {0.0};
    contrast(probability,NGTDM, count_valid,contrast_result);
    features[1] = contrast_result[0];

    float busyness_result[1] = {0.0};
    busyness(probability, NGTDM, count_valid, busyness_result);
    features[2] = busyness_result[0];

    float complexity_result[1] = {0.0};
    complexity(probability, NGTDM, count_valid, complexity_result);
    features[3] = complexity_result[0];

    float strength_result[1] = {0.0};
    strength(probability, NGTDM, strength_result);
    features[4] = strength_result[0];

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
            computeNGTDMFeatures(sub_matrix, features);

            // Copy NGTDM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                for (int idx = 0; idx < 5; ++idx) {
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
                //!!! ON A AJOUTER MIN_VAL !!!
                discretize(sub_matrix, max_vol, min_val);
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
            computeNGTDMFeatures(sub_matrix, features);

            // Copy NGTDM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                for (int idx = 0; idx < 5; ++idx) {
                    vol[i][j][k][idx] = features[idx];
                }
            }
        }
    }
}  
                            
                         """)