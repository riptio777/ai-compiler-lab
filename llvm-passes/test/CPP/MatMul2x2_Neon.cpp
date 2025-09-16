#include <arm_neon.h>
#include <iostream>
#include <ostream>

void vec_mul(float32x2_t v1, float32x2_t v2, float32x2_t res) {
    res = vdup_n_f32(0.0f);
    res = vmul_f32(v1, v2);
}

void mat_mul(float a[4][4], float b[4][4], float c[4][4], int N) {
    // 2 x 2 tiling
    float32x2_t vaCol = vdup_n_f32(0.0f);
    float32x2_t vb = vdup_n_f32(0.0f);
    float32x2_t vc0 = vdup_n_f32(0.0f);
    float32x2_t vc1 = vdup_n_f32(0.0f);

    // current stride = 2
    // need to make sure that N is a multiple of stride
    
    for (int i = 0; i < N; i+=2) {
        for (int j = 0; j < N; j+=2) {
            vc0 = vdup_n_f32(0.0f);
            vc1 = vdup_n_f32(0.0f);
            for (int k = 0; k < N; k++) {
                vaCol = vset_lane_f32(a[i][k], vaCol, 0);
                vaCol = vset_lane_f32(a[i+1][k], vaCol, 1);

                // splat lanes of elements of b across lanes
                float32x2_t vb0 = vdup_n_f32(b[k][j]);
                float32x2_t vb1 = vdup_n_f32(b[k][j+1]);

                vc0 = vfma_f32(vc0, vaCol, vb0);
                vc1 = vfma_f32(vc1, vaCol, vb1);
            }

            c[i][j] = vget_lane_f32(vc0, 0);
            c[i+1][j] = vget_lane_f32(vc0, 1);
            c[i][j+1] = vget_lane_f32(vc1, 0);
            c[i+1][j+1] = vget_lane_f32(vc1, 1);
        }
    }
}

int main() {
    float a[2] = {1.0f, 2.0f};
    // or use neon registers directly
    // float32x2_t v = {1.0f, 2.0f};

    float b[2] = {3.0f, 4.0f};
    float32x2_t v1 = vld1_f32(a);
    float32x2_t v2 = vld1_f32(b);

    float c[2];
    float32x2_t res;
    vec_mul(v1, v2, res);

    vst1_f32(c, res);
/*
    for(int i = 0; i < 2; i++) {
        std::cout << c[i] << " " << std::endl;
    }
*/
    float A[4][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12},
        {13,14, 15, 16}
    };

// Matrix B (4x4)
    float B[4][4] = {
        {17, 18, 19, 20},
        {21, 22, 23, 24},
        {25, 26, 27, 28},
        {29, 30, 31, 32}
    };

    // Matrix C = A * B (4x4)
    float C[4][4] = {
        {250, 260, 270, 280},
        {618, 644, 670, 696},
        {986,1028,1070,1112},
        {1354,1412,1470,1528}
    };

    float res_mat[4][4] {0.0f};
    mat_mul(A, B, res_mat, 4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << res_mat[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}