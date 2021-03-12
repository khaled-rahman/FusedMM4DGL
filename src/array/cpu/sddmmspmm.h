#ifndef DGL_ARRAY_CPU_SDDMMSPMM_H_
#define DGL_ARRAY_CPU_SDDMMSPMM_H_
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <math.h>
#include <iostream>
#include <omp.h>
#include "../selector.h"
#include "sddmm.h"
#include "./fusedmm/fusedMM.h"

using namespace std;

namespace dgl {
namespace aten {
namespace cpu {

template <typename IdType, typename DType, typename Op,
int LhsTarget = 0, int RhsTarget = 2>
void SDDMMSPMMCsr(const BcastOff& bcast,
const CSRMatrix& csr,
              NDArray lhs, NDArray rhs, NDArray out, int32_t imsg) {

const IdType* indptr = csr.indptr.Ptr<IdType>();
const IdType* indices = csr.indices.Ptr<IdType>();
const IdType* edges = csr.data.Ptr<IdType>();
const DType* X = lhs.Ptr<DType>();
const DType* Y = rhs.Ptr<DType>();
const int32_t dim = bcast.out_len;
// lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;

DType* O = out.Ptr<DType>();
bool dataType = 1;

//int32_t imsg;
//imsg = VOP_COPY_RHS | ROP_DOT | SOP_UDEF | VSC_MUL | AOP_ADD;

fusedMM_csr(imsg, csr.num_rows, csr.num_rows, dim, 1.0, 0.0, csr.num_rows, csr.num_rows, NULL, (const long int*)indices, (const long int*)indptr, (const long int*)indptr+1, (const float*)X, dim, (const float*)Y, dim, 1.0, (float*)O, dim);

}	
}
}
}
#endif 
