#ifndef DGL_ARRAY_CPU_FUSEDMMH_H_
#define DGL_ARRAY_CPU_FUSEDMMH_H_
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <math.h>
#include <iostream>
#include <omp.h>
#include "../selector.h"
#include "sddmm.h"
//#include "./FusedMM/kernels/include/kernels.h"
#include "./FusedMM/fusedMM.h"
//#include "fusedMM.h"
#define SM_TABLE_SIZE 2048
#define SM_BOUND 5.0
#define SM_RESOLUTION SM_TABLE_SIZE/(2.0 * SM_BOUND)

using namespace std;

VALUETYPE *SM_TABLE;
void uinit_SM_TABLE()
{
   VALUETYPE x;
   SM_TABLE = (VALUETYPE*)malloc(SM_TABLE_SIZE*sizeof(VALUETYPE));
   assert(SM_TABLE);
   for(INDEXTYPE i = 0; i < SM_TABLE_SIZE; i++)
   {
      x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
      SM_TABLE[i] = 1.0 / (1 + exp(-x));
   }
}
inline VALUETYPE uscale_SM(VALUETYPE val)
{
   VALUETYPE sval;
   /* hopefully compiler will figure out and replace it max min instruction */
   sval = (val > SM_BOUND) ? SM_BOUND : val;
   sval = (val < -SM_BOUND) ? -SM_BOUND : val;
   return(sval);
}
VALUETYPE ufast_SM(VALUETYPE v)
{
   if (v > SM_BOUND) return 1.0;
   else if (v < -SM_BOUND) return 0.0;
   return SM_TABLE[(INDEXTYPE)((v + SM_BOUND) * SM_RESOLUTION)];
}

int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE &out)
{
   out = 1.0 - ufast_SM(val);
   return FUSEDMM_SUCCESS_RETURN;
}

VALUETYPE scale(VALUETYPE v){
        if(v > SM_BOUND) return SM_BOUND;
        else if(v < -SM_BOUND) return -SM_BOUND;
        return v;
}

namespace dgl {
namespace aten {
namespace cpu {

template <typename DType>
DType scale(DType v){
	if(v > SM_BOUND) return SM_BOUND;
        else if(v < -SM_BOUND) return -SM_BOUND;
        return v;
}

template <typename DType>
DType fast_SM(DType v, DType *sm_table){
        if(v > SM_BOUND) return 1.0;
        else if(v < -SM_BOUND) return 0.0;
        return sm_table[(int)((v + SM_BOUND) * SM_RESOLUTION)];
}
template <typename IdType, typename DType>
void init_SM_TABLE(DType *sm_table){
        DType x;
        for(IdType i = 0; i < SM_TABLE_SIZE; i++){
                x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
                sm_table[i] = 1.0 / (1 + exp(-x));
        }
}

template <typename IdType, typename DType>
void FUSEDMMCsrTdist(const IdType *indptr, const IdType *indices, const IdType *edges,
                const DType *X, const DType *Y, DType *O, const IdType N, const int64_t dim) {

#pragma omp parallel for
for (IdType rid = 0; rid < N; ++rid) {
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
	DType T[dim];
        for (IdType j = row_start; j < row_end; ++j){
                const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                DType attrc = 0;
                for (int64_t k = 0; k < dim; ++k) {
			T[k] = X[iindex + k] - Y[jindex + k];
                        attrc += T[k] * T[k];
                }
                DType d1 = -2.0 / (1.0 + attrc);
                for (int64_t k = 0; k < dim; ++k) {
			T[k] = scale<DType>(T[k] * d1);
                        O[iindex+k] = O[iindex+k]  + T[k];
                }
        }
}

}

template <typename IdType, typename DType>
void FUSEDMMCsrSigmoid(const IdType *indptr, const IdType *indices, const IdType *edges, 
		const DType *X, const DType *Y, DType *O, const IdType N, const int64_t dim) {

DType *sm_table;
sm_table = static_cast<DType *> (::operator new (sizeof(DType[SM_TABLE_SIZE])));
init_SM_TABLE<IdType, DType>(sm_table);
cout << "Calling FUSEDMMCsrSigmoid..." << indptr[0] << ":" << indices[0] << ":" << edges[0] << ":" << X[0] << endl;
#pragma omp parallel for
for (IdType rid = 0; rid < N; ++rid){
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
	for (IdType j = row_start; j < row_end; ++j){
		const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                DType attrc = 0;
                for (int64_t k = 0; k < dim; ++k) {
                        attrc += X[iindex + k] * X[jindex + k];
                }
		DType d1 = 0.5;//fast_SM<DType>(attrc, sm_table);
		for (int64_t k = 0; k < dim; ++k) {
                        O[iindex+k] = O[iindex+k]  + (1.0 - d1) * X[jindex + k];
                }
        }
}

cout << "End of Calling FUSEDMMCsrSigmoid..." << endl;		

}


template <typename IdType, typename DType>
void FUSEDMMCsrGCN(const IdType *indptr, const IdType *indices, const IdType *edges,
                const DType *X, const DType *Y, DType *O, const IdType N, const int64_t dim) {

//cout << "Calling FUSEDMMCsrGCN..." << indptr[0] << ":" << indices[0] << ":" << edges[0] << ":" << X[0] << endl;
#pragma omp parallel for
for (IdType rid = 0; rid < N; ++rid){
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
        for (IdType j = row_start; j < row_end; ++j){
                const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                for (int64_t k = 0; k < dim; ++k) {
                        O[iindex+k] = O[iindex+k] + X[jindex + k];
                }
        }
}

//cout << "End of Calling FUSEDMMCsrGCN..." << endl;

}




template <typename IdType, typename DType>
void FUSEDMMCsr(const BcastOff& bcast,
const CSRMatrix& csr,
              NDArray lhs, NDArray rhs, NDArray out, int ftype = 1) {
const IdType* indptr = csr.indptr.Ptr<IdType>();
const IdType* indices = csr.indices.Ptr<IdType>();
const IdType* edges = csr.data.Ptr<IdType>();
const DType* X = lhs.Ptr<DType>();
const DType* Y = rhs.Ptr<DType>();
const int32_t dim = bcast.out_len;
//std::cout << "From FusedMMCsr function...(dim):" << dim << ", num_rows:" << csr.num_rows << endl;

DType* O = out.Ptr<DType>();

FUSEDMMCsrGCN<IdType, DType>(indptr, indices, edges, X, Y, O, csr.num_rows, dim);

//std::cout << "Returning from FusedMMCsr function..." << endl;

/*
if(ftype == 1){
   uinit_SM_TABLE();
   int32_t imsg;
   imsg = VOP_COPY_RHS | ROP_DOT | SOP_UDEF | VSC_MUL | AOP_ADD;
   fusedMM_csr(imsg, csr.num_rows, csr.num_rows, dim, 1.0, 0.0, csr.num_rows, csr.num_rows, NULL, (const long int*)indices, (const long int*)indptr, (const long int*)indptr+1, (const float*)X, dim, (const float*)Y, dim, 1.0, (float*)O, dim);
}else{
   int32_t imsg;
   imsg = VOP_SUBR | ROP_UDEF | SOP_UDEF | VSC_MUL | AOP_ADD;
   fusedMM_csr(imsg, csr.num_rows, csr.num_rows, dim, 1.0, 0.0, csr.num_rows, csr.num_rows, NULL, (const long int*)indices, (const long int*)indptr, (const long int*)indptr+1, (const float*)X, dim, (const float*)Y, dim, 1.0, (float*)O, dim);
}
*/
}	
}
}
}
#endif  
