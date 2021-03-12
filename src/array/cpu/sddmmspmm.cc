/*!
 *  Copyright (c) 2020 by Contributors
 * \file aten/cpu/sddmm.cc
 * \brief SDDMM C APIs and definitions.
 */
#include "./sddmmspmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

#define SWITCH_RHS(rhs_target, RhsTarget, ...)                        \
  do {                                                                \
    if ((rhs_target) == 0) {                                          \
      constexpr int RhsTarget = 0;                                    \
      { __VA_ARGS__ }                                                 \
    } else if ((rhs_target) == 1) {                                   \
      constexpr int RhsTarget = 1;                                    \
      { __VA_ARGS__ }                                                 \
    } else if ((rhs_target) == 2) {                                   \
      constexpr int RhsTarget = 2;                                    \
      { __VA_ARGS__ }                                                 \
    } else {                                                          \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target);            \
    }                                                                 \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...)\
  do {                                                                  \
    if ((lhs_target) == 0) {                                            \
      constexpr int LhsTarget = 0;                                      \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                   \
    } else if ((lhs_target) == 1) {                                     \
      constexpr int LhsTarget = 1;                                      \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                   \
    } else if ((lhs_target) == 2) {                                     \
      constexpr int LhsTarget = 2;                                      \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                   \
    } else {                                                            \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);              \
    }                                                                   \
  } while (0)


/*! \brief Generalized SDDMM on Csr format. */
template <int XPU, typename IdType, typename DType>
void SDDMMSPMMCsr(const std::string& op,
              const BcastOff& bcast,
              const CSRMatrix& csr,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target, int32_t imsg) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      cpu::SDDMMSPMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, csr, lhs, rhs, out, imsg);
    });
  });
}

template void SDDMMSPMMCsr<kDLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target, int32_t imsg);
template void SDDMMSPMMCsr<kDLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target, int32_t imsg);
template void SDDMMSPMMCsr<kDLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target, int32_t imsg);
template void SDDMMSPMMCsr<kDLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target, int32_t imsg);

}  // namespace aten
}  // namespace dgl
