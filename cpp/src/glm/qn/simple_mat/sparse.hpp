/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <vector>

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <linalg/ternary_op.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/mr/device/allocator.hpp>
#include <rmm/device_uvector.hpp>
#include "base.hpp"

namespace ML {

/**
 * Sparse matrix in CSR format.
 *
 * Note, we use cuSPARSE to manimulate matrices, and it guarantees:
 *
 *  1. row_ids[m] == nnz
 *  2. cols are sorted within rows.
 *
 * However, when the data comes from the outside, we cannot guarantee that.
 */
template <typename T>
struct SimpleSparseMat : SimpleMat<T> {
  typedef SimpleMat<T> Super;
  T *values;
  int *cols;
  int *row_ids;
  int nnz;

  SimpleSparseMat()
    : Super(0, 0), values(nullptr), cols(nullptr), row_ids(nullptr), nnz(0) {}

  SimpleSparseMat(T *values, int *cols, int *row_ids, int nnz, int m, int n)
    : Super(m, n), values(values), cols(cols), row_ids(row_ids), nnz(nnz) {
    check_csr(*this, 0);
  }

  void print(std::ostream &oss) const override { oss << (*this) << std::endl; }

  void operator=(const SimpleSparseMat<T> &other) = delete;

  inline void gemmb(const raft::handle_t &handle, const T alpha,
                    const SimpleDenseMat<T> &A, const bool transA,
                    const bool transB, const T beta, SimpleDenseMat<T> &C,
                    cudaStream_t stream) const override {
    const SimpleSparseMat<T> &B = *this;
    int kA = A.n;
    int kB = B.m;

    if (transA) {
      ASSERT(A.n == C.m, "GEMM invalid dims: m");
      kA = A.m;
    } else {
      ASSERT(A.m == C.m, "GEMM invalid dims: m");
    }

    if (transB) {
      ASSERT(B.m == C.n, "GEMM invalid dims: n");
      kB = B.n;
    } else {
      ASSERT(B.n == C.n, "GEMM invalid dims: n");
    }
    ASSERT(kA == kB, "GEMM invalid dims: k");

    // matrix C must change the order and be transposed, because we need
    // to swap arguments A and B in cusparseSpMM.
    cusparseDnMatDescr_t descrC;
    auto order = C.ord == COL_MAJOR ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    CUSPARSE_CHECK(raft::sparse::cusparsecreatednmat(
      &descrC, C.n, C.m, order == CUSPARSE_ORDER_COL ? C.n : C.m, C.data,
      order));

    /*
      Dense matrix A must have the same order as the sparse matrix C
      (i.e. swapped order w.r.t. original C).
      To account this requirement, I may need to flip transA (whether to transpose A).

         C   C' rowsC' colsC' ldC'   A  A' rowsA' colsA' ldA'
         c   r    n      m     m     c  r    n      m     m
         c   r    n      m     m     r  r    m      n     n
         r   c    n      m     n     c  c    m      n     m
         r   c    n      m     n     r  c    n      m     n
     */
    cusparseDnMatDescr_t descrA;
    CUSPARSE_CHECK(raft::sparse::cusparsecreatednmat(
      &descrA, C.ord == A.ord ? A.n : A.m, C.ord == A.ord ? A.m : A.n,
      A.ord == COL_MAJOR ? A.m : A.n, A.data, order));
    auto opA = transA ^ (C.ord == A.ord) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                         : CUSPARSE_OPERATION_TRANSPOSE;

    cusparseSpMatDescr_t descrB;
    CUSPARSE_CHECK(raft::sparse::cusparsecreatecsr(
      &descrB, B.m, B.n, B.nnz, B.row_ids, B.cols, B.values));
    auto opB =
      transB ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

    auto alg = order == CUSPARSE_ORDER_COL ? CUSPARSE_SPMM_CSR_ALG1
                                           : CUSPARSE_SPMM_CSR_ALG2;

    size_t bufferSize;
    CUSPARSE_CHECK(raft::sparse::cusparsespmm_bufferSize(
      handle.get_cusparse_handle(), opB, opA, &alpha, descrB, descrA, &beta,
      descrC, alg, &bufferSize, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    rmm::device_uvector<T> tmp(bufferSize, stream);

    CUSPARSE_CHECK(raft::sparse::cusparsespmm(
      handle.get_cusparse_handle(), opB, opA, &alpha, descrB, descrA, &beta,
      descrC, alg, tmp.data(), stream));

    CUSPARSE_CHECK(cusparseDestroyDnMat(descrA));
    CUSPARSE_CHECK(cusparseDestroySpMat(descrB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(descrC));
  }
};

template <typename T>
inline void check_csr(const SimpleSparseMat<T> &mat, cudaStream_t stream) {
  int row_ids_nnz;
  raft::update_host(&row_ids_nnz, &mat.row_ids[mat.m], 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(row_ids_nnz == mat.nnz,
         "SimpleSparseMat: the size of CSR row_ids array must be `m + 1`, and "
         "the last element must be equal nnz.");
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const SimpleSparseMat<T> &mat) {
  check_csr(mat, 0);
  os << "SimpleSparseMat (CSR)"
     << "\n";
  std::vector<T> values(mat.nnz);
  std::vector<int> cols(mat.nnz);
  std::vector<int> row_ids(mat.m + 1);
  raft::update_host(&values[0], mat.values, mat.nnz, 0);
  raft::update_host(&cols[0], mat.cols, mat.nnz, 0);
  raft::update_host(&row_ids[0], mat.row_ids, mat.m + 1, 0);
  CUDA_CHECK(cudaStreamSynchronize(0));

  int i, row_end = 0;
  for (int row = 0; row < mat.m; row++) {
    i = row_end;
    row_end = row_ids[row + 1];
    for (int col = 0; col < mat.n; col++) {
      if (i >= row_end || col < cols[i]) {
        os << "0";
      } else {
        os << values[i];
        i++;
      }
      if (col < mat.n - 1) os << ",";
    }

    os << std::endl;
  }

  return os;
}

};  // namespace ML
