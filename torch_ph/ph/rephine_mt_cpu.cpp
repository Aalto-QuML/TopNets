#include "ATen/core/function_schema.h"
#include "unionfind.hh"
#include <ATen/Parallel.h>
#include <algorithm>
#include <iostream>
#include <torch/extension.h>
#pragma omp parallel num_threads(4)

using namespace torch::indexing;

torch::Tensor uf_find(torch::Tensor parents, int u) {
  // Creating a single element tensor seems a bit hacky, but I didn't find an
  // alternative way to return a single element while supporting multiple
  // integer types.
  auto out = torch::empty(1, parents.options());
  AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_find", ([&] {
                               out[0] = UnionFind<scalar_t>::find(
                                   parents.accessor<scalar_t, 1>(),
                                   static_cast<scalar_t>(u));
                             }));
  return out[0];
}

void uf_merge(torch::Tensor parents, int u, int v) {
  AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_merge", ([&] {
                               UnionFind<scalar_t>::merge(
                                   parents.accessor<scalar_t, 1>(),
                                   static_cast<scalar_t>(u),
                                   static_cast<scalar_t>(v));
                             }));
}

template <typename float_t, typename int_t>
void compute_rephine_raw(
    torch::TensorAccessor<float_t, 1> filtered_v,
    torch::TensorAccessor<float_t, 1> filtered_e,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> parents,
    torch::TensorAccessor<int_t, 1> sorting_space, // no idea what this is
    torch::TensorAccessor<int_t, 2> pers_indices,
    torch::TensorAccessor<int_t, 2> pers1_indices, int_t vertex_begin,
    int_t vertex_end, int_t edge_begin, int_t edge_end) {

  auto n_vertices = vertex_end - vertex_begin;
  auto n_edges = edge_end - edge_begin;

  int_t *sorting_begin = sorting_space.data() + edge_begin;
  int_t *sorting_end = sorting_space.data() + edge_end;
  std::stable_sort(sorting_begin, sorting_end, [&filtered_e](int_t i, int_t j) {
    return filtered_e[i] < filtered_e[j];
  });

//  auto unpaired_index = *(sorting_end - 1);
//  int_t unpaired_vertex_index;
  // TODO: This has assumptions on the edge filtration selected
//  if (filtered_v[edge_index[unpaired_index][0]] <
//      filtered_v[edge_index[unpaired_index][1]])
//    unpaired_vertex_index = edge_index[unpaired_index][1];
//  else
//    unpaired_vertex_index = edge_index[unpaired_index][0];

  //unpaired_vertex_index =  torch::ones({1}, filtered_v.options()); // I added this. I should change the name!

  for (auto i = 0; i < n_edges; i++) {
    auto cur_edge_index = sorting_space[edge_begin + i];
//    auto cur_edge_weight = filtered_e[i];

    auto node1 = edge_index[cur_edge_index][0];
    auto node2 = edge_index[cur_edge_index][1];

    if (pers_indices[node1][1] == -1){
      pers_indices[node1][1] = cur_edge_index;
    }

    if (pers_indices[node2][1] == -1){
      pers_indices[node2][1] = cur_edge_index;
    }

    auto younger = UnionFind<int_t>::find(parents, node1);
    auto older = UnionFind<int_t>::find(parents, node2);

    if (younger == older) {
      pers1_indices[cur_edge_index][0] = cur_edge_index;
//      pers1_indices[cur_edge_index][1] = 0;
      continue;
    } else {
      if (filtered_v[younger] == filtered_v[older]){
        // Flip older and younger, node1 and node 2
        if(filtered_e[pers_indices[younger][1]] < filtered_e[pers_indices[older][1]]){
          auto tmp = younger;
          younger = older;
          older = tmp;
          tmp = node1;
          node1 = node2;
          node2 = tmp;
        }
      }
      else if (filtered_v[younger] < filtered_v[older]) {
        // Flip older and younger, node1 and node 2
        auto tmp = younger;
        younger = older;
        older = tmp;
        tmp = node1;
        node1 = node2;
        node2 = tmp;
      }
    }
    pers_indices[younger][0] = cur_edge_index; //cur_edge_weight; cur_vertex_index;
    UnionFind<int_t>::merge(parents, node1, node2);
  }
  // Handle roots, would make sense to do this outside as it can be
  // parallelized quite esily using torch operations.  Yet this would
  // require having access to the graph wise unpaired value, which we
  // usually dont have.
  //
  for (auto i = 0; i < n_vertices; i++) {
    auto vertex_index = vertex_begin + i;
    auto parent_value = parents[vertex_index];
    if (vertex_index == parent_value) {
//      pers_indices[vertex_index][0] = unpaired__index;
      pers_indices[vertex_index][0] = - 1; // sorting_space[edge_begin + n_eddges - 1]; //n_edges - 1;
    }
  }
}


template <typename float_t, typename int_t>
void compute_rephine_ptrs(
    torch::TensorAccessor<float_t, 2> filtered_v,
    torch::TensorAccessor<float_t, 2> filtered_e,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> vertex_slices,
    torch::TensorAccessor<int_t, 1> edge_slices,
    torch::TensorAccessor<int_t, 2> parents,
    torch::TensorAccessor<int_t, 2> sorting_space,
    torch::TensorAccessor<int_t, 3> pers_ind,
    torch::TensorAccessor<int_t, 3> pers1_ind) {
  auto n_graphs = vertex_slices.size(0) - 1;
  auto n_filtrations = filtered_v.size(0);

  at::parallel_for(
      0, n_graphs * n_filtrations, 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
          auto instance = i / n_filtrations;
          auto filtration = i % n_filtrations;
          compute_rephine_raw<float_t, int_t>(
              filtered_v[filtration], filtered_e[filtration], edge_index,
              parents[filtration], sorting_space[filtration],
              pers_ind[filtration], pers1_ind[filtration],
              vertex_slices[instance], vertex_slices[instance + 1],
              edge_slices[instance], edge_slices[instance + 1]);
        }
      });
}

std::tuple<torch::Tensor, torch::Tensor>
compute_rephine_batched_mt(torch::Tensor filtered_v,
                                        torch::Tensor filtered_e,
                                        torch::Tensor edge_index,
                                        torch::Tensor vertex_slices,
                                        torch::Tensor edge_slices) {
  // Changed index orders are required in order to allow slicing into
  // contingous memory regions Assumes shapes: filtered_v: [n_filtrations,
  // n_nodes] filtered_e: [n_filtrations, n_edges, 2] edge_index: [n_edges,
  // 2] vertex_slices: [n_graphs+1] edge_slices: [n_graphs+1]
  bool set_invalid_to_nan = true; // This might be relevant in the future when
                                   // we decide to handle cycles differently

  auto n_nodes = filtered_v.size(1);
  auto n_edges = filtered_e.size(1);
  auto n_filtrations = filtered_v.size(0);
  auto integer_no_grad = torch::TensorOptions();
  integer_no_grad = integer_no_grad.requires_grad(false);
  integer_no_grad = integer_no_grad.device(edge_index.options().device());
  integer_no_grad = integer_no_grad.dtype(edge_index.options().dtype());

  // Output indicators
  auto pers_ind = torch::full({n_filtrations, n_nodes, 2}, -1, integer_no_grad);
  // Already set the first part of the tuple
//  pers_ind.index_put_({"...", 1}, torch::arange(0, n_nodes, integer_no_grad));
  auto pers1_ind =
      torch::full({n_filtrations, n_edges, 3}, -1, integer_no_grad);

  // Datastructure for UnionFind and sorting operations
  auto parents = torch::arange(0, n_nodes, integer_no_grad)
                     .unsqueeze(0)
                     .repeat({n_filtrations, 1});
  auto sorting_space = torch::arange(0, n_edges, integer_no_grad)
                           .unsqueeze(0)
                           .repeat({n_filtrations, 1})
                           .contiguous();

  // Double dispatch over int and float types
  AT_DISPATCH_FLOATING_TYPES(
      filtered_v.scalar_type(), "compute_rephine_batched_mt1", ([&] {
        using float_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(
            edge_index.scalar_type(),
            "compute_rephine_batched_"
            "mt2",
            ([&] {
              using int_t = scalar_t;
              compute_rephine_ptrs<float_t, int_t>(
                  filtered_v.accessor<float_t, 2>(),
                  filtered_e.accessor<float_t, 2>(),
                  edge_index.accessor<int_t, 2>(),
                  vertex_slices.accessor<int_t, 1>(),
                  edge_slices.accessor<int_t, 1>(),
                  parents.accessor<int_t, 2>(),
                  sorting_space.accessor<int_t, 2>(),
                  pers_ind.accessor<int_t, 3>(),
                  pers1_ind.accessor<int_t, 3>());
            }));
      }));

  // Construct tensors with values from the indicators in order to retain
  // gradient information

  // Gather the filtration values according to the indices definde in pers_ind.
  auto pers_e =  filtered_e
          .index({torch::arange(0, n_filtrations, integer_no_grad).unsqueeze(1),
                  pers_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_nodes, 2});

  float_t invalid_fill_value;
  if (set_invalid_to_nan)
    invalid_fill_value = std::numeric_limits<float_t>::quiet_NaN();
  else
    invalid_fill_value = 0;

  pers_e.index_put_({pers_ind == -1}, invalid_fill_value);

  auto pers_v_ind = torch::full({n_filtrations, n_nodes, 1}, -1, integer_no_grad);
  pers_v_ind.index_put_({"...", 0}, torch::arange(0, n_nodes, integer_no_grad));

  auto pers_v =  filtered_v
          .index({torch::arange(0, n_filtrations, integer_no_grad).unsqueeze(1),
                  pers_v_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_nodes, 1});

  auto pers = torch::cat({pers_e, pers_v}, 2);

//  auto pers =
//      torch::full({n_filtrations, n_nodes, 2}, -1, integer_no_grad);

  // Gather filtration values according to the indices defined in pers_ind1.
  // Here we append a "fake value" to the filtration tensor. This value is
  // collected if no cycles were registered for the edge as the default value is
  // -1, i.e. the last element of the tensor.
  invalid_fill_value = 0;
  auto pers1 =
      torch::cat(
          {filtered_e, torch::full({n_filtrations, 1}, invalid_fill_value,
                                   filtered_v.options())},
         1)
          .index({torch::arange(n_filtrations, integer_no_grad).unsqueeze(1),
                  pers1_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_edges, 3});
  return std::make_tuple(std::move(pers), std::move(pers1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_rephine_batched_mt",
        &compute_rephine_batched_mt,
        py::call_guard<py::gil_scoped_release>(),
        "Persistence routine multi threading");
}

