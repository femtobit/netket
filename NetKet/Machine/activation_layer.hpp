// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_ACTIVATIONLAYER_HH
#define NETKET_ACTIVATIONLAYER_HH

#include <Eigen/Dense>
#include <complex>
#include <fstream>
#include <random>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_layer.hpp"

namespace netket {

template <typename T, typename A>
class Activation : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;
  using VectorRefType = typename AbstractLayer<T>::VectorRefType;
  using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

  A activation_;  // activation
  int size_;      // size_ = input size = output size
 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  Activation(const int input_size) : activation_(), size_(input_size) {
    Init();
  }

  void Init() {
    std::string buffer = "";
    InfoMessage(buffer) << activation_.name << " Activation Layer " << size_
                        << " --> " << size_ << std::endl;
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "Activation";
    layerpar["Size"] = size_;

    pars["Machine"]["Layers"].push_back(layerpar);
  }

  void from_json(const json & /*pars*/) override {}

  void InitRandomPars(int /*seed*/, double /*sigma*/) override {}

  int Npar() const override { return 0; }

  int Ninput() const override { return size_; }

  int Noutput() const override { return size_; }

  void GetParameters(VectorRefType /*pars*/, int /*start_idx*/) const override {
  }

  void SetParameters(VectorConstRefType /*pars*/, int /*start_idx*/) override {}

  void InitLookup(const VectorType &v, LookupType &lt,
                  VectorType &output) override {
    lt.resize(1);
    lt[0].resize(size_);

    Forward(v, lt, output);
  }

  void UpdateLookup(const VectorType & /*input*/,
                    const std::vector<int> &input_changes,
                    const VectorType &new_input, LookupType &theta,
                    const VectorType & /*output*/,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    const int num_of_changes = input_changes.size();
    if (num_of_changes == size_) {
      output_changes.resize(size_);
      new_output.resize(size_);
      theta[0] = new_input;
      activation_.operator()(new_input, new_output);
    } else if (num_of_changes > 0) {
      output_changes = input_changes;
      new_output.resize(num_of_changes);
      for (int i = 0; i < num_of_changes; ++i) {
        theta[0](input_changes[i]) = new_input(i);
      }
      activation_.operator()(new_input, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  void UpdateLookup(const Eigen::VectorXd & /*input*/,
                    const std::vector<int> &tochange,
                    const std::vector<double> &newconf, LookupType &theta,
                    const VectorType & /*output*/,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    const int num_of_changes = tochange.size();
    if (num_of_changes > 0) {
      output_changes = tochange;
      new_output.resize(num_of_changes);
      for (int i = 0; i < num_of_changes; ++i) {
        theta[0](tochange[i]) = newconf[i];
      }
      Eigen::VectorXcd new_input(num_of_changes);
      for (int j = 0; j < num_of_changes; ++j) {
        new_input(j) = newconf[j];
      }
      activation_.operator()(new_input, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  // Feedforward
  void Forward(const VectorType &prev_layer_output, LookupType &theta,
               VectorType &output) override {
    theta[0] = prev_layer_output;
    activation_.operator()(theta[0], output);
  }

  // Feedforward Using lookup
  void Forward(const LookupType &theta, VectorType &output) override {
    // Apply activation function
    activation_.operator()(theta[0], output);
  }

  // Computes derivative.
  void Backprop(const VectorType & /*prev_layer_output*/,
                const VectorType &this_layer_output,
                const LookupType &this_layer_theta, const VectorType &dout,
                VectorType &din, VectorType & /*der*/,
                int /*start_idx*/) override {
    din.resize(size_);
    activation_.ApplyJacobian(this_layer_theta[0], this_layer_output, dout,
                              din);
  }
};
}  // namespace netket

#endif