#pragma once

#include <vector>

#include "IntegerArray.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            using pankey::Type::Array::IntegerArray;
            using IntegerArrayList = std::vector<IntegerArray>;

            struct DenseNetwork{
                IntegerArray m_index;
                IntegerArray m_topology;
                bool m_kind = false;

                DenseNetwork() = default;

                DenseNetwork(const IntegerArray& a_index,
                             const IntegerArray& a_topology,
                             bool a_kind)
                    : m_index(a_index),
                      m_topology(a_topology),
                      m_kind(a_kind){}
            };

            struct DynamicNetwork{
                IntegerArrayList m_index;
                IntegerArray m_topology;
                IntegerArrayList m_kind;
                IntegerArrayList m_connections;

                DynamicNetwork() = default;

                DynamicNetwork(const IntegerArrayList& a_index,
                               const IntegerArray& a_topology,
                               const IntegerArrayList& a_kind,
                               const IntegerArrayList& a_connections)
                    : m_index(a_index),
                      m_topology(a_topology),
                      m_kind(a_kind),
                      m_connections(a_connections){}
            };

            using NeuralNetwork = DenseNetwork;

        }

    }

}
