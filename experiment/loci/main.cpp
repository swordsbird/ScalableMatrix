//example.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace py = pybind11;
using std::cout;

class LOCI {
public:
    
};

PYBIND11_MODULE(LOCI, m){
    m.doc() = "pybind11 example";
    pybind11::class_<Space>(m, "Space")
        .def( pybind11::init() )
        .def( "init", &Space::init301 )
        .def( "numOfDAG", &Space::numOfDAG )
        .def( "meanshift", &Space::meanshift )
        .def( "meanshiftAdaptive", &Space::meanshiftAdaptive )
        .def( "meanshiftInSubset", &Space::meanshiftInSubset )
        .def( "meanshiftInSubsetAdaptive", &Space::meanshiftInSubsetAdaptive )
        .def( "getDist", &Space::getDistMatrix );
}