#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::vector<std::string> asText(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts);
std::vector<std::string> asTextNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors);
std::vector<std::string> asTextArxiv(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _backwards,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> title,
    std::vector<std::string> abstract,
    std::vector<std::string> input_title,
    std::vector<std::string> input_abstract,
    std::vector<std::string> input_labels);
std::vector<std::string> asTextNeighborsArxiv(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _backwards,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> title,
    std::vector<std::string> abstract,
    std::vector<std::string> input_title,
    std::vector<std::string> input_abstract,
    std::vector<std::string> input_labels);
