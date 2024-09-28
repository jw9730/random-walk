#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include "asText.hpp"
#include "threading.hpp"

namespace py = pybind11;

std::vector<std::string> asText(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        for (size_t k = 0; k < walkLen; k++)
        {
            walkStream << walks[i * walkLen + k];
            if (k < walkLen - 1)
            {
                bool restart = restarts[i * walkLen + k + 1];
                walkStream << (restart ? ";" : "-");
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}


std::vector<std::string> asTextNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        for (size_t k = 0; k < walkLen; k++)
        {
            walkStream << walks[i * walkLen + k];
            if (k < walkLen - 1)
            {
                bool restart = restarts[i * walkLen + k + 1];
                bool neighbor = neighbors[i * walkLen + k + 1];
                walkStream << (neighbor ? "#" : (restart ? ";" : "-"));
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}

std::vector<std::string> asTextArxiv(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _backwards,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> title,
    std::vector<std::string> abstract,
    std::vector<std::string> input_title,
    std::vector<std::string> input_abstract,
    std::vector<std::string> input_labels)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info backwardsBuf = _backwards.request();
    bool *backwards = (bool *)backwardsBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        std::vector<bool> visited(title.size(), false);
        size_t start_index = indices[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            bool backward = backwards[i * walkLen + k];
            size_t index = indices[i * walkLen + k];
            if (k > 0)
            {
                size_t prev = walks[i * walkLen + k - 1];
                if (!restart)
                {
                    walkStream << " Restart at ";
                }
                else if (backward)
                {
                    walkStream << " Paper " << prev << " is cited by ";
                }
                else
                {
                    walkStream << " Paper " << prev << " cites ";
                }
                walkStream << value;
            }
            else
            {
                walkStream << "Paper " << value;
            }
            if (!visited[index])
            {
                if (index == start_index)
                {
                    walkStream << " - Title: " << title[index];
                }
                else
                {
                    walkStream << " - " << input_title[index];
                }
                if ((index != start_index) && (input_labels[index] != "Unknown"))
                {
                    walkStream << ", Category: " << input_labels[index];
                }
                if (index == start_index)
                {
                    walkStream << ", Abstract: " << abstract[index];
                }
                else
                {
                    walkStream << ", Abstract: " << input_abstract[index];
                }
                visited[index] = true;
            }
            else
            {
                walkStream << ".";
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}

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
    std::vector<std::string> input_labels)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info backwardsBuf = _backwards.request();
    bool *backwards = (bool *)backwardsBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        std::vector<bool> visited(title.size(), false);
        size_t start_index = indices[i * walkLen];
        size_t prev = walks[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            bool neighbor = neighbors[i * walkLen + k];
            bool backward = backwards[i * walkLen + k];
            size_t index = indices[i * walkLen + k];
            if (k > 0)
            {
                if (restart)
                {
                    walkStream << " Restart at ";
                }
                else if (backward)
                {
                    walkStream << " Paper " << prev << " is cited by ";
                }
                else
                {
                    walkStream << " Paper " << prev << " cites ";
                }
                walkStream << value;
            }
            else
            {
                walkStream << "Paper " << value;
            }
            if (!visited[index])
            {
                if (index == start_index)
                {
                    walkStream << " - Title: " << title[index];
                }
                else
                {
                    walkStream << " - " << input_title[index];
                }
                if ((index != start_index) && (input_labels[index] != "Unknown"))
                {
                    walkStream << ", Category: " << input_labels[index];
                }
                if (index == start_index)
                {
                    walkStream << ", Abstract: " << abstract[index];
                }
                else
                {
                    walkStream << ", Abstract: " << input_abstract[index];
                }
                visited[index] = true;
            }
            else
            {
                walkStream << ".";
            }
            if (!neighbor)
            {
                prev = value;
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}
