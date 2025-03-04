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

std::vector<std::string> asTextAmazon(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> product,
    std::vector<std::string> input_product,
    std::vector<std::string> input_labels)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

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
        std::vector<bool> visited(product.size(), false);
        size_t start_index = indices[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            size_t index = indices[i * walkLen + k];
            if (k > 0)
            {
                size_t prev = walks[i * walkLen + k - 1];
                if (restart)
                {
                    walkStream << " Restart at ";
                }
                else
                {
                    walkStream << " Product " << prev << " is often purchased with ";
                }
                walkStream << value;
            }
            else
            {
                walkStream << "Product " << value;
            }
            if (!visited[index])
            {
                if (index == start_index)
                {
                    walkStream << " - " << product[index];
                }
                else
                {
                    walkStream << " - " << input_product[index];
                }
                if ((index != start_index) && (input_labels[index] != "Unknown"))
                {
                    walkStream << ", Rating: " << input_labels[index] << ".";
                }
                else
                {
                    walkStream << ".";
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

std::vector<std::string> asTextNeighborsAmazon(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> product,
    std::vector<std::string> input_product,
    std::vector<std::string> input_labels)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

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
        std::vector<bool> visited(product.size(), false);
        size_t start_index = indices[i * walkLen];
        size_t prev = walks[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            bool neighbor = neighbors[i * walkLen + k];
            size_t index = indices[i * walkLen + k];
            if (k > 0)
            {
                if (restart)
                {
                    walkStream << " Restart at ";
                }
                else
                {
                    walkStream << " Product " << prev << " is often purchased with ";
                }
                walkStream << value;
            }
            else
            {
                walkStream << "Product " << value;
            }
            if (!visited[index])
            {
                if (index == start_index)
                {
                    walkStream << " - " << product[index];
                }
                else
                {
                    walkStream << " - " << input_product[index];
                }
                if ((index != start_index) && (input_labels[index] != "Unknown"))
                {
                    walkStream << ", Rating: " << input_labels[index] << ".";
                }
                else
                {
                    walkStream << ".";
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

std::vector<std::string> asTextPeptides(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> node_attr,
    py::array_t<uint32_t> _indptr_edge_attr,
    py::array_t<uint32_t> _indices_edge_attr,
    std::vector<std::string> edge_attr)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info indptrEdgeAttrBuf = _indptr_edge_attr.request();
    uint32_t *indptr_edge_attr = (uint32_t *)indptrEdgeAttrBuf.ptr;

    py::buffer_info indicesEdgeAttrBuf = _indices_edge_attr.request();
    uint32_t *indices_edge_attr = (uint32_t *)indicesEdgeAttrBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        std::vector<bool> visited(node_attr.size(), false);
        size_t start_index = indices[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            size_t index = indices[i * walkLen + k];
            if (k > 0)
            {
                size_t prev_index = indices[i * walkLen + k - 1];
                if (restart)
                {
                    walkStream << ";";
                }
                else
                {
                    // edge attribute
                    size_t start = indptr_edge_attr[prev_index];
                    size_t end = indptr_edge_attr[prev_index + 1];
                    for (size_t z = start; z < end; z++)
                    {
                        size_t next = indices_edge_attr[z];
                        if (next == index)
                        {
                            walkStream << edge_attr[z];
                            break;
                        }
                    }
                }
                walkStream << value;
            }
            else
            {
                walkStream << value;
            }
            if (!visited[index])
            {
                walkStream << node_attr[index];
                visited[index] = true;
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}

std::vector<std::string> asTextNeighborsPeptides(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors,
    py::array_t<uint32_t> _indices,
    std::vector<std::string> node_attr,
    py::array_t<uint32_t> _indptr_edge_attr,
    py::array_t<uint32_t> _indices_edge_attr,
    std::vector<std::string> edge_attr)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info indptrEdgeAttrBuf = _indptr_edge_attr.request();
    uint32_t *indptr_edge_attr = (uint32_t *)indptrEdgeAttrBuf.ptr;

    py::buffer_info indicesEdgeAttrBuf = _indices_edge_attr.request();
    uint32_t *indices_edge_attr = (uint32_t *)indicesEdgeAttrBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        std::vector<bool> visited(node_attr.size(), false);
        size_t start_index = indices[i * walkLen];
        size_t prev_index = indices[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            bool neighbor = neighbors[i * walkLen + k];
            size_t index = indices[i * walkLen + k];
            if (k > 0)
            {
                if (restart)
                {
                    walkStream << ";";
                }
                else
                {
                    assert(prev_index + 1 < indptrEdgeAttrBuf.shape[0]);
                    // edge attribute
                    size_t start = indptr_edge_attr[prev_index];
                    size_t end = indptr_edge_attr[prev_index + 1];
                    for (size_t z = start; z < end; z++)
                    {
                        assert(z < indptrEdgeAttrBuf.shape[0]);
                        size_t next = indices_edge_attr[z];
                        if (next == index)
                        {
                            assert(z < edge_attr.size());
                            walkStream << (neighbor ? "#" : "") << edge_attr[z];
                            break;
                        }
                    }
                }
                walkStream << value;
            }
            else
            {
                walkStream << value;
            }
            if (!visited[index])
            {
                walkStream << node_attr[index];
                visited[index] = true;
            }
            if (!neighbor)
            {
                prev_index = index;
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}
