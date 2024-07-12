#include "tplcpp/binding_helpers.hpp"

std::string shapeToStr(py::array& array) {

    std::stringstream ss;
    ss << "(";

    for (int k = 0; k < array.ndim(); ++k) {
        ss << array.shape()[k];
        if (k != array.ndim() - 1) {
            ss <<  ", ";
        }
    }
    ss << ")";

    return ss.str();
}

void assertArrayShape(std::string name,
                      py::array& array,
                      std::vector<std::vector<int>> shapes) {

    bool foundShape = false;

    for (auto& shape : shapes) { 
        if ((int)shape.size() != array.ndim()) {
            continue;
        }
        bool ok = true;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                continue;
            }
            ok &= shape[i] == array.shape()[i];
        }
        if (ok) { 
            foundShape = true;
            break;
        }
    }

    if (!foundShape) {
        std::stringstream ss;
        ss << "Expected \"" + name + "\" with shape ";

        for (size_t i = 0; i < shapes.size(); ++i) { 
            ss << "(";
            auto& shape = shapes[i];
            for (size_t k = 0; k < shape.size(); ++k) { 
                ss << shape[k]; 
                if (k != shape.size() - 1) { 
                    ss <<  ", ";
                }
            }
            ss << ")";
            if (i != shapes.size() - 1) { 
                ss << " | ";
            }
        }

        ss << ", but found " << shapeToStr(array) << std::endl;

        throw py::value_error(ss.str());
    }
}
