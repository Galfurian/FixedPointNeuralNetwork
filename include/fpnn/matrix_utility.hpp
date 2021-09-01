#pragma once

#include <vector>
#include <ostream>
#include <cassert>

namespace fpnn
{
    template <typename T>
    inline T eye(size_t size)
    {
        T m(size, size, 0);
        for (size_t i = 0; i < size; ++i)
            m(i, i) = 1;
        return m;
    }
}