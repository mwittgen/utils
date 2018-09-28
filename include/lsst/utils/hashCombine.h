/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef LSST_UTILS_HASH_COMBINE_H
#define LSST_UTILS_HASH_COMBINE_H

#include <functional>

namespace lsst {
namespace utils {

//@{
/** Combine hashes
 *
 * This is provided as a convenience for those who need to hash a composite.
 * C++11 includes std::hash, but neglects to include a facility for
 * combining hashes.
 *
 * To use it:
 *
 *    std::size_t seed = 0;
 *    result = hashCombine(seed, obj1, obj2, obj3);
 *
 * This solution is provided by Matteo Italia
 * https://stackoverflow.com/a/38140932/834250
 */
inline std::size_t hashCombine(std::size_t seed) { return seed; }

template <typename T, typename... Rest>
std::size_t hashCombine(std::size_t seed, const T& value, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return hashCombine(seed, rest...);
}
//@}

}} // namespace lsst::utils

#endif
