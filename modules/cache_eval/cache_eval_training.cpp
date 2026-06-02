#include "cache_eval/cache_eval_training.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

#include <unordered_set>

namespace pfm::cache_eval
{
namespace
{

bool allDigits(const std::string& value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(),
                                         [](const char ch)
                                         {
                                             return ch >= '0' && ch <= '9';
                                         });
}

bool tryParsePairIndexToken(const std::string& value, int64_t& output)
{
    const auto stem = std::filesystem::path(value).stem().string();
    const auto prefix = std::string("pair_");
    if (stem.rfind(prefix, 0) != 0)
    {
        return false;
    }
    const auto digits = stem.substr(prefix.size());
    if (!allDigits(digits))
    {
        return false;
    }
    output = std::stoll(digits);
    return true;
}

} // namespace

int64_t extractSyntheticPairCacheIndex(const PairManifestEntry& entry)
{
    int64_t index = 0;
    if (tryParsePairIndexToken(entry.warp_a_to_b, index) || tryParsePairIndexToken(entry.pair_id, index))
    {
        return index;
    }
    throw std::invalid_argument("cache eval manifest entry cannot be mapped to a synthetic pair cache index: " +
                                entry.pair_id);
}

std::vector<int64_t> extractSyntheticPairCacheIndices(const std::vector<PairManifestEntry>& entries)
{
    std::vector<int64_t> indices;
    std::unordered_set<int64_t> seen;
    for (const auto& entry : entries)
    {
        const auto index = extractSyntheticPairCacheIndex(entry);
        if (seen.insert(index).second)
        {
            indices.push_back(index);
        }
    }
    return indices;
}

std::string hardCacheIndexCsv(const std::vector<int64_t>& indices)
{
    std::ostringstream out;
    out << "hard_cache_index\n";
    for (const auto index : indices)
    {
        if (index < 0)
        {
            throw std::invalid_argument("hard cache indices must be non-negative");
        }
        out << index << '\n';
    }
    return out.str();
}

} // namespace pfm::cache_eval
