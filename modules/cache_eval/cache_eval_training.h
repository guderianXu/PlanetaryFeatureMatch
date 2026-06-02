#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cache_eval/cache_eval_manifest.h"

namespace pfm::cache_eval
{

/// Extracts the synthetic pair cache index from a manifest entry.
/// The parser accepts pair ids or filenames like pair_000077.
int64_t extractSyntheticPairCacheIndex(const PairManifestEntry& entry);

/// Extracts stable unique synthetic pair cache indices from manifest entries.
std::vector<int64_t> extractSyntheticPairCacheIndices(const std::vector<PairManifestEntry>& entries);

/// Serializes hard cache indices as a one-column CSV.
std::string hardCacheIndexCsv(const std::vector<int64_t>& indices);

} // namespace pfm::cache_eval
