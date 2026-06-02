#pragma once

#include <string>
#include <vector>

#include <istream>

namespace pfm::cache_eval
{

struct PairManifestEntry
{
    std::string pair_id;
    std::string feature_a;
    std::string feature_b;
    std::string matches;
    std::string warp_a_to_b;
};

/// Parses CSV rows with columns: pair_id,feature_a,feature_b,matches,warp_a_to_b.
/// Blank lines, comment lines starting with '#', and an optional header row are ignored.
std::vector<PairManifestEntry> parsePairManifest(std::istream& input);

/// Loads a pair manifest from disk and resolves relative file paths against the manifest directory.
std::vector<PairManifestEntry> loadPairManifest(const std::string& path);

/// Serializes manifest entries as CSV with the standard header.
std::string pairManifestCsv(const std::vector<PairManifestEntry>& entries);

/// Writes manifest entries as CSV.
void writePairManifest(const std::vector<PairManifestEntry>& entries, const std::string& path);

} // namespace pfm::cache_eval
