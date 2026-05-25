#include "cache_eval/cache_eval_manifest.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pfm::cache_eval {
namespace {

std::string trim(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

bool isCommentOrBlank(const std::string& line) {
    const auto stripped = trim(line);
    return stripped.empty() || stripped.front() == '#';
}

std::vector<std::string> parseCsvLine(const std::string& line, int64_t line_number) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];
        if (in_quotes) {
            if (ch == '"') {
                if (index + 1 < line.size() && line[index + 1] == '"') {
                    field.push_back('"');
                    ++index;
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(ch);
            }
        } else if (ch == '"') {
            in_quotes = true;
        } else if (ch == ',') {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    if (in_quotes) {
        throw std::invalid_argument("unterminated quote in cache eval manifest line " + std::to_string(line_number));
    }
    fields.push_back(trim(field));
    return fields;
}

bool isHeaderRow(const std::vector<std::string>& fields) {
    return fields.size() >= 5 &&
           fields[0] == "pair_id" &&
           fields[1] == "feature_a" &&
           fields[2] == "feature_b" &&
           fields[3] == "matches" &&
           fields[4] == "warp_a_to_b";
}

std::string resolvePath(const std::filesystem::path& base_dir, const std::string& value) {
    const auto path = std::filesystem::path(value);
    if (path.empty() || path.is_absolute() || base_dir.empty()) {
        return value;
    }
    return (base_dir / path).lexically_normal().string();
}

std::string csvEscape(const std::string& value) {
    const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needs_quotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

}  // namespace

std::vector<PairManifestEntry> parsePairManifest(std::istream& input) {
    std::vector<PairManifestEntry> entries;
    std::string line;
    int64_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (isCommentOrBlank(line)) {
            continue;
        }

        auto fields = parseCsvLine(line, line_number);
        if (entries.empty() && isHeaderRow(fields)) {
            continue;
        }
        if (fields.size() != 5) {
            throw std::invalid_argument(
                "cache eval manifest line " + std::to_string(line_number) +
                " must have 5 columns: pair_id,feature_a,feature_b,matches,warp_a_to_b");
        }
        for (const auto& field : fields) {
            if (field.empty()) {
                throw std::invalid_argument("cache eval manifest line " + std::to_string(line_number) +
                                            " contains an empty required field");
            }
        }
        entries.push_back(PairManifestEntry{
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            fields[4]});
    }
    if (entries.empty()) {
        throw std::invalid_argument("cache eval manifest has no pair rows");
    }
    return entries;
}

std::vector<PairManifestEntry> loadPairManifest(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("failed to open cache eval manifest: " + path);
    }
    auto entries = parsePairManifest(input);
    const auto base_dir = std::filesystem::path(path).parent_path();
    for (auto& entry : entries) {
        entry.feature_a = resolvePath(base_dir, entry.feature_a);
        entry.feature_b = resolvePath(base_dir, entry.feature_b);
        entry.matches = resolvePath(base_dir, entry.matches);
        entry.warp_a_to_b = resolvePath(base_dir, entry.warp_a_to_b);
    }
    return entries;
}

std::string pairManifestCsv(const std::vector<PairManifestEntry>& entries) {
    if (entries.empty()) {
        throw std::invalid_argument("cache eval manifest entries must not be empty");
    }
    std::ostringstream out;
    out << "pair_id,feature_a,feature_b,matches,warp_a_to_b\n";
    for (const auto& entry : entries) {
        if (entry.pair_id.empty() || entry.feature_a.empty() || entry.feature_b.empty() ||
            entry.matches.empty() || entry.warp_a_to_b.empty()) {
            throw std::invalid_argument("cache eval manifest entry contains an empty required field");
        }
        out << csvEscape(entry.pair_id) << ','
            << csvEscape(entry.feature_a) << ','
            << csvEscape(entry.feature_b) << ','
            << csvEscape(entry.matches) << ','
            << csvEscape(entry.warp_a_to_b) << '\n';
    }
    return out.str();
}

void writePairManifest(const std::vector<PairManifestEntry>& entries, const std::string& path) {
    const auto output_path = std::filesystem::path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(output_path);
    if (!output) {
        throw std::invalid_argument("failed to open cache eval manifest for writing: " + path);
    }
    output << pairManifestCsv(entries);
}

}  // namespace pfm::cache_eval
