// NID -> symbol name lookup.
//
// The database isn't shipped with the repo; scripts/fetch-nids.sh downloads a public one.
// The file format is one "NID name" pair per line.
#pragma once

#include <string>
#include <unordered_map>

namespace ps4elf {

class NidDb {
public:
    bool load(const std::string& path, std::string* error);
    // Returns the symbol name, or an empty string if the NID is unknown.
    const std::string& lookup(const std::string& nid) const;
    size_t size() const { return names_.size(); }

private:
    std::unordered_map<std::string, std::string> names_;
};

} // namespace ps4elf
