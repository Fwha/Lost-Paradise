#include "ps4elf/nid.h"

#include <fstream>

namespace ps4elf {

bool NidDb::load(const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto sp = line.find(' ');
        if (sp != 11) continue; // NIDs are always 11 characters
        names_.emplace(line.substr(0, sp), line.substr(sp + 1));
    }
    return true;
}

const std::string& NidDb::lookup(const std::string& nid) const {
    static const std::string empty;
    auto it = names_.find(nid);
    return it != names_.end() ? it->second : empty;
}

} // namespace ps4elf
