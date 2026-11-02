#include "format.hh"

namespace Jetstream::Modules {

bool FileReaderFormat::Raw::describesSamples() const {
    return false;
}

Result FileReaderFormat::Raw::load(const std::string&, Recording&) const {
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
