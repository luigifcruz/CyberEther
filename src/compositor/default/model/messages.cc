#include "messages.hh"

namespace Jetstream {

Mail::Mail() = default;

Mail::Mail(const Mail& other) = default;

Mail::Mail(Mail&& other) noexcept = default;

Mail& Mail::operator=(const Mail& other) = default;

Mail& Mail::operator=(Mail&& other) noexcept = default;

Mail::~Mail() = default;

}  // namespace Jetstream
