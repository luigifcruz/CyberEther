#include "messages.hh"

namespace Jetstream {

Mail::Mail() noexcept(std::is_nothrow_default_constructible_v<variant>) = default;

Mail::Mail(const Mail& other) = default;

Mail::Mail(Mail&& other) noexcept(std::is_nothrow_move_constructible_v<variant>) = default;

Mail& Mail::operator=(const Mail& other) = default;

Mail& Mail::operator=(Mail&& other) noexcept(std::is_nothrow_move_assignable_v<variant>) = default;

Mail::~Mail() = default;

}  // namespace Jetstream
