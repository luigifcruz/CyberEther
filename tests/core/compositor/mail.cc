#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/base.hh"
#include "mail_hidden_producer.hh"

#include <cstdlib>
#include <deque>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using namespace Jetstream;

namespace {

thread_local volatile U64 Allocations = 0;
const void* volatile Escaped = nullptr;

void* Allocate(std::size_t size) {
    Allocations = Allocations + 1;
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

}  // namespace

[[gnu::noinline]] void* operator new(std::size_t size) {
    return Allocate(size);
}

[[gnu::noinline]] void* operator new[](std::size_t size) {
    return Allocate(size);
}

[[gnu::noinline]] void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return Allocate(size);
    } catch (...) {
        return nullptr;
    }
}

[[gnu::noinline]] void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return Allocate(size);
    } catch (...) {
        return nullptr;
    }
}

[[gnu::noinline]] void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

[[gnu::noinline]] void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

[[gnu::noinline]] void operator delete(void* pointer, const std::nothrow_t&) noexcept {
    std::free(pointer);
}

[[gnu::noinline]] void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
    std::free(pointer);
}

[[gnu::noinline]] void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

[[gnu::noinline]] void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace {

template<typename T>
concept Accessible = requires(Mail& mail, const Mail& view) {
    view.is<T>();
    view.isAnyOf<T>();
    mail.getIf<T>();
    view.getIf<T>();
    mail.get<T>();
    view.get<T>();
};

static_assert(!Mail::Registered<void>);
static_assert(!Mail::Registered<int>);
static_assert(!Mail::Registered<std::string>);
static_assert(!Mail::Registered<Mail>);
static_assert(Mail::Registered<MailQuit>);

static_assert(!Accessible<void>);
static_assert(!Accessible<int>);
static_assert(!Accessible<std::string>);
static_assert(Accessible<MailQuit>);

static_assert(!std::is_constructible_v<Mail, int>);
static_assert(!std::is_constructible_v<Mail, std::string>);
static_assert(!std::is_convertible_v<std::string, Mail>);
static_assert(std::is_convertible_v<MailQuit, Mail>);
static_assert(std::is_convertible_v<const MailSetWelcomeQuery&, Mail>);

static_assert(std::is_nothrow_move_constructible_v<Mail>);
static_assert(std::is_nothrow_move_assignable_v<Mail>);
static_assert(std::is_nothrow_default_constructible_v<Mail>);

static_assert(Mail::IsInline<MailSetNodeMeta>);
static_assert(Mail::IsInline<MailSurfaceInput>);
static_assert(Mail::IsInline<MailResizeSurface>);
static_assert(Mail::IsInline<MailTerminalInput>);
static_assert(!Mail::IsInline<MailSaveAgentProvider>);

template<typename T>
U64 CountAllocations(T payload) {
    const U64 before = Allocations;
    Mail mail(std::move(payload));
    Escaped = mail.getIf<T>();
    const U64 after = Allocations;
    return after - before;
}

template<typename T, typename Read>
void CheckValueSemantics(const T& value, Read read) {
    const auto expected = read(value);

    Mail original = value;
    Mail& alias = original;
    Mail copy(alias);
    REQUIRE(copy.is<T>());
    CHECK(read(copy.get<T>()) == expected);
    CHECK(read(original.get<T>()) == expected);
    CHECK(copy.getIf<T>() != original.getIf<T>());

    const T* address = copy.getIf<T>();
    Mail moved(std::move(copy));
    REQUIRE(moved.is<T>());
    CHECK(read(moved.get<T>()) == expected);
    if constexpr (!Mail::IsInline<T>) {
        CHECK(moved.getIf<T>() == address);
    }
    CHECK(copy.empty());
    CHECK(copy.index() == Mail::EmptyIndex);
    CHECK_FALSE(copy.is<T>());
    CHECK(copy.getIf<T>() == nullptr);
    CHECK_THROWS_AS(copy.get<T>(), std::logic_error);

    Mail assigned = MailQuit{};
    assigned = original;
    CHECK(read(assigned.get<T>()) == expected);
    assigned = std::move(moved);
    CHECK(read(assigned.get<T>()) == expected);
    CHECK(moved.empty());

    copy = original;
    CHECK(read(copy.get<T>()) == expected);
    moved = MailQuit{};
    CHECK(moved.is<MailQuit>());

    Mail& self = assigned;
    assigned = self;
    CHECK(read(assigned.get<T>()) == expected);
    assigned = std::move(self);
    CHECK(read(assigned.get<T>()) == expected);
}

}  // namespace

TEST_CASE("Default mail is empty and stays empty through copies and moves", "[core][compositor][mail]") {
    const Mail mail{};
    CHECK(mail.empty());
    CHECK(mail.index() == Mail::EmptyIndex);
    CHECK_FALSE(mail.is<MailQuit>());
    CHECK(mail.getIf<MailQuit>() == nullptr);
    CHECK_FALSE(mail.isAnyOf<MailQuit, MailNewFlowgraph>());
    CHECK_THROWS_AS(mail.get<MailQuit>(), std::logic_error);

    Mail copy = mail;
    CHECK(copy.empty());
    Mail moved = std::move(copy);
    CHECK(moved.empty());
    CHECK(copy.empty());
    moved = MailNewFlowgraph{};
    CHECK(moved.is<MailNewFlowgraph>());
    moved = mail;
    CHECK(moved.empty());
}

TEST_CASE("Mail reports the payload it holds", "[core][compositor][mail]") {
    Mail mail = MailSetWelcomeQuery{.query = "radio"};
    CHECK_FALSE(mail.empty());
    CHECK(mail.index() == Mail::IndexOf<MailSetWelcomeQuery>);
    CHECK(mail.is<MailSetWelcomeQuery>());
    CHECK_FALSE(mail.is<MailSetDocsQuery>());
    CHECK(mail.getIf<MailSetDocsQuery>() == nullptr);
    CHECK(mail.isAnyOf<MailSetDocsQuery, MailSetWelcomeQuery>());
    CHECK_FALSE(mail.isAnyOf<MailSetDocsQuery, MailQuit>());
    CHECK_THROWS_AS(mail.get<MailSetDocsQuery>(), std::logic_error);
    CHECK_THROWS_AS(std::as_const(mail).get<MailSetDocsQuery>(), std::logic_error);

    REQUIRE(mail.getIf<MailSetWelcomeQuery>() != nullptr);
    mail.get<MailSetWelcomeQuery>().query = "maps";
    CHECK(std::as_const(mail).get<MailSetWelcomeQuery>().query == "maps");
    CHECK(std::as_const(mail).getIf<MailSetWelcomeQuery>() == mail.getIf<MailSetWelcomeQuery>());
}

TEST_CASE("Mail copies and moves inline and heap payloads", "[core][compositor][mail]") {
    SECTION("inline") {
        CheckValueSemantics(MailSetWelcomeQuery{.query = std::string(64, 'q')},
                            [](const MailSetWelcomeQuery& mail) { return mail.query; });
    }
    SECTION("heap") {
        CheckValueSemantics(MailSaveAgentProvider{.replaceName = std::string(48, 'n'), .apiKey = std::string(48, 'k')},
                            [](const MailSaveAgentProvider& mail) { return *mail.replaceName + mail.apiKey; });
    }
}

TEST_CASE("Hot mail payloads are stored without heap allocations", "[core][compositor][mail]") {
    CHECK(CountAllocations(MailSetNodeMeta{.flowgraph = "graph", .block = "a"}) == 0);
    CHECK(CountAllocations(MailSurfaceInput{}) == 0);
    CHECK(CountAllocations(MailResizeSurface{.flowgraph = "graph", .block = "a", .metaKey = "surface"}) == 0);
    CHECK(CountAllocations(MailTerminalInput{.terminal = 1, .input = MailTerminalInput::Focus{.value = true}}) == 0);
    CHECK(CountAllocations(MailSaveAgentProvider{.apiKey = "key"}) >= 1);
}

TEST_CASE("Mail made in a hidden visibility library dispatches through the default actions",
          "[core][compositor][mail]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime{state, callbacks};
    DefaultActions actions{state, callbacks, runtime};
    state.flowgraph.items["graph"] = std::make_shared<Flowgraph>();

    const Mail query = MailHiddenProducer::WelcomeQuery("spectrum");
    REQUIRE(query.is<MailSetWelcomeQuery>());
    CHECK(query.index() == Mail::IndexOf<MailSetWelcomeQuery>);
    REQUIRE(actions.handle(query) == Result::SUCCESS);
    CHECK(state.interface.welcomeQuery == "spectrum");

    const Mail copy = query;
    CHECK(MailHiddenProducer::WelcomeQueryOf(copy) == "spectrum");
    CHECK(MailHiddenProducer::WelcomeQueryOf(MailSetWelcomeQuery{.query = "local"}) == "local");

    std::deque<Mail> pending;
    pending.push_back(MailHiddenProducer::FlowgraphLayout("graph", TileMeta{.pane = "flowgraph"}));
    REQUIRE(pending.front().is<MailSetFlowgraphLayout>());
    REQUIRE(actions.dispatch(pending) == Result::SUCCESS);
    REQUIRE(state.flowgraph.layouts.contains("graph"));
    CHECK(state.flowgraph.layouts.at("graph").pane == std::optional<std::string>("flowgraph"));

    CHECK(actions.handle(Mail{}) == Result::ERROR);
}
