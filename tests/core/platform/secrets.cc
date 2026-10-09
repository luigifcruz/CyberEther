#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/platform.hh"

using namespace Jetstream;

namespace {

constexpr const char* kOptInVariable = "CYBERETHER_TEST_SECRET_STORE";
constexpr const char* kServicePrefix = "ltd.luigi.cyberether.test.secrets.";
constexpr const char* kUntouched = "untouched-sentinel";
constexpr const char* kAccount = "account";
constexpr std::size_t kLongValueSize = 2048;
constexpr std::size_t kManyEntries = 24;

std::string RandomToken() {
    std::random_device device;
    const auto nonce = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::uint64_t random = (static_cast<std::uint64_t>(device()) << 32) ^ device();

    static constexpr char digits[] = "0123456789abcdef";
    std::string token;
    for (int shift = 60; shift >= 0; shift -= 4) {
        token.push_back(digits[(random >> shift) & 0xF]);
    }
    token.push_back('-');
    for (int shift = 60; shift >= 0; shift -= 4) {
        token.push_back(digits[(nonce >> shift) & 0xF]);
    }
    return token;
}

std::string UniqueService(const std::string& label) {
    return kServicePrefix + label + "." + RandomToken();
}

bool RoundTripRequested() {
    std::string value;
    if (Platform::EnvironmentVariable(kOptInVariable, value) != Result::SUCCESS) {
        return false;
    }
    return value == "1";
}

class SecretScope {
 public:
    SecretScope() = default;
    ~SecretScope() {
        for (const auto& [service, account] : keys) {
            (void)Platform::DeleteSecret(service, account);
        }
    }

    SecretScope(const SecretScope&) = delete;
    SecretScope& operator=(const SecretScope&) = delete;

    void track(const std::string& service, const std::string& account) {
        keys.emplace_back(service, account);
    }

    Result write(const std::string& service, const std::string& account, const std::string& value) {
        track(service, account);
        return Platform::WriteSecret(service, account, value);
    }

 private:
    std::vector<std::pair<std::string, std::string>> keys;
};

void RequireRoundTripStore() {
    if (!RoundTripRequested()) {
        SKIP("Secret store round trips are opt-in: set " << kOptInVariable << "=1");
    }
    if (!Platform::SecretStoreAvailable()) {
        SKIP("Secret store is unavailable on this machine");
    }
}

std::string ReadOrSentinel(const std::string& service, const std::string& account, Result& result) {
    std::string value = kUntouched;
    result = Platform::ReadSecret(service, account, value);
    return value;
}

}  // namespace

TEST_CASE("Platform secrets reject empty keys", "[core][platform][secrets]") {
    const std::string service = UniqueService("empty-keys");

    struct Case {
        const char* label;
        std::string service;
        std::string account;
    };

    const Case cases[] = {
        {"empty service", "", kAccount},
        {"empty account", service, ""},
        {"both empty", "", ""},
    };

    for (const auto& item : cases) {
        INFO(item.label);

        std::string value = kUntouched;
        CHECK(Platform::ReadSecret(item.service, item.account, value) == Result::ERROR);
        CHECK(value == kUntouched);

        CHECK(Platform::WriteSecret(item.service, item.account, "value") == Result::ERROR);
        CHECK(Platform::DeleteSecret(item.service, item.account) == Result::ERROR);
    }
}

TEST_CASE("Platform secret reads leave the output untouched on failure", "[core][platform][secrets]") {
    const std::string service = UniqueService("untouched");

    SECTION("missing key") {
        std::string value = kUntouched;
        CHECK(Platform::ReadSecret(service, kAccount, value) == Result::ERROR);
        CHECK(value == kUntouched);
    }

    SECTION("invalid key") {
        std::string value = kUntouched;
        CHECK(Platform::ReadSecret("", "", value) == Result::ERROR);
        CHECK(value == kUntouched);
    }

    SECTION("empty output stays empty") {
        std::string value;
        CHECK(Platform::ReadSecret(service, kAccount, value) == Result::ERROR);
        CHECK(value.empty());
    }
}

TEST_CASE("Platform secret calls agree with store availability", "[core][platform][secrets]") {
    const std::string service = UniqueService("availability");

    if (!Platform::SecretStoreAvailable()) {
        std::string value = kUntouched;
        CHECK(Platform::ReadSecret(service, kAccount, value) == Result::ERROR);
        CHECK(value == kUntouched);
        CHECK(Platform::WriteSecret(service, kAccount, "value") == Result::ERROR);
        CHECK(Platform::DeleteSecret(service, kAccount) == Result::ERROR);
        return;
    }

    CHECK(Platform::DeleteSecret(service, kAccount) == Result::SUCCESS);

    std::string value = kUntouched;
    CHECK(Platform::ReadSecret(service, kAccount, value) == Result::ERROR);
    CHECK(value == kUntouched);
}

TEST_CASE("Platform secrets round trip a value", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("round-trip");

    REQUIRE(scope.write(service, kAccount, "first-value") == Result::SUCCESS);

    Result result = Result::ERROR;
    CHECK(ReadOrSentinel(service, kAccount, result) == "first-value");
    CHECK(result == Result::SUCCESS);
}

TEST_CASE("Platform secrets overwrite an existing value", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("overwrite");

    REQUIRE(scope.write(service, kAccount, "first-value") == Result::SUCCESS);
    REQUIRE(scope.write(service, kAccount, "second-value") == Result::SUCCESS);

    Result result = Result::ERROR;
    CHECK(ReadOrSentinel(service, kAccount, result) == "second-value");
    CHECK(result == Result::SUCCESS);

    REQUIRE(scope.write(service, kAccount, "short") == Result::SUCCESS);
    CHECK(ReadOrSentinel(service, kAccount, result) == "short");
    CHECK(result == Result::SUCCESS);
}

TEST_CASE("Platform secrets delete a stored value", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("delete");

    REQUIRE(scope.write(service, kAccount, "value") == Result::SUCCESS);
    REQUIRE(Platform::DeleteSecret(service, kAccount) == Result::SUCCESS);

    Result result = Result::SUCCESS;
    CHECK(ReadOrSentinel(service, kAccount, result) == kUntouched);
    CHECK(result == Result::ERROR);
}

TEST_CASE("Platform secrets delete is idempotent", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("delete-twice");

    SECTION("twice after a write") {
        REQUIRE(scope.write(service, kAccount, "value") == Result::SUCCESS);
        CHECK(Platform::DeleteSecret(service, kAccount) == Result::SUCCESS);
        CHECK(Platform::DeleteSecret(service, kAccount) == Result::SUCCESS);
    }

    SECTION("never written") {
        CHECK(Platform::DeleteSecret(service, kAccount) == Result::SUCCESS);
    }
}

TEST_CASE("Platform secrets preserve value contents", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("contents");

    struct Case {
        const char* label;
        std::string value;
    };

    const Case cases[] = {
        {"utf8 multibyte", "chave \xC3\xA7\xC3\xA3o \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x94\x90"},
        {"spaces", "  leading and trailing spaces  "},
        {"newlines", "line one\nline two\r\nline three\n"},
        {"quotes", "He said \"hi\" and 'bye'"},
        {"backslashes", "C:\\Users\\name\\\\double \\n not a newline"},
        {"tabs and control bytes", "tab\tseparated\x01" "\x1F" "control"},
        {"key value shape", "sk-live-ABCdef0123456789=/+"},
    };

    std::size_t index = 0;
    for (const auto& item : cases) {
        INFO(item.label);
        const std::string account = "value-" + std::to_string(index++);

        REQUIRE(scope.write(service, account, item.value) == Result::SUCCESS);

        Result result = Result::ERROR;
        CHECK(ReadOrSentinel(service, account, result) == item.value);
        CHECK(result == Result::SUCCESS);
    }
}

TEST_CASE("Platform secrets reject an empty value", "[core][platform][secrets]") {
    const std::string service = UniqueService("empty-value-rejected");

    CHECK(Platform::WriteSecret(service, kAccount, "") == Result::ERROR);

    Result result = Result::SUCCESS;
    CHECK(ReadOrSentinel(service, kAccount, result) == kUntouched);
    CHECK(result == Result::ERROR);
}

TEST_CASE("Platform secrets keep the stored value when an empty write is rejected",
          "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("empty-value");

    SECTION("fresh key") {
        CHECK(scope.write(service, kAccount, "") == Result::ERROR);

        Result result = Result::SUCCESS;
        CHECK(ReadOrSentinel(service, kAccount, result) == kUntouched);
        CHECK(result == Result::ERROR);
    }

    SECTION("existing key") {
        REQUIRE(scope.write(service, kAccount, "placeholder") == Result::SUCCESS);
        CHECK(scope.write(service, kAccount, "") == Result::ERROR);

        Result result = Result::ERROR;
        CHECK(ReadOrSentinel(service, kAccount, result) == "placeholder");
        CHECK(result == Result::SUCCESS);
    }
}

TEST_CASE("Platform secrets store a long value", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("long-value");

    std::string value;
    value.reserve(kLongValueSize);
    for (std::size_t index = 0; value.size() < kLongValueSize; ++index) {
        value.push_back(static_cast<char>('!' + (index % 94)));
    }
    REQUIRE(value.size() == kLongValueSize);

    REQUIRE(scope.write(service, kAccount, value) == Result::SUCCESS);

    Result result = Result::ERROR;
    CHECK(ReadOrSentinel(service, kAccount, result) == value);
    CHECK(result == Result::SUCCESS);
}

#if defined(JST_OS_WINDOWS)
TEST_CASE("Platform secrets reject values over the Credential Manager limit",
          "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("oversized-value");

    const std::string value(5 * 512 + 1, 'x');
    CHECK(scope.write(service, kAccount, value) == Result::ERROR);

    Result result = Result::SUCCESS;
    CHECK(ReadOrSentinel(service, kAccount, result) == kUntouched);
    CHECK(result == Result::ERROR);
}
#endif

TEST_CASE("Platform secrets keep accounts apart under one service", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("accounts");

    REQUIRE(scope.write(service, "alice", "alice-secret") == Result::SUCCESS);
    REQUIRE(scope.write(service, "bob", "bob-secret") == Result::SUCCESS);

    Result result = Result::ERROR;
    CHECK(ReadOrSentinel(service, "alice", result) == "alice-secret");
    CHECK(result == Result::SUCCESS);
    CHECK(ReadOrSentinel(service, "bob", result) == "bob-secret");
    CHECK(result == Result::SUCCESS);

    REQUIRE(Platform::DeleteSecret(service, "alice") == Result::SUCCESS);
    CHECK(ReadOrSentinel(service, "alice", result) == kUntouched);
    CHECK(result == Result::ERROR);
    CHECK(ReadOrSentinel(service, "bob", result) == "bob-secret");
    CHECK(result == Result::SUCCESS);
}

TEST_CASE("Platform secrets keep services apart under one account", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string first = UniqueService("services-first");
    const std::string second = UniqueService("services-second");

    REQUIRE(scope.write(first, kAccount, "first-secret") == Result::SUCCESS);
    REQUIRE(scope.write(second, kAccount, "second-secret") == Result::SUCCESS);

    Result result = Result::ERROR;
    CHECK(ReadOrSentinel(first, kAccount, result) == "first-secret");
    CHECK(result == Result::SUCCESS);
    CHECK(ReadOrSentinel(second, kAccount, result) == "second-secret");
    CHECK(result == Result::SUCCESS);

    REQUIRE(Platform::DeleteSecret(first, kAccount) == Result::SUCCESS);
    CHECK(ReadOrSentinel(first, kAccount, result) == kUntouched);
    CHECK(result == Result::ERROR);
    CHECK(ReadOrSentinel(second, kAccount, result) == "second-secret");
    CHECK(result == Result::SUCCESS);
}

TEST_CASE("Platform secrets accept UTF-8 and punctuation in keys", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string base = UniqueService("keys");

    struct Case {
        const char* label;
        std::string service;
        std::string account;
    };

    const Case cases[] = {
        {"utf8 service and account", base + ".servi\xC3\xA7o \xE6\x9C\x8D\xE5\x8B\x99", "usu\xC3\xA1rio \xE7\x94\xA8\xE6\x88\xB7"},
        {"slashes", base + "/api/v1", "team/user/token"},
        {"backslashes", base + "\\domain", "domain\\user"},
        {"spaces and punctuation", base + " (staging) [eu-west]", "user@example.com: key #1, v2!"},
        {"url shaped", base + "://host:8443/path?x=1&y=2", "https://example.com/callback"},
        {"colons and semicolons", base + ":section;part", "a:b;c"},
    };

    for (const auto& item : cases) {
        INFO(item.label);
        const std::string expected = std::string("secret for ") + item.label;

        REQUIRE(scope.write(item.service, item.account, expected) == Result::SUCCESS);

        Result result = Result::ERROR;
        CHECK(ReadOrSentinel(item.service, item.account, result) == expected);
        CHECK(result == Result::SUCCESS);
    }

    Result result = Result::ERROR;
    CHECK(ReadOrSentinel(cases[1].service, cases[2].account, result) == kUntouched);
    CHECK(result == Result::ERROR);
    CHECK(ReadOrSentinel(cases[2].service, cases[1].account, result) == kUntouched);
    CHECK(result == Result::ERROR);
}

TEST_CASE("Platform secrets hold many entries", "[core][platform][secrets][store]") {
    RequireRoundTripStore();
    SecretScope scope;
    const std::string service = UniqueService("many");

    std::vector<std::pair<std::string, std::string>> entries;
    for (std::size_t index = 0; index < kManyEntries; ++index) {
        entries.emplace_back("account-" + std::to_string(index),
                             "value-" + std::to_string(index) + "-" + std::string(index, 'x'));
    }

    for (const auto& [account, value] : entries) {
        REQUIRE(scope.write(service, account, value) == Result::SUCCESS);
    }

    for (const auto& [account, value] : entries) {
        INFO(account);
        Result result = Result::ERROR;
        CHECK(ReadOrSentinel(service, account, result) == value);
        CHECK(result == Result::SUCCESS);
    }

    for (std::size_t index = 0; index < entries.size(); index += 2) {
        REQUIRE(Platform::DeleteSecret(service, entries[index].first) == Result::SUCCESS);
    }

    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& [account, value] = entries[index];
        INFO(account);
        Result result = Result::ERROR;
        const std::string read = ReadOrSentinel(service, account, result);
        if (index % 2 == 0) {
            CHECK(read == kUntouched);
            CHECK(result == Result::ERROR);
        } else {
            CHECK(read == value);
            CHECK(result == Result::SUCCESS);
        }
    }
}
