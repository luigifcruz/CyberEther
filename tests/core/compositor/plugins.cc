#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/actions/plugins.hh"
#include "compositor/default/actions/workbench.hh"
#include "environment.hh"
#include "home_gallery_remote.hh"

#include <chrono>
#include <future>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace Jetstream;

namespace {

std::vector<char> Blob(const std::string& text) {
    return {text.begin(), text.end()};
}

Plugin::Gallery::Source Serve(std::map<std::string, std::string> files) {
    return {
        .send = [files = std::move(files)](const Platform::Http::Client::Request& request) {
            Platform::Http::Client::Response response;
            const auto file = files.find(request.host + request.path);
            response.status = file == files.end() ? 404 : 200;
            if (file != files.end()) {
                response.body = file->second;
            }
            return response;
        },
    };
}

template<typename T>
void Wait(const std::future<T>& future) {
    REQUIRE(future.valid());
    REQUIRE(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
}

}  // namespace

TEST_CASE("Flowgraphs that need missing plugins open the plugin prompt instead", "[core][compositor][plugins]") {
    CHECK(PluginActions::Missing(Blob("version: 2\ngraph: []\n")).empty());
    CHECK(PluginActions::Missing(Blob("version: [\n")).empty());
    CHECK(PluginActions::Missing(Blob("version: 3\nplugins: [nothere]\ngraph: []\n")).empty());
    CHECK(PluginActions::Missing(Blob("plugins: [nothere]\ngraph: []\n")) == std::vector<std::string>{"nothere"});
    CHECK(PluginActions::Missing(Blob("protocolVersion: 3\nplugins: [nothere]\ngraph: []\n")).empty());
    CHECK(PluginActions::Missing(Blob("protocolVersion: 1.0.0\nplugins: [nothere]\ngraph: {}\n")) ==
          std::vector<std::string>{"nothere"});
    CHECK(PluginActions::Missing(Blob("version: 2\nplugins: ['https://cyberether.org/gallery/plugins/x']\ngraph: []\n")).empty());
    CHECK(PluginActions::Missing(Blob("version: 2\nplugins: [nothere, other, nothere]\ngraph: []\n")) ==
          std::vector<std::string>{"nothere", "other"});

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    FlowgraphActions actions(state, callbacks);

    const auto blob = Blob("version: 2\nplugins: [nothere]\ngraph: []\n");
    REQUIRE(actions.handle(MailOpenFlowgraphBlob{.blob = blob}) == Result::SUCCESS);
    CHECK(state.modal.content == ModalContent::MissingPlugins);
    CHECK(state.plugins.missing == std::vector<std::string>{"nothere"});
    CHECK(state.plugins.blob == blob);
    CHECK(state.plugins.path.empty());
    CHECK(state.flowgraph.items.empty());
}

TEST_CASE("Gallery plugin installs report failures and the catalog loads in the background",
          "[core][compositor][plugins]") {
    TestSupport::SettingsSandbox sandbox("compositor-plugins");
    sandbox.require();

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<std::string> notices;
    std::vector<Mail> mails;
    callbacks.notifyResult = [&](Result, const std::string& message) {
        notices.push_back(message);
    };
    callbacks.notify = [&](Sakura::ToastType, I32, const std::string& message) {
        notices.push_back(message);
    };
    callbacks.enqueueMail = [&](Mail mail) {
        mails.push_back(std::move(mail));
    };

    const std::string host = Home::Gallery::Remote::Host;
    const std::string body = "{\"plugins\":[{\"id\":\"alpha\",\"title\":\"Alpha\"}]}";
    PluginActions actions(state, callbacks, Serve({
        {host + Plugin::Gallery::Path(Plugin::Gallery::Current()), body},
        {host + Plugin::Gallery::Path(Plugin::Gallery::Current({"alpha"})), body},
    }));
    state.plugins.checked = true;

    actions.reconcile();
    CHECK(state.plugins.catalog.loading);
    Wait(state.plugins.catalog.future);
    actions.reconcile();
    CHECK_FALSE(state.plugins.catalog.loading);
    CHECK_FALSE(state.plugins.catalog.failed);
    REQUIRE(state.plugins.catalog.entries.size() == 1);
    CHECK(state.plugins.catalog.entries.front().id == "alpha");
    CHECK_FALSE(state.plugins.catalog.entries.front().available());

    state.modal = {.content = ModalContent::MissingPlugins};
    REQUIRE(actions.handle(MailInstallGalleryPlugins{.ids = {"alpha"}, .reopen = true}) == Result::SUCCESS);
    CHECK_FALSE(state.modal.content.has_value());
    CHECK(state.plugins.install.loading);

    REQUIRE(actions.handle(MailInstallGalleryPlugins{.ids = {"beta"}}) == Result::SUCCESS);
    REQUIRE_FALSE(notices.empty());
    CHECK(notices.back() == "Another plugin is still being installed.");

    Wait(state.plugins.install.future);
    actions.reconcile();
    CHECK_FALSE(state.plugins.install.loading);
    REQUIRE(notices.size() == 2);
    CHECK(notices.back().find("does not have a build for CyberEther") != std::string::npos);
    CHECK(mails.empty());

    actions.shutdown();
}

TEST_CASE("Plugin details open and close in the side panel", "[core][compositor][plugins]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    PluginActions actions(state, callbacks);

    REQUIRE(actions.handle(MailOpenPluginDetails{.id = "alpha"}) == Result::SUCCESS);
    CHECK(state.plugins.details == "alpha");
    CHECK_FALSE(state.modal.content.has_value());

    REQUIRE(actions.handle(MailOpenPluginDetails{.id = "beta"}) == Result::SUCCESS);
    CHECK(state.plugins.details == "beta");

    REQUIRE(actions.handle(MailClosePluginDetails{}) == Result::SUCCESS);
    CHECK(state.plugins.details.empty());
}

TEST_CASE("The plugin update policy decides what happens to gallery updates", "[core][compositor][plugins]") {
    TestSupport::SettingsSandbox sandbox("compositor-plugin-updates");
    sandbox.require();

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<std::string> notices;
    callbacks.notifyResult = [&](Result, const std::string& message) {
        notices.push_back(message);
    };
    callbacks.notify = [&](Sakura::ToastType, I32, const std::string& message) {
        notices.push_back(message);
    };

    PluginActions actions(state, callbacks, Serve({}));
    state.plugins.checked = true;
    state.registry.plugins = {{.id = "alpha", .version = "1.0.0", .gallery = true}};
    state.registry.stale = false;

    const auto offer = [&](const std::string& policy) {
        state.plugins.updatePolicy = policy;
        std::promise<std::optional<std::vector<Plugin::Gallery::Entry>>> promise;
        promise.set_value(std::vector<Plugin::Gallery::Entry>{
            {.id = "alpha", .title = "Alpha", .version = "2.0.0", .url = "https://files.example/alpha.cep"},
        });
        state.plugins.updates.future = promise.get_future();
        state.plugins.updates.loading = true;
    };

    SECTION("prompt waits for the open modal and then asks") {
        offer("prompt");
        state.modal = {.content = ModalContent::Onboarding};
        actions.reconcile();
        REQUIRE(state.plugins.updates.entries.size() == 1);
        CHECK(state.plugins.prompt);
        CHECK(state.modal.content == ModalContent::Onboarding);
        CHECK_FALSE(state.plugins.update.loading);

        state.modal = {};
        state.interface.closingFlowgraph = "graph";
        actions.reconcile();
        CHECK_FALSE(state.modal.content.has_value());
        CHECK(state.plugins.prompt);

        state.interface.closingFlowgraph.reset();
        state.filePicker.active = true;
        actions.reconcile();
        CHECK_FALSE(state.modal.content.has_value());

        state.filePicker.active = false;
        state.plugins.update.loading = true;
        actions.reconcile();
        CHECK_FALSE(state.modal.content.has_value());
        CHECK(state.plugins.prompt);
        state.plugins.update.loading = false;

        state.plugins.install.loading = true;
        actions.reconcile();
        CHECK_FALSE(state.modal.content.has_value());
        CHECK(state.plugins.prompt);
        state.plugins.install.loading = false;

        state.registry.stale = true;
        actions.reconcile();
        CHECK_FALSE(state.modal.content.has_value());

        state.registry.stale = false;
        actions.reconcile();
        CHECK(state.modal.content == ModalContent::PluginUpdates);

        state.registry.plugins.front().pending = "2.0.0";
        actions.reconcile();
        CHECK_FALSE(state.modal.content.has_value());
        state.registry.plugins.front().pending.clear();
        state.modal = {.content = ModalContent::PluginUpdates};
        CHECK_FALSE(state.plugins.prompt);

        REQUIRE(actions.handle(MailUpdateGalleryPlugins{.ids = {"alpha"}}) == Result::SUCCESS);
        CHECK_FALSE(state.modal.content.has_value());
        CHECK(state.plugins.update.loading);
        CHECK(state.plugins.update.ids == std::vector<std::string>{"alpha"});

        REQUIRE(actions.handle(MailUpdateGalleryPlugins{.ids = {"alpha"}}) == Result::SUCCESS);
        REQUIRE_FALSE(notices.empty());
        CHECK(notices.back() == "Plugin updates are still being installed.");

        REQUIRE(actions.handle(MailInstallGalleryPlugins{.ids = {"alpha"}}) == Result::SUCCESS);
        CHECK(notices.back() == "Another plugin is still being installed.");
        CHECK_FALSE(state.plugins.install.loading);

        Wait(state.plugins.update.future);
        actions.reconcile();
        CHECK_FALSE(state.plugins.update.loading);
        CHECK(state.plugins.update.ids.empty());
        CHECK(state.plugins.updates.entries.empty());
    }

    SECTION("a deferred prompt follows a policy change") {
        offer("prompt");
        state.modal = {.content = ModalContent::Onboarding};
        actions.reconcile();
        CHECK(state.plugins.prompt);

        REQUIRE(actions.handle(MailSetPluginUpdatePolicy{.value = "manual"}) == Result::SUCCESS);
        state.modal = {};
        actions.reconcile();
        CHECK_FALSE(state.plugins.prompt);
        CHECK_FALSE(state.modal.content.has_value());
    }

    SECTION("prompt skips updates for plugins that left the gallery") {
        offer("prompt");
        state.registry.plugins.front().gallery = false;
        actions.reconcile();
        CHECK(state.plugins.updates.entries.size() == 1);
        CHECK_FALSE(state.plugins.prompt);
        CHECK_FALSE(state.modal.content.has_value());
    }

    SECTION("manual only remembers the update") {
        offer("manual");
        actions.reconcile();
        REQUIRE(state.plugins.updates.entries.size() == 1);
        CHECK_FALSE(state.plugins.prompt);
        CHECK_FALSE(state.modal.content.has_value());
        CHECK_FALSE(state.plugins.update.loading);
    }

    SECTION("auto installs the update right away") {
        offer("auto");
        actions.reconcile();
        CHECK_FALSE(state.plugins.prompt);
        CHECK_FALSE(state.modal.content.has_value());

        if (state.plugins.update.loading) {
            CHECK(state.plugins.update.ids == std::vector<std::string>{"alpha"});
            Wait(state.plugins.update.future);
            actions.reconcile();
        }
        CHECK_FALSE(state.plugins.update.loading);
        CHECK(state.plugins.update.ids.empty());
        CHECK(state.plugins.updates.entries.empty());
    }

    SECTION("the policy is validated and saved") {
        REQUIRE(actions.handle(MailSetPluginUpdatePolicy{.value = "sometimes"}) == Result::SUCCESS);
        CHECK(state.plugins.updatePolicy == "prompt");

        REQUIRE(actions.handle(MailSetPluginUpdatePolicy{.value = "manual"}) == Result::SUCCESS);
        CHECK(state.plugins.updatePolicy == "manual");
        REQUIRE(Settings::Flush() == Result::SUCCESS);
        Settings settings;
        REQUIRE(Settings::Get(settings) == Result::SUCCESS);
        CHECK(settings.plugins.updatePolicy == "manual");
    }

    actions.shutdown();
}
