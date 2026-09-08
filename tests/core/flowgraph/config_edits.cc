#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <unordered_map>

#include "flowgraph_fixture.hh"

namespace {

using namespace Jetstream;

struct ConfigEditModuleConfig : Module::Config {
    F32 level = 0.5f;
    U64 size = 1;

    JST_MODULE_TYPE(config_edit_test_module);
    JST_MODULE_PARAMS(level, size);
};

struct ConfigEditModule : Module::Impl,
                          DynamicConfig<ConfigEditModuleConfig>,
                          NativeCpuRuntimeContext,
                          Scheduler::Context {
    using Module::Impl::requestConfigChange;
    using Module::Impl::configChangeEnabled;
    using Module::Impl::configChangePending;
    using Module::Impl::configChangeResult;

    Parser::Map presentEdit;
    bool failApply = false;
    bool failPresent = false;
    bool failDestroy = false;
    U64 reconfigurations = 0;
    U64 destructions = 0;
    Tensor output;

    Result validate() override {
        return std::isfinite(candidate()->level) && candidate()->level >= 0.0f &&
                       candidate()->level <= 1.0f && candidate()->size > 0
                   ? Result::SUCCESS : Result::ERROR;
    }
    Result define() override {
        JST_CHECK(defineTaint(Module::Taint::SURFACE));
        return defineInterfaceOutput("out");
    }
    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {size}));
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }
    Result computeSubmit() override { return Result::SUCCESS; }
    Result presentSubmit() override {
        if (failPresent) {
            failPresent = false;
            JST_ERROR("[CONFIG_EDIT_TEST] Forced presentation failure.");
            return Result::ERROR;
        }
        if (!presentEdit.empty()) {
            const auto edit = std::move(presentEdit);
            presentEdit.clear();
            return requestConfigChange(edit);
        }
        return Result::SUCCESS;
    }
    Result destroy() override {
        ++destructions;
        if (failDestroy) {
            failDestroy = false;
            JST_ERROR("[CONFIG_EDIT_TEST] Forced destruction failure.");
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }
    Result reconfigure() override {
        ++reconfigurations;
        if (failApply) {
            JST_ERROR("[CONFIG_EDIT_TEST] Forced apply failure.");
            return Result::ERROR;
        }
        if (candidate()->size != size) {
            return Result::RECREATE;
        }
        level = candidate()->level;
        return Result::SUCCESS;
    }
};

struct ConfigEditBlockConfig : Block::Config {
    F32 threshold = 0.5f;
    U64 size = 1;
    std::string caption = "before";
    bool editable = true;

    JST_BLOCK_TYPE(config_edit_test_block);
    JST_BLOCK_DOMAIN("Test");
    JST_BLOCK_PARAMS(threshold, size, caption, editable);
    JST_BLOCK_DESCRIPTION("Config Edit Test", "Tests upward edits.", "Tests upward edits.");
};

std::unordered_map<std::string, std::shared_ptr<Module>> editModules;

struct ConfigEditBlock : Block::Impl, DynamicConfig<ConfigEditBlockConfig> {
    std::shared_ptr<ConfigEditModuleConfig> config = std::make_shared<ConfigEditModuleConfig>();

    Result configure() override {
        config->level = threshold;
        config->size = size;
        return Result::SUCCESS;
    }
    Result create() override {
        JST_CHECK(moduleCreate("control", config, {}));
        JST_CHECK(moduleCreate("mirror", config, {}));
        editModules[name()] = moduleHandle("control");
        editModules[name() + "-mirror"] = moduleHandle("mirror");
        if (editable) {
            JST_CHECK(moduleBindConfigEdit("control", "level", "threshold"));
            JST_CHECK(moduleBindConfigEdit("control", "size", "size"));
        }
        return Result::SUCCESS;
    }
};

JST_REGISTER_MODULE(ConfigEditModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_BLOCK(ConfigEditBlock, {"config_edit_test_module"});

ConfigEditModule* Control(const std::string& name = "edit") {
    return editModules.at(name)->getImpl<ConfigEditModule>();
}

Parser::Map Level(F32 level) {
    Parser::Map edit;
    edit["level"] = level;
    return edit;
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "Module edits are deferred, mapped, coalesced and applied to all children",
                 "[core][flowgraph][config-edits]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    auto* control = Control();
    const auto original = editModules.at("edit");
    REQUIRE(control->configChangeEnabled("level"));
    REQUIRE_FALSE(Control("edit-mirror")->configChangeEnabled("level"));

    // Exercise the request while the scheduler owns its present locks.
    control->presentEdit = Level(0.2f);
    REQUIRE(flowgraph->present() == Result::SUCCESS);
    REQUIRE(control->configChangePending());
    REQUIRE(control->level == 0.5f);
    REQUIRE(std::any_cast<F32>(viewBlock("edit").config.at("threshold")) == 0.5f);
    REQUIRE(control->requestConfigChange(Level(0.3f)) == Result::SUCCESS);

    Parser::Map unrelated;
    unrelated["caption"] = std::string("after");
    REQUIRE(flowgraph->blockReconfigure("edit", unrelated) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE_FALSE(control->configChangePending());
    REQUIRE(control->configChangeResult() == Result::SUCCESS);
    REQUIRE(editModules.at("edit") == original);
    REQUIRE(control->level == 0.3f);
    REQUIRE(Control("edit-mirror")->level == 0.3f);
    const auto saved = viewBlock("edit").config;
    REQUIRE(std::any_cast<F32>(saved.at("threshold")) == 0.3f);
    REQUIRE(std::any_cast<std::string>(saved.at("caption")) == "after");
    REQUIRE(control->reconfigurations == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Unbound module edits reject the whole patch",
                 "[core][flowgraph][config-edits]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    auto* control = Control();
    REQUIRE(control->requestConfigChange(Level(0.2f)) == Result::SUCCESS);
    auto invalid = Level(0.3f);
    invalid["caption"] = std::string("not a module field");
    REQUIRE(control->requestConfigChange(invalid) == Result::ERROR);
    REQUIRE(Control("edit-mirror")->requestConfigChange(Level(0.4f)) == Result::ERROR);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(control->level == 0.2f);
    REQUIRE(std::any_cast<std::string>(viewBlock("edit").config.at("caption")) == "before");
}

TEST_CASE_METHOD(FlowgraphFixture, "Module edits follow validation recreation and rollback contracts",
                 "[core][flowgraph][config-edits]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    const auto original = editModules.at("edit");
    auto* control = Control();

    SECTION("semantic errors publish the invalid candidate") {
        REQUIRE(control->requestConfigChange(Level(-1.0f)) == Result::SUCCESS);
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
        REQUIRE(viewBlock("edit").state == Block::State::Errored);
        REQUIRE(std::any_cast<F32>(viewBlock("edit").config.at("threshold")) == -1.0f);
        REQUIRE_FALSE(control->configChangePending());
        REQUIRE(control->configChangeResult() == Result::ERROR);
    }
    SECTION("apply failures restore the previous graph") {
        control->failApply = true;
        REQUIRE(control->requestConfigChange(Level(0.2f)) == Result::SUCCESS);
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
        REQUIRE(viewBlock("edit").state == Block::State::Created);
        REQUIRE(std::any_cast<F32>(viewBlock("edit").config.at("threshold")) == 0.5f);
        REQUIRE(control->configChangeResult() == Result::ERROR);
        REQUIRE(Control()->level == 0.5f);
    }
    SECTION("structural edits recreate") {
        Parser::Map edit;
        edit["size"] = U64{4};
        REQUIRE(control->requestConfigChange(edit) == Result::SUCCESS);
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
        REQUIRE(editModules.at("edit") != original);
        REQUIRE(Control()->size == 4);
        REQUIRE(Control("edit-mirror")->size == 4);
        REQUIRE(control->configChangeResult() == Result::SUCCESS);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Old module edits cannot affect recreated or renamed blocks",
                 "[core][flowgraph][config-edits]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    const auto original = editModules.at("edit");
    auto* control = Control();
    REQUIRE(control->requestConfigChange(Level(0.2f)) == Result::SUCCESS);
    std::string target = "edit";
    SECTION("recreate") {
        REQUIRE(flowgraph->blockRecreate("edit", viewBlock("edit").config) == Result::SUCCESS);
    }
    SECTION("rename") {
        target = "renamed";
        REQUIRE(flowgraph->blockRename("edit", target) == Result::SUCCESS);
    }
    SECTION("delete and reuse name") {
        REQUIRE(flowgraph->blockDestroy("edit") == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    }
    REQUIRE_FALSE(control->configChangePending());
    REQUIRE(control->requestConfigChange(Level(0.1f)) == Result::ERROR);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(Control(target)->level == 0.5f);
}

TEST_CASE_METHOD(FlowgraphFixture, "Saving flushes queued module edits without a compute cycle",
                 "[core][flowgraph][config-edits][serialization]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    REQUIRE(Control()->requestConfigChange(Level(0.25f)) == Result::SUCCESS);
    std::vector<char> blob;
    REQUIRE(flowgraph->exportToBlob(blob) == Result::SUCCESS);
    REQUIRE_FALSE(Control()->configChangePending());

    Flowgraph restored;
    REQUIRE(restored.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(restored.importFromBlob(blob) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(ViewBlock(restored, "edit").config.at("threshold")) == 0.25f);
    REQUIRE(Control()->level == 0.25f);
    REQUIRE(restored.destroy() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture, "Permanent detachment cancels edits even when destruction fails",
                 "[core][flowgraph][config-edits][detachment]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("healthy", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    const auto retained = editModules.at("edit");
    auto* control = Control();

    control->presentEdit = Level(0.2f);
    REQUIRE(flowgraph->present() == Result::SUCCESS);
    REQUIRE(control->configChangePending());
    REQUIRE(Control("healthy")->requestConfigChange(Level(0.8f)) == Result::SUCCESS);

    control->failPresent = true;
    control->failDestroy = true;
    REQUIRE(flowgraph->present() == Result::SUCCESS);
    REQUIRE(control->destructions == 1);
    REQUIRE(retained->state() == Module::State::ERRORED);
    REQUIRE(viewBlock("edit").state == Block::State::Errored);
    REQUIRE(viewBlock("edit").surfaces.empty());
    CHECK_FALSE(control->configChangePending());
    CHECK_FALSE(control->configChangeEnabled("level"));
    CHECK_FALSE(control->configChangeEnabled("size"));
    CHECK(control->configChangeResult() == Result::ERROR);
    CHECK(control->requestConfigChange(Level(0.3f)) == Result::ERROR);
    CHECK(control->requestConfigChange({}) == Result::ERROR);

    // Cancelling one channel must not clear the shared notification for peers.
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(Control("healthy")->level == 0.8f);
    REQUIRE_FALSE(Control("healthy")->configChangePending());
    CHECK_FALSE(control->configChangePending());
    REQUIRE(std::any_cast<F32>(viewBlock("edit").config.at("threshold")) == 0.5f);

    // Retrying resource cleanup must not reopen the detached edit channel.
    REQUIRE(retained->destroy() == Result::SUCCESS);
    REQUIRE_FALSE(control->configChangeEnabled("level"));
    REQUIRE(control->requestConfigChange(Level(0.4f)) == Result::ERROR);
}

TEST_CASE_METHOD(FlowgraphFixture, "Failed destruction with retained ownership preserves the edit channel",
                 "[core][flowgraph][config-edits][detachment]") {
    REQUIRE(flowgraph->blockCreate("edit", ConfigEditBlockConfig{}, {}) == Result::SUCCESS);
    const auto retained = editModules.at("edit");
    auto* control = Control();
    REQUIRE(control->requestConfigChange(Level(0.2f)) == Result::SUCCESS);
    control->failDestroy = true;

    REQUIRE(flowgraph->blockDestroy("edit", false) == Result::ERROR);
    REQUIRE(control->destructions == 1);
    REQUIRE(retained->state() == Module::State::ERRORED);
    REQUIRE(viewBlock("edit").state == Block::State::Errored);
    REQUIRE(control->configChangePending());
    REQUIRE(control->configChangeEnabled("level"));
    REQUIRE(control->configChangeResult() == Result::SUCCESS);
    REQUIRE(control->requestConfigChange(Level(0.3f)) == Result::SUCCESS);

    // The owner can still drain the replacement edit and recover the block.
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("edit").state == Block::State::Created);
    REQUIRE(editModules.at("edit") != retained);
    REQUIRE(Control()->level == 0.3f);
    REQUIRE_FALSE(control->configChangePending());
    REQUIRE_FALSE(control->configChangeEnabled("level"));
    REQUIRE(control->configChangeResult() == Result::SUCCESS);
}
