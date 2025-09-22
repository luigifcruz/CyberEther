#include <jetstream/domains/visualization/note/block.hh>
#include <jetstream/detail/block_impl.hh>

namespace Jetstream::Blocks {

struct NoteImpl : public Block::Impl,
                  public DynamicConfig<Blocks::Note> {
    Result define() override;
    Result create() override;
};

Result NoteImpl::define() {
    JST_CHECK(defineInterfaceConfig("content",
                                    "Content",
                                    "Markdown text to display.",
                                    "multiline:collapsible"));

    JST_CHECK(defineInterfaceMetric("note",
                                    "",
                                    "Rendered markdown content.",
                                    "markdown",
        [this]() -> std::any {
            return content;
        }));

    return Result::SUCCESS;
}

Result NoteImpl::create() {
    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(NoteImpl);

}  // namespace Jetstream::Blocks
