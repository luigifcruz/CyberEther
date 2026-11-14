#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH

#include "bool.hh"
#include "console.hh"
#include "dropdown.hh"
#include "float.hh"
#include "int.hh"
#include "markdown.hh"
#include "multiline.hh"
#include "path.hh"
#include "python.hh"
#include "range.hh"
#include "tensor.hh"
#include "text.hh"
#include "types.hh"
#include "uint.hh"
#include "vector.hh"
#include "vector_inline.hh"

#include "../field_tooltip.hh"

#include <memory>
#include <unordered_map>
#include <vector>

namespace Jetstream {

struct FlowgraphConfigFieldInstance {
    enum class Kind {
        Unknown,
        Dropdown,
        Float,
        Int,
        UInt,
        Vector,
        VectorInline,
        Path,
        Bool,
        Range,
        Tensor,
        Markdown,
        Python,
        PythonConsole,
        Multiline,
        Text,
    };

    FlowgraphNodeHeightSpec heightSpec() const;
    void update(FlowgraphConfigFieldConfig config);
    bool isSimple() const;
    Sakura::Component& component();

 private:
    static Kind ParseKind(const std::string& value);

    template<typename T>
    static T& ensure(std::unique_ptr<T>& field) {
        if (!field) {
            field = std::make_unique<T>();
        }
        return *field;
    }

    void reset();

    Kind kind = Kind::Unknown;
    bool failed = false;
    std::string builtKind;
    std::string builtId;
    std::string notifiedDecodeError;
    bool fullHeight = false;
    std::unique_ptr<FlowgraphConfigDropdownField> dropdown;
    std::unique_ptr<FlowgraphConfigFloatField> floatField;
    std::unique_ptr<FlowgraphConfigIntField> intField;
    std::unique_ptr<FlowgraphConfigUIntField> uintField;
    std::unique_ptr<FlowgraphConfigVectorField> vectorField;
    std::unique_ptr<FlowgraphConfigVectorInlineField> vectorInline;
    std::unique_ptr<FlowgraphConfigPathField> path;
    std::unique_ptr<FlowgraphConfigBoolField> boolField;
    std::unique_ptr<FlowgraphConfigRangeField> range;
    std::unique_ptr<FlowgraphConfigTensorField> tensor;
    std::unique_ptr<FlowgraphConfigMarkdownField> markdown;
    std::unique_ptr<FlowgraphConfigPythonField> python;
    std::unique_ptr<FlowgraphConfigPythonConsoleField> pythonConsole;
    std::unique_ptr<FlowgraphConfigMultilineField> multiline;
    std::unique_ptr<FlowgraphConfigTextField> text;
    FlowgraphFieldTooltip unknownHelp;
    Sakura::NodeField unknownFrame;
    Sakura::NodeLabel unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
