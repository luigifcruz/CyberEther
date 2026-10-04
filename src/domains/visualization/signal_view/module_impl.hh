#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_IMPL_HH

#include <jetstream/domains/visualization/signal_view/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/tensor.hh>

#include "common.hh"
#include "lineplot.hh"
#include "waterfall.hh"
#include "waterfall_3d.hh"

namespace Jetstream::Modules {

struct SignalViewImpl : public Module::Impl,
                        public DynamicConfig<SignalView> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    U64 inputElementStride = 0;
    U64 inputBatchStride = 0;
    U64 maxHoldWarmupBlocks = 0;
    F32 normalizationFactor = 0.0f;
    bool lineplotEnabled = false;
    bool lineplotAveragingInitialized = false;
    bool waterfallEnabled = false;
    bool waterfall3dEnabled = false;
    U64 waterfallAveragingCount = 0;

    U64 validatedNumberOfElements = 0;
    U64 validatedNumberOfBatches = 0;
    U64 validatedInputElementStride = 0;
    U64 validatedInputBatchStride = 0;
    F32 validatedNormalizationFactor = 0.0f;
    bool validatedLineplotEnabled = false;
    bool validatedWaterfallEnabled = false;
    bool validatedWaterfall3dEnabled = false;

    Tensor signalPoints;
    Tensor signalVertices;
    Tensor fillVertices;
    Tensor maxHoldPoints;
    Tensor maxHoldVertices;

    bool updateSignalPointsFlag = false;
    bool updateHoldPointsFlag = false;

    Tensor waterfallBins;
    WaterfallHistory waterfallHistory;

    SignalViewCanvas canvas;
    SignalViewLineplot lineplot;
    SignalViewWaterfall waterfall;
    SignalViewWaterfall3D waterfall3d;
    bool updateLayoutFlag = false;

    Result createPresent();
    Result destroyPresent();
    Result present();

    SignalViewCanvas::Context canvasContext();
    WaterfallFrame waterfallFrame() const;
    SignalViewWaterfall3DLabels waterfall3dLabels() const;
    Result resetLineplotHistory();
    Result resetHistoryState();
    virtual Buffer::Config renderStateBufferConfig() const = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_IMPL_HH
