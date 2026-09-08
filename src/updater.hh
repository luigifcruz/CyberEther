#ifndef JETSTREAM_UPDATER_HH
#define JETSTREAM_UPDATER_HH

#include "jetstream/types.hh"

#include <memory>
#include <string>

namespace Jetstream {

class Updater {
 public:
    struct Snapshot {
        bool supported = false;
        bool upToDate = false;
        bool failed = false;
        bool checking = false;
        bool available = false;
        bool downloading = false;
        bool ready = false;
        bool applying = false;
        F32 progress = 0.0f;
        std::string currentVersion;
        std::string version;
        std::string releaseNotes;
        std::string message;
    };

    Updater();
    ~Updater();

    Updater(const Updater&) = delete;
    Updater& operator=(const Updater&) = delete;

    static JETSTREAM_API void Initialize(int argc, char* argv[]);

    void start();
    Snapshot snapshot() const;
    void check();
    void download();
    bool apply(bool restart = true);
    void dismiss();
    void shutdown();

 private:
    struct Impl;
    std::shared_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_UPDATER_HH
