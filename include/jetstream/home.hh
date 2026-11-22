#ifndef JETSTREAM_HOME_HH
#define JETSTREAM_HOME_HH

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "jetstream/parser.hh"
#include "jetstream/types.hh"

namespace Jetstream {

class JETSTREAM_API Home {
 public:
    class JETSTREAM_API Gallery {
     public:
        class Remote;

        struct JETSTREAM_API Flowgraph {
            std::string key;
            std::string title;
            std::string summary = "No summary.";
            std::string description = "No description.";
            std::string category;
            std::string accent;
            std::string diagram;
            bool featured = false;
            std::string content;

            JST_SERDES(title, summary, description, category, accent, diagram, featured);

            static Result Parse(const std::string& key, std::string content, Flowgraph& flowgraph);
        };

        struct Section {
            std::string key;
            std::string title;
            std::string description;
            std::vector<Flowgraph> flowgraphs;
        };

        static constexpr const char* Origin = "cyberether";

        Gallery();
        ~Gallery();

        Gallery(const Gallery&) = delete;
        Gallery& operator=(const Gallery&) = delete;

        Result publish(const std::string& origin, std::vector<Section> sections);
        std::vector<Section> sections() const;
        std::optional<Flowgraph> flowgraph(const std::string& key) const;
        U64 revision() const;

     private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    static Home& Get();

    Gallery& gallery();

 private:
    Home();
    ~Home();

    Home(const Home&) = delete;
    Home& operator=(const Home&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_HOME_HH
