#ifndef JETSTREAM_SETTINGS_HH
#define JETSTREAM_SETTINGS_HH

#include <string>

#include "jetstream/parser.hh"

namespace Jetstream {

class JETSTREAM_API Settings {
 public:
    template<typename T>
    static Result Load(const std::string& key, T& value) {
        Parser::Map data;
        JST_CHECK(LoadData(key, data));
        return Parser::Deserialize(data, key, value);
    }

    template<typename T>
    static Result Save(const std::string& key, const T& value) {
        Parser::Map data;
        JST_CHECK(LoadData(key, data));
        JST_CHECK(Parser::Serialize(data, key, value));
        return SaveData(key, data);
    }

    static Result Erase(const std::string& key);

 private:
    static Result LoadData(const std::string& key, Parser::Map& data);
    static Result SaveData(const std::string& key, const Parser::Map& data);
};

}  // namespace Jetstream

#endif  // JETSTREAM_SETTINGS_HH
