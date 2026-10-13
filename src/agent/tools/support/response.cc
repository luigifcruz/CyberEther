#include "base.hh"

#include <algorithm>

namespace Jetstream::Agent::Detail {

Json Response::Error(const std::string& message) {
    return Json({{"error", message}});
}

bool Response::Failed(const Json& body) {
    if (!body.is_object()) {
        return false;
    }

    if (body.contains("error")) {
        return true;
    }

    if (body.contains("status") && body["status"] == "partial") {
        return true;
    }

    if (!body.contains("results") || !body["results"].is_array()) {
        return false;
    }

    const auto& results = body["results"];
    return std::any_of(results.begin(), results.end(), [](const Json& result) {
        return result.is_object() && result.contains("error");
    });
}

Tools::Output Response::Reply(const Json& body, std::vector<std::string> images) {
    return {body.dump(), std::move(images), Failed(body)};
}

Tools::Output Response::Failure(const std::string& message) {
    return Reply(Error(message));
}

}  // namespace Jetstream::Agent::Detail
