#include "mail_hidden_producer.hh"

#include <utility>

namespace Jetstream::MailHiddenProducer {

Mail WelcomeQuery(std::string query) {
    return MailSetWelcomeQuery{.query = std::move(query)};
}

Mail FlowgraphLayout(std::string flowgraph, TileMeta tiles) {
    return MailSetFlowgraphLayout{.flowgraph = std::move(flowgraph), .tiles = std::move(tiles)};
}

std::string WelcomeQueryOf(const Mail& mail) {
    const auto* query = mail.getIf<MailSetWelcomeQuery>();
    return query != nullptr ? query->query : std::string{};
}

}  // namespace Jetstream::MailHiddenProducer
