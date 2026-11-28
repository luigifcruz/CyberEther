#ifndef JETSTREAM_TESTS_CORE_COMPOSITOR_MAIL_HIDDEN_PRODUCER_HH
#define JETSTREAM_TESTS_CORE_COMPOSITOR_MAIL_HIDDEN_PRODUCER_HH

#include "compositor/default/model/messages.hh"

#include <string>

namespace Jetstream::MailHiddenProducer {

Mail WelcomeQuery(std::string query);
Mail FlowgraphLayout(std::string flowgraph, TileMeta tiles);
std::string WelcomeQueryOf(const Mail& mail);

}  // namespace Jetstream::MailHiddenProducer

#endif  // JETSTREAM_TESTS_CORE_COMPOSITOR_MAIL_HIDDEN_PRODUCER_HH
