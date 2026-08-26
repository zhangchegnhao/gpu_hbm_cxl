#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class SievePIMFrontend final : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, SievePIMFrontend, "SievePIM")

 private:
  std::string m_operation;
  int m_waves = 0;
  int m_channels = 0;
  int m_pseudo_channels = 0;
  int m_row_span_waves = 32;
  int m_request_type = -1;
  int m_current_wave = 0;
  std::vector<bool> m_sent;

  size_t s_injected_requests = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_operation, std::string, "operation").required();
    RAMULATOR_PARSE_PARAM(m_waves, int, "waves").required();
    RAMULATOR_PARSE_PARAM(m_channels, int, "channels").required();
    RAMULATOR_PARSE_PARAM(m_pseudo_channels, int, "pseudo_channels").required();
    RAMULATOR_PARSE_PARAM(m_row_span_waves, int, "row_span_waves").default_val(32);

    if (m_waves <= 0 || m_channels <= 0 || m_pseudo_channels <= 0 || m_row_span_waves <= 0) {
      throw std::runtime_error("SievePIM dimensions must be positive");
    }
    if (m_operation == "PIM_GWRITE") {
      m_request_type = Request::Type::PIM_GWRITE;
    } else if (m_operation == "PIM_MAC") {
      m_request_type = Request::Type::PIM_MAC;
    } else if (m_operation == "PIM_READ") {
      m_request_type = Request::Type::PIM_READ;
    } else {
      throw std::runtime_error(fmt::format("unsupported Sieve PIM operation: {}", m_operation));
    }
    m_sent.resize(static_cast<size_t>(m_channels * m_pseudo_channels), false);
    m_stats.add("injected_requests", s_injected_requests);
  }

  void tick() override {
    if (is_finished()) {
      return;
    }

    for (int channel = 0; channel < m_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels; pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(channel * m_pseudo_channels + pseudo_channel);
        if (m_sent[endpoint]) {
          continue;
        }
        const int row = m_current_wave / m_row_span_waves;
        const int column = m_current_wave % m_row_span_waves;
        Request request(
            AddrVec_t{channel, pseudo_channel, 0, 0, 0, row, column}, m_request_type);
        request.size_bytes = m_memory_system->get_tx_bytes();
        if (m_memory_system->send(request)) {
          m_sent[endpoint] = true;
          s_injected_requests++;
        }
      }
    }

    if (std::all_of(m_sent.begin(), m_sent.end(), [](bool sent) { return sent; })) {
      m_current_wave++;
      std::fill(m_sent.begin(), m_sent.end(), false);
    }
  }

  bool is_finished() override {
    return m_current_wave >= m_waves;
  }
};

}  // namespace Ramulator
