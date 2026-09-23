#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class SieveCXLPIMPipelineFrontend final : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd, SieveCXLPIMPipelineFrontend, "SieveCXLPIMPipeline")

 private:
  enum Phase : int {
    QueryLink = 0,
    PIMGWrite = 1,
    PIMMAC = 2,
    PIMRead = 3,
    ResultLink = 4,
    Done = 5,
  };

  int m_pim_channels = 0;
  int m_link_channels = 0;
  int m_pseudo_channels = 0;
  int m_row_span_waves = 0;
  size_t m_query_link_transactions = 0;
  size_t m_result_link_transactions = 0;
  int m_pim_gwrite_waves = 0;
  int m_pim_mac_waves = 0;
  int m_pim_read_waves = 0;
  int m_phase = QueryLink;
  int m_current_wave = 0;
  std::vector<size_t> m_link_targets;
  std::vector<size_t> m_link_sent;
  std::vector<bool> m_pim_sent;

  size_t s_query_link_injected_requests = 0;
  size_t s_query_link_completed_requests = 0;
  size_t s_query_link_completion_cycles = 0;
  size_t s_query_link_rejected_attempts = 0;
  size_t s_pim_gwrite_injected_requests = 0;
  size_t s_pim_gwrite_completed_requests = 0;
  size_t s_pim_gwrite_start_cycles = 0;
  size_t s_pim_gwrite_completion_cycles = 0;
  size_t s_pim_mac_injected_requests = 0;
  size_t s_pim_mac_completed_requests = 0;
  size_t s_pim_mac_start_cycles = 0;
  size_t s_pim_mac_completion_cycles = 0;
  size_t s_pim_read_injected_requests = 0;
  size_t s_pim_read_completed_requests = 0;
  size_t s_pim_read_start_cycles = 0;
  size_t s_pim_read_completion_cycles = 0;
  size_t s_result_link_injected_requests = 0;
  size_t s_result_link_completed_requests = 0;
  size_t s_result_link_start_cycles = 0;
  size_t s_result_link_completion_cycles = 0;
  size_t s_result_link_rejected_attempts = 0;
  size_t s_pipeline_completion_cycles = 0;

 public:
  int get_num_cores() override { return 2; }

  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_pim_channels, int, "pim_channels").required();
    RAMULATOR_PARSE_PARAM(m_link_channels, int, "link_channels").required();
    RAMULATOR_PARSE_PARAM(m_pseudo_channels, int, "pseudo_channels").required();
    RAMULATOR_PARSE_PARAM(m_row_span_waves, int, "row_span_waves").required();
    RAMULATOR_PARSE_PARAM(
        m_query_link_transactions, size_t, "query_link_transactions").required();
    RAMULATOR_PARSE_PARAM(
        m_result_link_transactions, size_t, "result_link_transactions").required();
    RAMULATOR_PARSE_PARAM(m_pim_gwrite_waves, int, "pim_gwrite_waves").required();
    RAMULATOR_PARSE_PARAM(m_pim_mac_waves, int, "pim_mac_waves").required();
    RAMULATOR_PARSE_PARAM(m_pim_read_waves, int, "pim_read_waves").required();

    if (m_pim_channels <= 0 || m_link_channels <= 0 || m_pseudo_channels <= 0 ||
        m_row_span_waves <= 0 || m_query_link_transactions == 0 ||
        m_result_link_transactions == 0 || m_pim_gwrite_waves <= 0 ||
        m_pim_mac_waves <= 0 || m_pim_read_waves <= 0) {
      throw std::runtime_error("SieveCXLPIMPipeline parameters must be positive");
    }

    const size_t link_endpoints =
        static_cast<size_t>(m_link_channels * m_pseudo_channels);
    distribute_link_targets(m_query_link_transactions, link_endpoints);
    m_pim_sent.assign(
        static_cast<size_t>(m_pim_channels * m_pseudo_channels), false);

    m_stats.add("query_link_injected_requests", s_query_link_injected_requests);
    m_stats.add("query_link_completed_requests", s_query_link_completed_requests);
    m_stats.add("query_link_completion_cycles", s_query_link_completion_cycles);
    m_stats.add("query_link_rejected_attempts", s_query_link_rejected_attempts);
    m_stats.add("pim_gwrite_injected_requests", s_pim_gwrite_injected_requests);
    m_stats.add("pim_gwrite_completed_requests", s_pim_gwrite_completed_requests);
    m_stats.add("pim_gwrite_start_cycles", s_pim_gwrite_start_cycles);
    m_stats.add("pim_gwrite_completion_cycles", s_pim_gwrite_completion_cycles);
    m_stats.add("pim_mac_injected_requests", s_pim_mac_injected_requests);
    m_stats.add("pim_mac_completed_requests", s_pim_mac_completed_requests);
    m_stats.add("pim_mac_start_cycles", s_pim_mac_start_cycles);
    m_stats.add("pim_mac_completion_cycles", s_pim_mac_completion_cycles);
    m_stats.add("pim_read_injected_requests", s_pim_read_injected_requests);
    m_stats.add("pim_read_completed_requests", s_pim_read_completed_requests);
    m_stats.add("pim_read_start_cycles", s_pim_read_start_cycles);
    m_stats.add("pim_read_completion_cycles", s_pim_read_completion_cycles);
    m_stats.add("result_link_injected_requests", s_result_link_injected_requests);
    m_stats.add("result_link_completed_requests", s_result_link_completed_requests);
    m_stats.add("result_link_start_cycles", s_result_link_start_cycles);
    m_stats.add("result_link_completion_cycles", s_result_link_completion_cycles);
    m_stats.add("result_link_rejected_attempts", s_result_link_rejected_attempts);
    m_stats.add("pipeline_completion_cycles", s_pipeline_completion_cycles);
  }

  void tick() override {
    m_clk++;
    advance_completed_phase();
    if (m_phase == QueryLink || m_phase == ResultLink) {
      inject_link_requests(m_phase == ResultLink);
    } else if (m_phase == PIMGWrite || m_phase == PIMMAC || m_phase == PIMRead) {
      inject_pim_wave();
    }
  }

  bool is_finished() override { return m_phase == Done; }

 private:
  static void distribute(
      std::vector<size_t>& targets, size_t total, size_t endpoints) {
    targets.assign(endpoints, total / endpoints);
    for (size_t endpoint = 0; endpoint < total % endpoints; endpoint++) {
      targets[endpoint]++;
    }
  }

  void distribute_link_targets(size_t total, size_t endpoints) {
    distribute(m_link_targets, total, endpoints);
    m_link_sent.assign(endpoints, 0);
  }

  size_t pim_endpoint_count() const {
    return static_cast<size_t>(m_pim_channels * m_pseudo_channels);
  }

  size_t expected_pim_requests(int waves) const {
    return static_cast<size_t>(waves) * pim_endpoint_count();
  }

  bool completion_reached(size_t completed, size_t expected, size_t completion_cycle) const {
    return completed >= expected && static_cast<size_t>(m_clk) >= completion_cycle;
  }

  void advance_completed_phase() {
    if (m_phase == QueryLink &&
        completion_reached(
            s_query_link_completed_requests, m_query_link_transactions,
            s_query_link_completion_cycles)) {
      m_phase = PIMGWrite;
      m_current_wave = 0;
      std::fill(m_pim_sent.begin(), m_pim_sent.end(), false);
      s_pim_gwrite_start_cycles = static_cast<size_t>(m_clk);
    } else if (
        m_phase == PIMGWrite &&
        completion_reached(
            s_pim_gwrite_completed_requests,
            expected_pim_requests(m_pim_gwrite_waves),
            s_pim_gwrite_completion_cycles)) {
      m_phase = PIMMAC;
      m_current_wave = 0;
      std::fill(m_pim_sent.begin(), m_pim_sent.end(), false);
      s_pim_mac_start_cycles = static_cast<size_t>(m_clk);
    } else if (
        m_phase == PIMMAC &&
        completion_reached(
            s_pim_mac_completed_requests,
            expected_pim_requests(m_pim_mac_waves), s_pim_mac_completion_cycles)) {
      m_phase = PIMRead;
      m_current_wave = 0;
      std::fill(m_pim_sent.begin(), m_pim_sent.end(), false);
      s_pim_read_start_cycles = static_cast<size_t>(m_clk);
    } else if (
        m_phase == PIMRead &&
        completion_reached(
            s_pim_read_completed_requests,
            expected_pim_requests(m_pim_read_waves), s_pim_read_completion_cycles)) {
      m_phase = ResultLink;
      const size_t endpoints =
          static_cast<size_t>(m_link_channels * m_pseudo_channels);
      distribute_link_targets(m_result_link_transactions, endpoints);
      s_result_link_start_cycles = static_cast<size_t>(m_clk);
    } else if (
        m_phase == ResultLink &&
        completion_reached(
            s_result_link_completed_requests, m_result_link_transactions,
            s_result_link_completion_cycles)) {
      m_phase = Done;
      s_pipeline_completion_cycles = s_result_link_completion_cycles;
    }
  }

  AddrVec_t link_address(
      int channel, int pseudo_channel, size_t local_index) const {
    return AddrVec_t{
        m_pim_channels + channel, pseudo_channel, 0, 0, 0,
        static_cast<int>(local_index / 32), static_cast<int>(local_index % 32) * 8};
  }

  void inject_link_requests(bool result_phase) {
    for (int channel = 0; channel < m_link_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels;
           pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(
            channel * m_pseudo_channels + pseudo_channel);
        if (m_link_sent[endpoint] >= m_link_targets[endpoint]) {
          continue;
        }
        const size_t local_index = m_link_sent[endpoint];
        Request request(
            link_address(channel, pseudo_channel, local_index), Request::Type::Read);
        request.addr = static_cast<Addr_t>(
            local_index * m_link_channels * m_pseudo_channels + endpoint);
        request.source_id = result_phase ? 1 : 0;
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this, result_phase](Request& completed) {
          const size_t depart = static_cast<size_t>(completed.depart);
          if (result_phase) {
            s_result_link_completed_requests++;
            s_result_link_completion_cycles =
                std::max(s_result_link_completion_cycles, depart);
          } else {
            s_query_link_completed_requests++;
            s_query_link_completion_cycles =
                std::max(s_query_link_completion_cycles, depart);
          }
        };
        if (m_memory_system->send(request)) {
          m_link_sent[endpoint]++;
          if (result_phase) {
            s_result_link_injected_requests++;
          } else {
            s_query_link_injected_requests++;
          }
        } else if (result_phase) {
          s_result_link_rejected_attempts++;
        } else {
          s_query_link_rejected_attempts++;
        }
      }
    }
  }

  int phase_waves() const {
    if (m_phase == PIMGWrite) return m_pim_gwrite_waves;
    if (m_phase == PIMMAC) return m_pim_mac_waves;
    return m_pim_read_waves;
  }

  int phase_request_type() const {
    if (m_phase == PIMGWrite) return Request::Type::PIM_GWRITE;
    if (m_phase == PIMMAC) return Request::Type::PIM_MAC;
    return Request::Type::PIM_READ;
  }

  void record_pim_injected() {
    if (m_phase == PIMGWrite) {
      s_pim_gwrite_injected_requests++;
    } else if (m_phase == PIMMAC) {
      s_pim_mac_injected_requests++;
    } else {
      s_pim_read_injected_requests++;
    }
  }

  void record_pim_completed(const Request& completed) {
    const size_t depart = static_cast<size_t>(completed.depart);
    if (completed.type_id == Request::Type::PIM_GWRITE) {
      s_pim_gwrite_completed_requests++;
      s_pim_gwrite_completion_cycles =
          std::max(s_pim_gwrite_completion_cycles, depart);
    } else if (completed.type_id == Request::Type::PIM_MAC) {
      s_pim_mac_completed_requests++;
      s_pim_mac_completion_cycles = std::max(s_pim_mac_completion_cycles, depart);
    } else {
      s_pim_read_completed_requests++;
      s_pim_read_completion_cycles =
          std::max(s_pim_read_completion_cycles, depart);
    }
  }

  void inject_pim_wave() {
    if (m_current_wave >= phase_waves()) {
      return;
    }
    for (int channel = 0; channel < m_pim_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels;
           pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(
            channel * m_pseudo_channels + pseudo_channel);
        if (m_pim_sent[endpoint]) {
          continue;
        }
        const int row = m_current_wave / m_row_span_waves;
        const int column = m_current_wave % m_row_span_waves;
        Request request(
            AddrVec_t{channel, pseudo_channel, 0, 0, 0, row, column},
            phase_request_type());
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback =
            [this](Request& completed) { record_pim_completed(completed); };
        if (m_memory_system->send(request)) {
          m_pim_sent[endpoint] = true;
          record_pim_injected();
        }
      }
    }
    if (std::all_of(
            m_pim_sent.begin(), m_pim_sent.end(),
            [](bool sent) { return sent; })) {
      m_current_wave++;
      std::fill(m_pim_sent.begin(), m_pim_sent.end(), false);
    }
  }
};

}  // namespace Ramulator
