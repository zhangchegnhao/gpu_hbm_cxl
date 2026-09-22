#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class SieveCXLKVFrontend final : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, SieveCXLKVFrontend, "SieveCXLKVRead")

 private:
  int m_local_channels = 0;
  int m_cxl_channels = 0;
  int m_pseudo_channels = 0;
  int m_banks_per_pseudo_channel = 0;
  int m_banks_per_bank_group = 0;
  int m_rows = 0;
  int m_gpu_cachelines_per_row = 0;
  int m_internal_prefetch_size = 0;
  int m_row_span_waves = 0;
  int m_tick_ps = 0;
  int m_pim_mac_interval_ps = 0;
  int m_pim_io_interval_ps = 0;
  Clk_t m_next_pim_wave_inject_ps = 0;
  size_t m_gpu_read_transactions = 0;
  size_t m_local_kv_read_transactions = 0;
  size_t m_cxl_kv_read_transactions = 0;
  std::vector<int> m_pim_waves;
  std::vector<int> m_pim_types;
  std::vector<size_t> m_gpu_targets;
  std::vector<size_t> m_gpu_sent;
  std::vector<size_t> m_local_kv_targets;
  std::vector<size_t> m_local_kv_sent;
  std::vector<size_t> m_cxl_targets;
  std::vector<size_t> m_cxl_sent;
  std::vector<bool> m_prefer_local_kv;
  std::vector<bool> m_pim_sent;
  int m_pim_stage = 0;
  int m_pim_wave = 0;

  size_t s_gpu_injected_requests = 0;
  size_t s_gpu_completed_requests = 0;
  size_t s_local_kv_injected_requests = 0;
  size_t s_local_kv_completed_requests = 0;
  size_t s_cxl_kv_injected_requests = 0;
  size_t s_cxl_kv_completed_requests = 0;
  size_t s_gpu_completion_cycles = 0;
  size_t s_local_kv_completion_cycles = 0;
  size_t s_cxl_kv_completion_cycles = 0;
  size_t s_pim_completion_cycles = 0;
  size_t s_gpu_injection_rejected_attempts = 0;
  size_t s_local_kv_injection_rejected_attempts = 0;
  size_t s_cxl_kv_injection_rejected_attempts = 0;
  size_t s_pim_injected_requests = 0;
  size_t s_pim_completed_requests = 0;

 public:
  int get_num_cores() override { return 3; }

  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_local_channels, int, "local_channels").required();
    RAMULATOR_PARSE_PARAM(m_cxl_channels, int, "cxl_channels").required();
    RAMULATOR_PARSE_PARAM(m_pseudo_channels, int, "pseudo_channels").required();
    RAMULATOR_PARSE_PARAM(m_banks_per_pseudo_channel, int, "banks_per_pseudo_channel").required();
    RAMULATOR_PARSE_PARAM(m_banks_per_bank_group, int, "banks_per_bank_group").required();
    RAMULATOR_PARSE_PARAM(m_rows, int, "rows").required();
    RAMULATOR_PARSE_PARAM(m_gpu_cachelines_per_row, int, "gpu_cachelines_per_row").required();
    RAMULATOR_PARSE_PARAM(m_internal_prefetch_size, int, "internal_prefetch_size").required();
    RAMULATOR_PARSE_PARAM(m_row_span_waves, int, "row_span_waves").required();
    RAMULATOR_PARSE_PARAM(m_tick_ps, int, "tick_ps").required();
    RAMULATOR_PARSE_PARAM(m_pim_mac_interval_ps, int, "pim_mac_interval_ps").required();
    RAMULATOR_PARSE_PARAM(m_pim_io_interval_ps, int, "pim_io_interval_ps").required();
    RAMULATOR_PARSE_PARAM(m_gpu_read_transactions, size_t, "gpu_read_transactions").required();
    RAMULATOR_PARSE_PARAM(m_local_kv_read_transactions, size_t, "local_kv_read_transactions").default_val(0);
    RAMULATOR_PARSE_PARAM(m_cxl_kv_read_transactions, size_t, "cxl_kv_read_transactions").default_val(0);
    int gwrite_waves = 0;
    int mac_waves = 0;
    int read_waves = 0;
    RAMULATOR_PARSE_PARAM(gwrite_waves, int, "pim_gwrite_waves").required();
    RAMULATOR_PARSE_PARAM(mac_waves, int, "pim_mac_waves").required();
    RAMULATOR_PARSE_PARAM(read_waves, int, "pim_read_waves").required();
    if (m_local_channels <= 0 || m_cxl_channels <= 0 || m_pseudo_channels <= 0 ||
        m_banks_per_pseudo_channel <= 0 || m_banks_per_bank_group <= 0 || m_rows < 2 ||
        m_gpu_cachelines_per_row <= 0 || m_internal_prefetch_size <= 0 || m_row_span_waves <= 0 ||
        m_tick_ps <= 0 || m_pim_mac_interval_ps <= 0 || m_pim_io_interval_ps <= 0 ||
        gwrite_waves < 0 || mac_waves < 0 || read_waves < 0) {
      throw std::runtime_error("SieveCXLKV dimensions and wave counts are invalid");
    }
    if (m_gpu_read_transactions == 0 && m_local_kv_read_transactions == 0 &&
        m_cxl_kv_read_transactions == 0 && gwrite_waves == 0 && mac_waves == 0 && read_waves == 0) {
      throw std::runtime_error("SieveCXLKV requires at least one request");
    }
    if (m_banks_per_pseudo_channel % m_banks_per_bank_group != 0) {
      throw std::runtime_error("banks_per_pseudo_channel must divide into bank groups");
    }

    const size_t local_endpoints = static_cast<size_t>(m_local_channels * m_pseudo_channels);
    const size_t cxl_endpoints = static_cast<size_t>(m_cxl_channels * m_pseudo_channels);
    distribute(m_gpu_targets, m_gpu_read_transactions, local_endpoints);
    distribute(m_local_kv_targets, m_local_kv_read_transactions, local_endpoints);
    distribute(m_cxl_targets, m_cxl_kv_read_transactions, cxl_endpoints);
    m_gpu_sent.assign(local_endpoints, 0);
    m_local_kv_sent.assign(local_endpoints, 0);
    m_cxl_sent.assign(cxl_endpoints, 0);
    m_prefer_local_kv.assign(local_endpoints, false);
    m_pim_sent.assign(local_endpoints, false);
    m_pim_waves = {gwrite_waves, mac_waves, read_waves};
    m_pim_types = {Request::Type::PIM_GWRITE, Request::Type::PIM_MAC, Request::Type::PIM_READ};
    skip_empty_pim_stages();

    m_stats.add("gpu_injected_requests", s_gpu_injected_requests);
    m_stats.add("gpu_completed_requests", s_gpu_completed_requests);
    m_stats.add("local_kv_injected_requests", s_local_kv_injected_requests);
    m_stats.add("local_kv_completed_requests", s_local_kv_completed_requests);
    m_stats.add("cxl_kv_injected_requests", s_cxl_kv_injected_requests);
    m_stats.add("cxl_kv_completed_requests", s_cxl_kv_completed_requests);
    m_stats.add("gpu_completion_cycles", s_gpu_completion_cycles);
    m_stats.add("local_kv_completion_cycles", s_local_kv_completion_cycles);
    m_stats.add("cxl_kv_completion_cycles", s_cxl_kv_completion_cycles);
    m_stats.add("pim_completion_cycles", s_pim_completion_cycles);
    m_stats.add("gpu_injection_rejected_attempts", s_gpu_injection_rejected_attempts);
    m_stats.add("local_kv_injection_rejected_attempts", s_local_kv_injection_rejected_attempts);
    m_stats.add("cxl_kv_injection_rejected_attempts", s_cxl_kv_injection_rejected_attempts);
    m_stats.add("pim_injected_requests", s_pim_injected_requests);
    m_stats.add("pim_completed_requests", s_pim_completed_requests);
  }

  void tick() override {
    m_clk++;
    inject_local_reads();
    inject_cxl_reads();
    inject_pim_wave();
  }

  bool is_finished() override {
    return s_gpu_injected_requests >= m_gpu_read_transactions &&
           s_local_kv_injected_requests >= m_local_kv_read_transactions &&
           s_cxl_kv_injected_requests >= m_cxl_kv_read_transactions &&
           m_pim_stage >= static_cast<int>(m_pim_waves.size());
  }

 private:
  static void distribute(std::vector<size_t>& targets, size_t total, size_t endpoints) {
    targets.assign(endpoints, total / endpoints);
    for (size_t i = 0; i < total % endpoints; i++) {
      targets[i]++;
    }
  }

  void skip_empty_pim_stages() {
    while (m_pim_stage < static_cast<int>(m_pim_waves.size()) && m_pim_waves[m_pim_stage] == 0) {
      m_pim_stage++;
      m_pim_wave = 0;
    }
  }

  AddrVec_t read_address(int channel, int pseudo_channel, size_t local_index, bool kv) const {
    const int flat_bank = static_cast<int>(local_index % static_cast<size_t>(m_banks_per_pseudo_channel));
    const size_t per_bank = local_index / static_cast<size_t>(m_banks_per_pseudo_channel);
    const int bank_group = flat_bank / m_banks_per_bank_group;
    const int bank = flat_bank % m_banks_per_bank_group;
    const size_t namespace_rows = static_cast<size_t>(m_rows / 2);
    const int row = static_cast<int>((per_bank / static_cast<size_t>(m_gpu_cachelines_per_row)) % namespace_rows) +
                    (kv ? static_cast<int>(namespace_rows) : 0);
    const int column = static_cast<int>(per_bank % static_cast<size_t>(m_gpu_cachelines_per_row)) *
                       m_internal_prefetch_size;
    return AddrVec_t{channel, pseudo_channel, 0, bank_group, bank, row, column};
  }

  AddrVec_t cxl_address(int channel, int pseudo_channel, size_t local_index) const {
    const int flat_bank = static_cast<int>(local_index % static_cast<size_t>(m_banks_per_pseudo_channel));
    const size_t per_bank = local_index / static_cast<size_t>(m_banks_per_pseudo_channel);
    const int bank_group = flat_bank / m_banks_per_bank_group;
    const int bank = flat_bank % m_banks_per_bank_group;
    const int row = static_cast<int>((per_bank / static_cast<size_t>(m_gpu_cachelines_per_row)) %
                                     static_cast<size_t>(m_rows));
    const int column = static_cast<int>(per_bank % static_cast<size_t>(m_gpu_cachelines_per_row)) *
                       m_internal_prefetch_size;
    return AddrVec_t{m_local_channels + channel, pseudo_channel, 0, bank_group, bank, row, column};
  }

  void inject_local_reads() {
    for (int channel = 0; channel < m_local_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels; pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(channel * m_pseudo_channels + pseudo_channel);
        const bool gpu_pending = m_gpu_sent[endpoint] < m_gpu_targets[endpoint];
        const bool kv_pending = m_local_kv_sent[endpoint] < m_local_kv_targets[endpoint];
        if (!gpu_pending && !kv_pending) continue;
        const bool kv = kv_pending && (!gpu_pending || m_prefer_local_kv[endpoint]);
        const size_t local_index = kv ? m_local_kv_sent[endpoint] : m_gpu_sent[endpoint];
        Request request(read_address(channel, pseudo_channel, local_index, kv), Request::Type::Read);
        request.addr = static_cast<Addr_t>(local_index * m_local_channels * m_pseudo_channels + endpoint +
                                           (kv ? m_gpu_read_transactions : 0));
        request.source_id = kv ? 1 : 0;
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this, kv](Request& completed) {
          const size_t depart = static_cast<size_t>(completed.depart);
          if (kv) {
            s_local_kv_completed_requests++;
            s_local_kv_completion_cycles = std::max(s_local_kv_completion_cycles, depart);
          } else {
            s_gpu_completed_requests++;
            s_gpu_completion_cycles = std::max(s_gpu_completion_cycles, depart);
          }
        };
        if (m_memory_system->send(request)) {
          if (kv) {
            m_local_kv_sent[endpoint]++;
            s_local_kv_injected_requests++;
          } else {
            m_gpu_sent[endpoint]++;
            s_gpu_injected_requests++;
          }
          m_prefer_local_kv[endpoint] = !kv;
        } else if (kv) {
          s_local_kv_injection_rejected_attempts++;
        } else {
          s_gpu_injection_rejected_attempts++;
        }
      }
    }
  }

  void inject_cxl_reads() {
    for (int channel = 0; channel < m_cxl_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels; pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(channel * m_pseudo_channels + pseudo_channel);
        if (m_cxl_sent[endpoint] >= m_cxl_targets[endpoint]) continue;
        const size_t local_index = m_cxl_sent[endpoint];
        Request request(cxl_address(channel, pseudo_channel, local_index), Request::Type::Read);
        request.addr = static_cast<Addr_t>(local_index * m_cxl_channels * m_pseudo_channels + endpoint);
        request.source_id = 2;
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this](Request& completed) {
          s_cxl_kv_completed_requests++;
          s_cxl_kv_completion_cycles = std::max(
              s_cxl_kv_completion_cycles, static_cast<size_t>(completed.depart));
        };
        if (m_memory_system->send(request)) {
          m_cxl_sent[endpoint]++;
          s_cxl_kv_injected_requests++;
        } else {
          s_cxl_kv_injection_rejected_attempts++;
        }
      }
    }
  }

  void inject_pim_wave() {
    if (m_pim_stage >= static_cast<int>(m_pim_waves.size())) return;
    const Clk_t current_ps = m_clk * m_tick_ps;
    if (current_ps < m_next_pim_wave_inject_ps) return;
    for (int channel = 0; channel < m_local_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels; pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(channel * m_pseudo_channels + pseudo_channel);
        if (m_pim_sent[endpoint]) continue;
        const int row = m_pim_wave / m_row_span_waves;
        const int column = m_pim_wave % m_row_span_waves;
        Request request(AddrVec_t{channel, pseudo_channel, 0, 0, 0, row, column},
                       m_pim_types[m_pim_stage]);
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this](Request& completed) {
          s_pim_completed_requests++;
          s_pim_completion_cycles = std::max(
              s_pim_completion_cycles, static_cast<size_t>(completed.depart));
        };
        if (m_memory_system->send(request)) {
          m_pim_sent[endpoint] = true;
          s_pim_injected_requests++;
        }
      }
    }
    if (std::all_of(m_pim_sent.begin(), m_pim_sent.end(), [](bool sent) { return sent; })) {
      const int interval_ps = m_pim_types[m_pim_stage] == Request::Type::PIM_MAC
                                  ? m_pim_mac_interval_ps : m_pim_io_interval_ps;
      m_next_pim_wave_inject_ps = std::max(
          m_next_pim_wave_inject_ps + interval_ps, current_ps + interval_ps);
      m_pim_wave++;
      std::fill(m_pim_sent.begin(), m_pim_sent.end(), false);
      if (m_pim_wave >= m_pim_waves[m_pim_stage]) {
        m_pim_stage++;
        m_pim_wave = 0;
        skip_empty_pim_stages();
      }
    }
  }
};

}  // namespace Ramulator
