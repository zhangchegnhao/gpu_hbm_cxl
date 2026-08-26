#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class SieveMixedFrontend final : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, SieveMixedFrontend, "SieveMixed")

 private:
  int m_channels = 0;
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
  std::vector<int> m_pim_waves;
  std::vector<int> m_pim_types;
  std::vector<size_t> m_gpu_targets;
  std::vector<size_t> m_gpu_sent;
  std::vector<bool> m_pim_sent;
  int m_pim_stage = 0;
  int m_pim_wave = 0;

  size_t s_gpu_injected_requests = 0;
  size_t s_gpu_completed_requests = 0;
  size_t s_pim_injected_requests = 0;
  size_t s_pim_completed_requests = 0;
  size_t s_pim_gwrite_injected_requests = 0;
  size_t s_pim_mac_injected_requests = 0;
  size_t s_pim_read_injected_requests = 0;
  size_t s_gpu_completion_cycles = 0;
  size_t s_pim_completion_cycles = 0;
  size_t s_pim_gwrite_completion_cycles = 0;
  size_t s_pim_mac_completion_cycles = 0;
  size_t s_pim_read_completion_cycles = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_channels, int, "channels").required();
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
    int gwrite_waves = 0;
    int mac_waves = 0;
    int read_waves = 0;
    RAMULATOR_PARSE_PARAM(gwrite_waves, int, "pim_gwrite_waves").required();
    RAMULATOR_PARSE_PARAM(mac_waves, int, "pim_mac_waves").required();
    RAMULATOR_PARSE_PARAM(read_waves, int, "pim_read_waves").required();

    if (m_channels <= 0 || m_pseudo_channels <= 0 || m_banks_per_pseudo_channel <= 0 ||
        m_banks_per_bank_group <= 0 || m_rows <= 0 || m_gpu_cachelines_per_row <= 0 ||
        m_internal_prefetch_size <= 0 || m_row_span_waves <= 0 || m_tick_ps <= 0 ||
        m_pim_mac_interval_ps <= 0 || m_pim_io_interval_ps <= 0 || gwrite_waves < 0 ||
        mac_waves < 0 || read_waves < 0) {
      throw std::runtime_error("SieveMixed dimensions and wave counts are invalid");
    }
    if (m_gpu_read_transactions == 0 && gwrite_waves == 0 && mac_waves == 0 && read_waves == 0) {
      throw std::runtime_error("SieveMixed requires at least one GPU or PIM request");
    }
    if (m_banks_per_pseudo_channel % m_banks_per_bank_group != 0) {
      throw std::runtime_error("banks_per_pseudo_channel must divide into bank groups");
    }

    const size_t endpoints = static_cast<size_t>(m_channels * m_pseudo_channels);
    m_gpu_targets.assign(endpoints, m_gpu_read_transactions / endpoints);
    for (size_t i = 0; i < m_gpu_read_transactions % endpoints; i++) {
      m_gpu_targets[i]++;
    }
    m_gpu_sent.assign(endpoints, 0);
    m_pim_sent.assign(endpoints, false);
    m_pim_waves = {gwrite_waves, mac_waves, read_waves};
    m_pim_types = {Request::Type::PIM_GWRITE, Request::Type::PIM_MAC, Request::Type::PIM_READ};
    skip_empty_pim_stages();

    m_stats.add("gpu_injected_requests", s_gpu_injected_requests);
    m_stats.add("gpu_completed_requests", s_gpu_completed_requests);
    m_stats.add("pim_injected_requests", s_pim_injected_requests);
    m_stats.add("pim_completed_requests", s_pim_completed_requests);
    m_stats.add("pim_gwrite_injected_requests", s_pim_gwrite_injected_requests);
    m_stats.add("pim_mac_injected_requests", s_pim_mac_injected_requests);
    m_stats.add("pim_read_injected_requests", s_pim_read_injected_requests);
    m_stats.add("gpu_completion_cycles", s_gpu_completion_cycles);
    m_stats.add("pim_completion_cycles", s_pim_completion_cycles);
    m_stats.add("pim_gwrite_completion_cycles", s_pim_gwrite_completion_cycles);
    m_stats.add("pim_mac_completion_cycles", s_pim_mac_completion_cycles);
    m_stats.add("pim_read_completion_cycles", s_pim_read_completion_cycles);
  }

  void tick() override {
    m_clk++;
    inject_gpu_reads();
    inject_pim_wave();
  }

  bool is_finished() override {
    const bool gpu_done = s_gpu_injected_requests >= m_gpu_read_transactions;
    const bool pim_done = m_pim_stage >= static_cast<int>(m_pim_waves.size());
    return gpu_done && pim_done;
  }

 private:
  void skip_empty_pim_stages() {
    while (m_pim_stage < static_cast<int>(m_pim_waves.size()) && m_pim_waves[m_pim_stage] == 0) {
      m_pim_stage++;
      m_pim_wave = 0;
    }
  }

  AddrVec_t gpu_address(int channel, int pseudo_channel, size_t local_index) const {
    const int flat_bank = static_cast<int>(local_index % static_cast<size_t>(m_banks_per_pseudo_channel));
    const size_t per_bank = local_index / static_cast<size_t>(m_banks_per_pseudo_channel);
    const int bank_group = flat_bank / m_banks_per_bank_group;
    const int bank = flat_bank % m_banks_per_bank_group;
    const int row = static_cast<int>(
        (per_bank / static_cast<size_t>(m_gpu_cachelines_per_row)) % static_cast<size_t>(m_rows));
    const int column = static_cast<int>(per_bank % static_cast<size_t>(m_gpu_cachelines_per_row)) *
                       m_internal_prefetch_size;
    return AddrVec_t{channel, pseudo_channel, 0, bank_group, bank, row, column};
  }

  void inject_gpu_reads() {
    for (int channel = 0; channel < m_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels; pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(channel * m_pseudo_channels + pseudo_channel);
        if (m_gpu_sent[endpoint] >= m_gpu_targets[endpoint]) {
          continue;
        }
        const size_t local_index = m_gpu_sent[endpoint];
        Request request(gpu_address(channel, pseudo_channel, local_index), Request::Type::Read);
        request.addr = static_cast<Addr_t>(endpoint * (m_gpu_targets[endpoint] + 1) + local_index);
        request.source_id = 0;
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this](Request& completed) {
          s_gpu_completed_requests++;
          s_gpu_completion_cycles = std::max(s_gpu_completion_cycles, static_cast<size_t>(completed.depart));
        };
        if (m_memory_system->send(request)) {
          m_gpu_sent[endpoint]++;
          s_gpu_injected_requests++;
        }
      }
    }
  }

  void inject_pim_wave() {
    if (m_pim_stage >= static_cast<int>(m_pim_waves.size())) {
      return;
    }
    const Clk_t current_ps = m_clk * m_tick_ps;
    if (current_ps < m_next_pim_wave_inject_ps) {
      return;
    }
    for (int channel = 0; channel < m_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels; pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(channel * m_pseudo_channels + pseudo_channel);
        if (m_pim_sent[endpoint]) {
          continue;
        }
        const int row = m_pim_wave / m_row_span_waves;
        const int column = m_pim_wave % m_row_span_waves;
        Request request(
            AddrVec_t{channel, pseudo_channel, 0, 0, 0, row, column}, m_pim_types[m_pim_stage]);
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this](Request& completed) {
          s_pim_completed_requests++;
          const size_t depart = static_cast<size_t>(completed.depart);
          s_pim_completion_cycles = std::max(s_pim_completion_cycles, depart);
          if (completed.type_id == Request::Type::PIM_GWRITE) {
            s_pim_gwrite_completion_cycles = std::max(s_pim_gwrite_completion_cycles, depart);
          } else if (completed.type_id == Request::Type::PIM_MAC) {
            s_pim_mac_completion_cycles = std::max(s_pim_mac_completion_cycles, depart);
          } else {
            s_pim_read_completion_cycles = std::max(s_pim_read_completion_cycles, depart);
          }
        };
        if (m_memory_system->send(request)) {
          m_pim_sent[endpoint] = true;
          s_pim_injected_requests++;
          if (m_pim_types[m_pim_stage] == Request::Type::PIM_GWRITE) {
            s_pim_gwrite_injected_requests++;
          } else if (m_pim_types[m_pim_stage] == Request::Type::PIM_MAC) {
            s_pim_mac_injected_requests++;
          } else {
            s_pim_read_injected_requests++;
          }
        }
      }
    }
    if (std::all_of(m_pim_sent.begin(), m_pim_sent.end(), [](bool sent) { return sent; })) {
      const int interval_ps = m_pim_types[m_pim_stage] == Request::Type::PIM_MAC
                                  ? m_pim_mac_interval_ps
                                  : m_pim_io_interval_ps;
      m_next_pim_wave_inject_ps =
          std::max(m_next_pim_wave_inject_ps + interval_ps, current_ps + interval_ps);
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
