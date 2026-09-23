#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class SieveCXLPIMChunkPipelineFrontend final : public IFrontEnd,
                                                public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd, SieveCXLPIMChunkPipelineFrontend, "SieveCXLPIMChunkPipeline")

 private:
  enum SetupPhase : int { QueryLink = 0, PIMGWrite = 1, Chunks = 2, Done = 3 };
  enum ChunkState : int {
    Pending = 0,
    MACActive = 1,
    ReadActive = 2,
    ResultReady = 3,
    ResultActive = 4,
    Complete = 5,
  };

  int m_pim_channels = 0;
  int m_link_channels = 0;
  int m_pseudo_channels = 0;
  int m_row_span_waves = 0;
  int m_chunk_count = 0;
  int m_buffer_slots = 0;
  size_t m_query_link_transactions = 0;
  size_t m_result_link_transactions_per_chunk = 0;
  int m_pim_gwrite_waves = 0;
  int m_pim_mac_waves = 0;
  int m_pim_read_waves_per_chunk = 0;

  int m_setup_phase = QueryLink;
  int m_pim_chunk = -1;
  int m_link_chunk = -1;
  int m_next_pim_chunk = 0;
  int m_current_wave = 0;
  size_t m_active_buffers = 0;
  std::vector<int> m_chunk_state;
  std::vector<int> m_chunk_mac_waves;
  std::vector<int> m_chunk_mac_wave_offsets;
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
  size_t s_pim_read_injected_requests = 0;
  size_t s_pim_read_completed_requests = 0;
  size_t s_result_link_injected_requests = 0;
  size_t s_result_link_completed_requests = 0;
  size_t s_result_link_rejected_attempts = 0;
  size_t s_pipeline_completion_cycles = 0;
  size_t s_max_active_buffers = 0;
  size_t s_buffer_stall_cycles = 0;
  std::vector<size_t> s_chunk_mac_injected_requests;
  std::vector<size_t> s_chunk_mac_completed_requests;
  std::vector<size_t> s_chunk_mac_start_cycles;
  std::vector<size_t> s_chunk_mac_completion_cycles;
  std::vector<size_t> s_chunk_read_injected_requests;
  std::vector<size_t> s_chunk_read_completed_requests;
  std::vector<size_t> s_chunk_read_start_cycles;
  std::vector<size_t> s_chunk_read_completion_cycles;
  std::vector<size_t> s_chunk_result_link_injected_requests;
  std::vector<size_t> s_chunk_result_link_completed_requests;
  std::vector<size_t> s_chunk_result_link_start_cycles;
  std::vector<size_t> s_chunk_result_link_completion_cycles;

 public:
  int get_num_cores() override { return 2; }

  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_pim_channels, int, "pim_channels").required();
    RAMULATOR_PARSE_PARAM(m_link_channels, int, "link_channels").required();
    RAMULATOR_PARSE_PARAM(m_pseudo_channels, int, "pseudo_channels").required();
    RAMULATOR_PARSE_PARAM(m_row_span_waves, int, "row_span_waves").required();
    RAMULATOR_PARSE_PARAM(m_chunk_count, int, "chunk_count").required();
    RAMULATOR_PARSE_PARAM(m_buffer_slots, int, "buffer_slots").required();
    RAMULATOR_PARSE_PARAM(
        m_query_link_transactions, size_t, "query_link_transactions").required();
    RAMULATOR_PARSE_PARAM(
        m_result_link_transactions_per_chunk, size_t,
        "result_link_transactions_per_chunk").required();
    RAMULATOR_PARSE_PARAM(m_pim_gwrite_waves, int, "pim_gwrite_waves").required();
    RAMULATOR_PARSE_PARAM(m_pim_mac_waves, int, "pim_mac_waves").required();
    RAMULATOR_PARSE_PARAM(
        m_pim_read_waves_per_chunk, int,
        "pim_read_waves_per_chunk").required();

    if (m_pim_channels <= 0 || m_link_channels <= 0 ||
        m_pseudo_channels <= 0 || m_row_span_waves <= 0 ||
        m_chunk_count <= 0 || m_buffer_slots <= 0 ||
        m_buffer_slots > m_chunk_count || m_query_link_transactions == 0 ||
        m_result_link_transactions_per_chunk == 0 ||
        m_pim_gwrite_waves <= 0 || m_pim_mac_waves < m_chunk_count ||
        m_pim_read_waves_per_chunk <= 0) {
      throw std::runtime_error(
          "SieveCXLPIMChunkPipeline parameters are inconsistent");
    }

    const size_t link_endpoints =
        static_cast<size_t>(m_link_channels * m_pseudo_channels);
    distribute_link_targets(m_query_link_transactions, link_endpoints);
    m_pim_sent.assign(pim_endpoint_count(), false);
    m_chunk_state.assign(static_cast<size_t>(m_chunk_count), Pending);
    m_chunk_mac_waves.assign(
        static_cast<size_t>(m_chunk_count), m_pim_mac_waves / m_chunk_count);
    for (int chunk = 0; chunk < m_pim_mac_waves % m_chunk_count; chunk++) {
      m_chunk_mac_waves[static_cast<size_t>(chunk)]++;
    }
    m_chunk_mac_wave_offsets.assign(static_cast<size_t>(m_chunk_count), 0);
    int offset = 0;
    for (int chunk = 0; chunk < m_chunk_count; chunk++) {
      m_chunk_mac_wave_offsets[static_cast<size_t>(chunk)] = offset;
      offset += m_chunk_mac_waves[static_cast<size_t>(chunk)];
    }

    resize_chunk_stats();
    register_stats();
  }

  void tick() override {
    m_clk++;
    advance_setup();
    if (m_setup_phase == Chunks) {
      advance_pim_chunk();
      advance_result_chunk();
      launch_result_chunk();
      launch_pim_chunk();
      if (m_pim_chunk < 0 && m_next_pim_chunk < m_chunk_count &&
          m_active_buffers >= static_cast<size_t>(m_buffer_slots)) {
        s_buffer_stall_cycles++;
      }
    }

    if (m_setup_phase == QueryLink) {
      inject_link_requests(true);
    } else if (m_setup_phase == PIMGWrite) {
      inject_pim_wave();
    } else if (m_setup_phase == Chunks) {
      if (m_pim_chunk >= 0) {
        inject_pim_wave();
      }
      if (m_link_chunk >= 0) {
        inject_link_requests(false);
      }
    }
  }

  bool is_finished() override { return m_setup_phase == Done; }

 private:
  void resize_chunk_stats() {
    const size_t chunks = static_cast<size_t>(m_chunk_count);
    s_chunk_mac_injected_requests.assign(chunks, 0);
    s_chunk_mac_completed_requests.assign(chunks, 0);
    s_chunk_mac_start_cycles.assign(chunks, 0);
    s_chunk_mac_completion_cycles.assign(chunks, 0);
    s_chunk_read_injected_requests.assign(chunks, 0);
    s_chunk_read_completed_requests.assign(chunks, 0);
    s_chunk_read_start_cycles.assign(chunks, 0);
    s_chunk_read_completion_cycles.assign(chunks, 0);
    s_chunk_result_link_injected_requests.assign(chunks, 0);
    s_chunk_result_link_completed_requests.assign(chunks, 0);
    s_chunk_result_link_start_cycles.assign(chunks, 0);
    s_chunk_result_link_completion_cycles.assign(chunks, 0);
  }

  void register_stats() {
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
    m_stats.add("pim_read_injected_requests", s_pim_read_injected_requests);
    m_stats.add("pim_read_completed_requests", s_pim_read_completed_requests);
    m_stats.add("result_link_injected_requests", s_result_link_injected_requests);
    m_stats.add("result_link_completed_requests", s_result_link_completed_requests);
    m_stats.add("result_link_rejected_attempts", s_result_link_rejected_attempts);
    m_stats.add("pipeline_completion_cycles", s_pipeline_completion_cycles);
    m_stats.add("max_active_buffers", s_max_active_buffers);
    m_stats.add("buffer_stall_cycles", s_buffer_stall_cycles);
    m_stats.add("chunk_mac_waves", m_chunk_mac_waves);
    m_stats.add("chunk_mac_injected_requests", s_chunk_mac_injected_requests);
    m_stats.add("chunk_mac_completed_requests", s_chunk_mac_completed_requests);
    m_stats.add("chunk_mac_start_cycles", s_chunk_mac_start_cycles);
    m_stats.add("chunk_mac_completion_cycles", s_chunk_mac_completion_cycles);
    m_stats.add("chunk_read_injected_requests", s_chunk_read_injected_requests);
    m_stats.add("chunk_read_completed_requests", s_chunk_read_completed_requests);
    m_stats.add("chunk_read_start_cycles", s_chunk_read_start_cycles);
    m_stats.add("chunk_read_completion_cycles", s_chunk_read_completion_cycles);
    m_stats.add(
        "chunk_result_link_injected_requests",
        s_chunk_result_link_injected_requests);
    m_stats.add(
        "chunk_result_link_completed_requests",
        s_chunk_result_link_completed_requests);
    m_stats.add(
        "chunk_result_link_start_cycles", s_chunk_result_link_start_cycles);
    m_stats.add(
        "chunk_result_link_completion_cycles",
        s_chunk_result_link_completion_cycles);
  }

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

  bool completion_reached(
      size_t completed, size_t expected, size_t completion_cycle) const {
    return completed >= expected && static_cast<size_t>(m_clk) >= completion_cycle;
  }

  void reset_pim_injection() {
    m_current_wave = 0;
    std::fill(m_pim_sent.begin(), m_pim_sent.end(), false);
  }

  void advance_setup() {
    if (m_setup_phase == QueryLink &&
        completion_reached(
            s_query_link_completed_requests, m_query_link_transactions,
            s_query_link_completion_cycles)) {
      m_setup_phase = PIMGWrite;
      reset_pim_injection();
      s_pim_gwrite_start_cycles = static_cast<size_t>(m_clk);
    } else if (
        m_setup_phase == PIMGWrite &&
        completion_reached(
            s_pim_gwrite_completed_requests,
            expected_pim_requests(m_pim_gwrite_waves),
            s_pim_gwrite_completion_cycles)) {
      m_setup_phase = Chunks;
      reset_pim_injection();
    }
  }

  void advance_pim_chunk() {
    if (m_pim_chunk < 0) {
      return;
    }
    const size_t chunk = static_cast<size_t>(m_pim_chunk);
    if (m_chunk_state[chunk] == MACActive &&
        completion_reached(
            s_chunk_mac_completed_requests[chunk],
            expected_pim_requests(m_chunk_mac_waves[chunk]),
            s_chunk_mac_completion_cycles[chunk])) {
      m_chunk_state[chunk] = ReadActive;
      reset_pim_injection();
      s_chunk_read_start_cycles[chunk] = static_cast<size_t>(m_clk);
    } else if (
        m_chunk_state[chunk] == ReadActive &&
        completion_reached(
            s_chunk_read_completed_requests[chunk],
            expected_pim_requests(m_pim_read_waves_per_chunk),
            s_chunk_read_completion_cycles[chunk])) {
      m_chunk_state[chunk] = ResultReady;
      m_pim_chunk = -1;
      reset_pim_injection();
    }
  }

  void advance_result_chunk() {
    if (m_link_chunk < 0) {
      return;
    }
    const size_t chunk = static_cast<size_t>(m_link_chunk);
    if (completion_reached(
            s_chunk_result_link_completed_requests[chunk],
            m_result_link_transactions_per_chunk,
            s_chunk_result_link_completion_cycles[chunk])) {
      m_chunk_state[chunk] = Complete;
      m_link_chunk = -1;
      m_active_buffers--;
      s_pipeline_completion_cycles =
          s_chunk_result_link_completion_cycles[chunk];
      if (std::all_of(
              m_chunk_state.begin(), m_chunk_state.end(),
              [](int state) { return state == Complete; })) {
        m_setup_phase = Done;
      }
    }
  }

  void launch_pim_chunk() {
    if (m_pim_chunk >= 0 || m_next_pim_chunk >= m_chunk_count ||
        m_active_buffers >= static_cast<size_t>(m_buffer_slots)) {
      return;
    }
    m_pim_chunk = m_next_pim_chunk++;
    const size_t chunk = static_cast<size_t>(m_pim_chunk);
    m_chunk_state[chunk] = MACActive;
    m_active_buffers++;
    s_max_active_buffers = std::max(s_max_active_buffers, m_active_buffers);
    reset_pim_injection();
    s_chunk_mac_start_cycles[chunk] = static_cast<size_t>(m_clk);
  }

  void launch_result_chunk() {
    if (m_link_chunk >= 0) {
      return;
    }
    for (int candidate = 0; candidate < m_chunk_count; candidate++) {
      const size_t chunk = static_cast<size_t>(candidate);
      if (m_chunk_state[chunk] != ResultReady) {
        continue;
      }
      m_link_chunk = candidate;
      m_chunk_state[chunk] = ResultActive;
      distribute_link_targets(
          m_result_link_transactions_per_chunk,
          static_cast<size_t>(m_link_channels * m_pseudo_channels));
      s_chunk_result_link_start_cycles[chunk] = static_cast<size_t>(m_clk);
      return;
    }
  }

  AddrVec_t link_address(
      int channel, int pseudo_channel, size_t local_index) const {
    return AddrVec_t{
        m_pim_channels + channel, pseudo_channel, 0, 0, 0,
        static_cast<int>(local_index / 32),
        static_cast<int>(local_index % 32) * 8};
  }

  void inject_link_requests(bool query_phase) {
    for (int channel = 0; channel < m_link_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels;
           pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(
            channel * m_pseudo_channels + pseudo_channel);
        if (m_link_sent[endpoint] >= m_link_targets[endpoint]) {
          continue;
        }
        const size_t local_index = m_link_sent[endpoint];
        const size_t chunk_offset =
            query_phase
                ? 0
                : m_query_link_transactions +
                      static_cast<size_t>(m_link_chunk) *
                          m_result_link_transactions_per_chunk;
        Request request(
            link_address(channel, pseudo_channel, chunk_offset + local_index),
            Request::Type::Read);
        request.addr = static_cast<Addr_t>(
            (chunk_offset + local_index) * m_link_channels *
                m_pseudo_channels +
            endpoint);
        request.source_id = query_phase ? 0 : 1;
        request.size_bytes = m_memory_system->get_tx_bytes();
        const int chunk_id = m_link_chunk;
        request.callback = [this, query_phase, chunk_id](Request& completed) {
          const size_t depart = static_cast<size_t>(completed.depart);
          if (query_phase) {
            s_query_link_completed_requests++;
            s_query_link_completion_cycles =
                std::max(s_query_link_completion_cycles, depart);
          } else {
            const size_t chunk = static_cast<size_t>(chunk_id);
            s_result_link_completed_requests++;
            s_chunk_result_link_completed_requests[chunk]++;
            s_chunk_result_link_completion_cycles[chunk] = std::max(
                s_chunk_result_link_completion_cycles[chunk], depart);
          }
        };
        if (m_memory_system->send(request)) {
          m_link_sent[endpoint]++;
          if (query_phase) {
            s_query_link_injected_requests++;
          } else {
            const size_t chunk = static_cast<size_t>(m_link_chunk);
            s_result_link_injected_requests++;
            s_chunk_result_link_injected_requests[chunk]++;
          }
        } else if (query_phase) {
          s_query_link_rejected_attempts++;
        } else {
          s_result_link_rejected_attempts++;
        }
      }
    }
  }

  int active_pim_waves() const {
    if (m_setup_phase == PIMGWrite) {
      return m_pim_gwrite_waves;
    }
    const size_t chunk = static_cast<size_t>(m_pim_chunk);
    return m_chunk_state[chunk] == MACActive
               ? m_chunk_mac_waves[chunk]
               : m_pim_read_waves_per_chunk;
  }

  int active_pim_type() const {
    if (m_setup_phase == PIMGWrite) {
      return Request::Type::PIM_GWRITE;
    }
    return m_chunk_state[static_cast<size_t>(m_pim_chunk)] == MACActive
               ? Request::Type::PIM_MAC
               : Request::Type::PIM_READ;
  }

  int active_global_wave() const {
    if (m_setup_phase == PIMGWrite) {
      return m_current_wave;
    }
    const size_t chunk = static_cast<size_t>(m_pim_chunk);
    if (m_chunk_state[chunk] == MACActive) {
      return m_pim_gwrite_waves + m_chunk_mac_wave_offsets[chunk] +
             m_current_wave;
    }
    return m_pim_gwrite_waves + m_pim_mac_waves +
           static_cast<int>(chunk) * m_pim_read_waves_per_chunk +
           m_current_wave;
  }

  void record_pim_injected(int request_type, int chunk_id) {
    if (request_type == Request::Type::PIM_GWRITE) {
      s_pim_gwrite_injected_requests++;
    } else if (request_type == Request::Type::PIM_MAC) {
      s_pim_mac_injected_requests++;
      s_chunk_mac_injected_requests[static_cast<size_t>(chunk_id)]++;
    } else {
      s_pim_read_injected_requests++;
      s_chunk_read_injected_requests[static_cast<size_t>(chunk_id)]++;
    }
  }

  void record_pim_completed(
      const Request& completed, int request_type, int chunk_id) {
    const size_t depart = static_cast<size_t>(completed.depart);
    if (request_type == Request::Type::PIM_GWRITE) {
      s_pim_gwrite_completed_requests++;
      s_pim_gwrite_completion_cycles =
          std::max(s_pim_gwrite_completion_cycles, depart);
    } else if (request_type == Request::Type::PIM_MAC) {
      const size_t chunk = static_cast<size_t>(chunk_id);
      s_pim_mac_completed_requests++;
      s_chunk_mac_completed_requests[chunk]++;
      s_chunk_mac_completion_cycles[chunk] =
          std::max(s_chunk_mac_completion_cycles[chunk], depart);
    } else {
      const size_t chunk = static_cast<size_t>(chunk_id);
      s_pim_read_completed_requests++;
      s_chunk_read_completed_requests[chunk]++;
      s_chunk_read_completion_cycles[chunk] =
          std::max(s_chunk_read_completion_cycles[chunk], depart);
    }
  }

  void inject_pim_wave() {
    if (m_current_wave >= active_pim_waves()) {
      return;
    }
    const int request_type = active_pim_type();
    const int chunk_id = m_setup_phase == PIMGWrite ? -1 : m_pim_chunk;
    const int global_wave = active_global_wave();
    for (int channel = 0; channel < m_pim_channels; channel++) {
      for (int pseudo_channel = 0; pseudo_channel < m_pseudo_channels;
           pseudo_channel++) {
        const size_t endpoint = static_cast<size_t>(
            channel * m_pseudo_channels + pseudo_channel);
        if (m_pim_sent[endpoint]) {
          continue;
        }
        const int row = global_wave / m_row_span_waves;
        const int column = global_wave % m_row_span_waves;
        Request request(
            AddrVec_t{channel, pseudo_channel, 0, 0, 0, row, column},
            request_type);
        request.size_bytes = m_memory_system->get_tx_bytes();
        request.callback = [this, request_type, chunk_id](Request& completed) {
          record_pim_completed(completed, request_type, chunk_id);
        };
        if (m_memory_system->send(request)) {
          m_pim_sent[endpoint] = true;
          record_pim_injected(request_type, chunk_id);
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
