#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/controller/impl/hbm_controller_base.h"

namespace Ramulator {

class SieveHBMPIMControllerV1 final : public HBMControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(
      IController, SieveHBMPIMControllerV1, HBMControllerBase, "SieveHBMPIMV1")

 private:
  ReqBuffer m_pim_buffer;
  int m_pim_buffer_size = 256;
  int m_pim_mac_interval_ps = 24576;
  int m_pim_io_interval_ps = 1024;
  int m_tick_ps = 0;
  int m_level_pseudo_channel = -1;
  int m_level_row = -1;
  int m_num_pseudo_channels = 0;
  bool m_dual_row_buffer = true;
  bool m_prefer_pim = false;
  std::vector<Clk_t> m_next_issue_ps;
  std::vector<int> m_pim_open_row;
  std::vector<Clk_t> m_pim_row_ready_clk;
  Clk_t m_last_completion_clk = 0;

  size_t s_num_pim_gwrite_reqs = 0;
  size_t s_num_pim_mac_reqs = 0;
  size_t s_num_pim_read_reqs = 0;
  size_t s_pim_row_activations = 0;
  size_t s_pim_row_conflicts = 0;
  size_t s_pim_queue_wait_cycles = 0;
  size_t s_gpu_blocked_by_pim_cycles = 0;
  size_t s_pim_blocked_by_gpu_cycles = 0;
  size_t s_gpu_column_issues = 0;
  size_t s_pim_column_issues = 0;

 public:
  void init() override {
    HBMControllerBase::init();
    RAMULATOR_PARSE_PARAM(m_pim_buffer_size, int, "pim_buffer_size").default_val(256);
    RAMULATOR_PARSE_PARAM(m_pim_mac_interval_ps, int, "pim_mac_interval_ps").required();
    RAMULATOR_PARSE_PARAM(m_pim_io_interval_ps, int, "pim_io_interval_ps").required();
    RAMULATOR_PARSE_PARAM(m_dual_row_buffer, bool, "dual_row_buffer").default_val(true);
    if (m_pim_buffer_size <= 0 || m_pim_mac_interval_ps <= 0 || m_pim_io_interval_ps <= 0) {
      throw std::runtime_error("SieveHBMPIMV1 timing and buffer parameters must be positive");
    }
    m_pim_buffer.max_size = static_cast<size_t>(m_pim_buffer_size);
    m_level_pseudo_channel = m_device.m_spec->get_level_id("PseudoChannel");
    m_level_row = m_device.m_spec->get_level_id("Row");
    m_num_pseudo_channels = m_device.m_spec->get_level_size("PseudoChannel");
    m_tick_ps = m_device.m_spec->get_timing_value("tCK_ps");
    m_next_issue_ps.assign(static_cast<size_t>(m_num_pseudo_channels), 0);
    m_pim_open_row.assign(static_cast<size_t>(m_num_pseudo_channels), -1);
    m_pim_row_ready_clk.assign(static_cast<size_t>(m_num_pseudo_channels), 0);
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    HBMControllerBase::setup(frontend, memory_system);
    m_stats.add("num_pim_gwrite_reqs", s_num_pim_gwrite_reqs);
    m_stats.add("num_pim_mac_reqs", s_num_pim_mac_reqs);
    m_stats.add("num_pim_read_reqs", s_num_pim_read_reqs);
    m_stats.add("pim_row_activations", s_pim_row_activations);
    m_stats.add("pim_row_conflicts", s_pim_row_conflicts);
    m_stats.add("pim_queue_wait_cycles", s_pim_queue_wait_cycles);
    m_stats.add("gpu_blocked_by_pim_cycles", s_gpu_blocked_by_pim_cycles);
    m_stats.add("pim_blocked_by_gpu_cycles", s_pim_blocked_by_gpu_cycles);
    m_stats.add("gpu_column_issues", s_gpu_column_issues);
    m_stats.add("pim_column_issues", s_pim_column_issues);
  }

  bool send(Request& req) override {
    if (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write) {
      return ControllerBase::send(req);
    }
    if (!is_pim_request(req.type_id)) {
      throw std::runtime_error(fmt::format("unsupported SieveHBMPIMV1 request type {}", req.type_id));
    }
    if (req.addr_vec.size() != static_cast<size_t>(m_device.m_spec->level_count)) {
      throw std::runtime_error("SieveHBMPIMV1 request address vector has the wrong number of levels");
    }
    const int pseudo_channel = req.addr_vec[m_level_pseudo_channel];
    if (pseudo_channel < 0 || pseudo_channel >= m_num_pseudo_channels) {
      throw std::runtime_error("SieveHBMPIMV1 request has an invalid pseudo-channel");
    }
    req.arrive = m_clk;
    if (!m_pim_buffer.enqueue(req)) {
      req.arrive = -1;
      return false;
    }
    return true;
  }

  void tick() override {
    hbm_tick_prologue();

    const bool transition_blocks_normal = !m_dual_row_buffer && single_buffer_transition_active();
    if (!transition_blocks_normal) {
      try_issue_slot(SlotType::RowBus);
    }

    auto pim = find_ready_pim_request();
    const bool gpu_queued = m_active_buffer.size() > 0 || m_read_buffer.size() > 0 ||
                            m_write_buffer.size() > 0 || m_priority_buffer.size() > 0;
    bool issued_pim = false;
    bool issued_gpu = false;

    if (pim != m_pim_buffer.end() && (!gpu_queued || m_prefer_pim || transition_blocks_normal)) {
      issue_pim_request(pim);
      issued_pim = true;
    } else if (!transition_blocks_normal) {
      issued_gpu = try_issue_slot(SlotType::ColumnBus).has_value();
      if (!issued_gpu && pim != m_pim_buffer.end()) {
        issue_pim_request(pim);
        issued_pim = true;
      }
    }

    if (issued_pim) {
      s_pim_column_issues++;
      if (gpu_queued) {
        s_gpu_blocked_by_pim_cycles++;
      }
      m_prefer_pim = false;
    } else if (issued_gpu) {
      s_gpu_column_issues++;
      if (pim != m_pim_buffer.end()) {
        s_pim_blocked_by_gpu_cycles++;
      }
      m_prefer_pim = true;
    }

    hbm_tick_epilogue();
  }

  bool is_pending() const override {
    return ControllerBase::is_pending() || m_pim_buffer.size() > 0 || m_clk < m_last_completion_clk;
  }

 private:
  static bool is_pim_request(int type_id) {
    return type_id == Request::Type::PIM_GWRITE || type_id == Request::Type::PIM_MAC ||
           type_id == Request::Type::PIM_READ;
  }

  int interval_ps_for(int type_id) const {
    return type_id == Request::Type::PIM_MAC ? m_pim_mac_interval_ps : m_pim_io_interval_ps;
  }

  bool single_buffer_transition_active() const {
    return std::any_of(
        m_pim_row_ready_clk.begin(), m_pim_row_ready_clk.end(),
        [this](Clk_t ready) { return ready > m_clk; });
  }

  void close_normal_rows_for_pseudo_channel(int pseudo_channel) {
    AddrVec_t scope(static_cast<size_t>(m_device.m_spec->level_count), -1);
    scope[0] = m_channel_id;
    scope[m_level_pseudo_channel] = pseudo_channel;
    const int closed = m_device.m_spec->get_state_id("Closed");
    for (auto* bank : m_device.m_bank_nodes) {
      if (m_device.bank_matches(bank, scope)) {
        bank->m_state = closed;
        bank->m_row_state.clear();
      }
    }
  }

  bool prepare_pim_row(const Request& request) {
    const int pseudo_channel = request.addr_vec[m_level_pseudo_channel];
    const size_t pc = static_cast<size_t>(pseudo_channel);
    const int requested_row = request.addr_vec[m_level_row];
    if (m_pim_open_row[pc] == requested_row) {
      return m_clk >= m_pim_row_ready_clk[pc];
    }
    if (!m_dual_row_buffer && m_active_buffer.size() > 0) {
      return false;
    }

    const bool conflict = m_pim_open_row[pc] >= 0;
    const int n_rcd = m_device.m_spec->get_timing_value("nRCDRD");
    const int n_rp = conflict ? m_device.m_spec->get_timing_value("nRP") : 0;
    if (!m_dual_row_buffer) {
      close_normal_rows_for_pseudo_channel(pseudo_channel);
    }
    m_pim_open_row[pc] = requested_row;
    m_pim_row_ready_clk[pc] = m_clk + n_rp + n_rcd;
    s_pim_row_activations++;
    if (conflict) {
      s_pim_row_conflicts++;
    }
    return false;
  }

  ReqBuffer::iterator find_ready_pim_request() {
    const Clk_t current_ps = m_clk * m_tick_ps;
    const bool endpoint_ready = std::any_of(
        m_next_issue_ps.begin(), m_next_issue_ps.end(),
        [current_ps](Clk_t next_issue) { return current_ps >= next_issue; });
    if (!endpoint_ready) {
      return m_pim_buffer.end();
    }
    std::vector<bool> considered(static_cast<size_t>(m_num_pseudo_channels), false);
    auto request = m_pim_buffer.begin();
    while (request != m_pim_buffer.end()) {
      const int pseudo_channel = request->addr_vec[m_level_pseudo_channel];
      const size_t pc = static_cast<size_t>(pseudo_channel);
      if (considered[pc]) {
        ++request;
        continue;
      }
      considered[pc] = true;
      if (current_ps >= m_next_issue_ps[pc] && prepare_pim_row(*request)) {
        return request;
      }
      ++request;
    }
    return m_pim_buffer.end();
  }

  void issue_pim_request(ReqBuffer::iterator request) {
    const int pseudo_channel = request->addr_vec[m_level_pseudo_channel];
    const size_t pc = static_cast<size_t>(pseudo_channel);
    const int type_id = request->type_id;
    const int interval_ps = interval_ps_for(type_id);
    const Clk_t current_ps = m_clk * m_tick_ps;
    const Clk_t completion = type_id == Request::Type::PIM_READ
                                 ? m_clk + m_device.m_spec->read_latency
                                 : (current_ps + interval_ps + m_tick_ps - 1) / m_tick_ps;
    m_next_issue_ps[pc] = std::max(m_next_issue_ps[pc] + interval_ps, current_ps + interval_ps);
    m_last_completion_clk = std::max(m_last_completion_clk, completion);
    s_pim_queue_wait_cycles += static_cast<size_t>(m_clk - request->arrive);

    if (type_id == Request::Type::PIM_GWRITE) {
      s_num_pim_gwrite_reqs++;
    } else if (type_id == Request::Type::PIM_MAC) {
      s_num_pim_mac_reqs++;
    } else {
      s_num_pim_read_reqs++;
    }
    request->depart = completion;
    if (request->callback) {
      request->callback(*request);
    }
    m_pim_buffer.remove(request);
  }
};

}  // namespace Ramulator
