#include <algorithm>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/controller/impl/hbm_controller_base.h"

namespace Ramulator {

class SieveHBMPIMController final : public HBMControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(
      IController, SieveHBMPIMController, HBMControllerBase, "SieveHBMPIM")

 private:
  ReqBuffer m_pim_buffer;
  int m_pim_buffer_size = 256;
  int m_pim_mac_interval_ps = 24576;
  int m_pim_io_interval_ps = 1024;
  int m_tick_ps = 0;
  int m_level_pseudo_channel = -1;
  int m_num_pseudo_channels = 0;
  std::vector<Clk_t> m_next_issue_ps;
  std::vector<size_t> m_completed_per_pseudo_channel;
  Clk_t m_last_completion_clk = 0;

  std::vector<int> m_milestone_waves;
  std::vector<Clk_t> s_milestone_cycles;
  size_t s_num_pim_gwrite_reqs = 0;
  size_t s_num_pim_mac_reqs = 0;
  size_t s_num_pim_read_reqs = 0;

 public:
  void init() override {
    HBMControllerBase::init();
    RAMULATOR_PARSE_PARAM(m_pim_buffer_size, int, "pim_buffer_size").default_val(256);
    RAMULATOR_PARSE_PARAM(m_pim_mac_interval_ps, int, "pim_mac_interval_ps").required();
    RAMULATOR_PARSE_PARAM(m_pim_io_interval_ps, int, "pim_io_interval_ps").required();
    const std::string milestones =
        ParamReader<std::string>{m_config, "pim_milestone_waves", m_name}.required();

    if (m_pim_buffer_size <= 0 || m_pim_mac_interval_ps <= 0 || m_pim_io_interval_ps <= 0) {
      throw std::runtime_error("SieveHBMPIM timing and buffer parameters must be positive");
    }
    m_pim_buffer.max_size = static_cast<size_t>(m_pim_buffer_size);
    m_level_pseudo_channel = m_device.m_spec->get_level_id("PseudoChannel");
    m_num_pseudo_channels = m_device.m_spec->get_level_size("PseudoChannel");
    m_tick_ps = m_device.m_spec->get_timing_value("tCK_ps");
    m_next_issue_ps.assign(static_cast<size_t>(m_num_pseudo_channels), 0);
    m_completed_per_pseudo_channel.assign(static_cast<size_t>(m_num_pseudo_channels), 0);
    parse_milestones(milestones);
    s_milestone_cycles.assign(m_milestone_waves.size(), 0);
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    HBMControllerBase::setup(frontend, memory_system);
    m_stats.add("num_pim_gwrite_reqs", s_num_pim_gwrite_reqs);
    m_stats.add("num_pim_mac_reqs", s_num_pim_mac_reqs);
    m_stats.add("num_pim_read_reqs", s_num_pim_read_reqs);
    for (size_t i = 0; i < m_milestone_waves.size(); i++) {
      m_stats.add(fmt::format("pim_milestone_{}_cycles", m_milestone_waves[i]), s_milestone_cycles[i]);
    }
  }

  bool send(Request& req) override {
    if (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write) {
      return ControllerBase::send(req);
    }
    if (!is_pim_request(req.type_id)) {
      throw std::runtime_error(fmt::format("unsupported SieveHBMPIM request type {}", req.type_id));
    }
    if (req.addr_vec.size() != static_cast<size_t>(m_device.m_spec->level_count)) {
      throw std::runtime_error("SieveHBMPIM request address vector has the wrong number of levels");
    }
    const int pseudo_channel = req.addr_vec[m_level_pseudo_channel];
    if (pseudo_channel < 0 || pseudo_channel >= m_num_pseudo_channels) {
      throw std::runtime_error("SieveHBMPIM request has an invalid pseudo-channel");
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

    if (m_pim_buffer.size() > 0) {
      issue_pim_requests();
    } else {
      try_issue_slot(SlotType::ColumnBus);
      try_issue_slot(SlotType::RowBus);
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

  void parse_milestones(const std::string& value) {
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
      if (token.empty()) {
        continue;
      }
      const int wave = std::stoi(token);
      if (wave <= 0) {
        throw std::runtime_error("PIM milestone waves must be positive");
      }
      m_milestone_waves.push_back(wave);
    }
    std::sort(m_milestone_waves.begin(), m_milestone_waves.end());
    m_milestone_waves.erase(
        std::unique(m_milestone_waves.begin(), m_milestone_waves.end()), m_milestone_waves.end());
    if (m_milestone_waves.empty()) {
      throw std::runtime_error("at least one PIM milestone wave is required");
    }
  }

  int interval_ps_for(int type_id) const {
    return type_id == Request::Type::PIM_MAC ? m_pim_mac_interval_ps : m_pim_io_interval_ps;
  }

  Clk_t completion_for(int type_id, Clk_t issue_clk) const {
    if (type_id == Request::Type::PIM_READ) {
      return issue_clk + m_device.m_spec->read_latency;
    }
    const Clk_t completion_ps = issue_clk * m_tick_ps + interval_ps_for(type_id);
    return (completion_ps + m_tick_ps - 1) / m_tick_ps;
  }

  void issue_pim_requests() {
    std::vector<bool> issued(static_cast<size_t>(m_num_pseudo_channels), false);
    auto request = m_pim_buffer.begin();
    while (request != m_pim_buffer.end()) {
      const int pseudo_channel = request->addr_vec[m_level_pseudo_channel];
      const size_t pc = static_cast<size_t>(pseudo_channel);
      const Clk_t current_ps = m_clk * m_tick_ps;
      if (issued[pc] || current_ps < m_next_issue_ps[pc]) {
        ++request;
        continue;
      }

      const int type_id = request->type_id;
      const int interval_ps = interval_ps_for(type_id);
      const Clk_t completion = completion_for(type_id, m_clk);
      if (m_completed_per_pseudo_channel[pc] == 0) {
        m_next_issue_ps[pc] = current_ps + interval_ps;
      } else if (current_ps - m_next_issue_ps[pc] >= interval_ps) {
        m_next_issue_ps[pc] = current_ps + interval_ps;
      } else {
        m_next_issue_ps[pc] += interval_ps;
      }
      m_last_completion_clk = std::max(m_last_completion_clk, completion);
      issued[pc] = true;
      m_completed_per_pseudo_channel[pc]++;

      if (type_id == Request::Type::PIM_GWRITE) {
        s_num_pim_gwrite_reqs++;
      } else if (type_id == Request::Type::PIM_MAC) {
        s_num_pim_mac_reqs++;
      } else {
        s_num_pim_read_reqs++;
      }
      if (request->callback) {
        request->depart = completion;
        request->callback(*request);
      }
      update_milestones(completion);

      auto completed = request++;
      m_pim_buffer.remove(completed);
    }
  }

  void update_milestones(Clk_t completion) {
    const size_t completed_waves = *std::min_element(
        m_completed_per_pseudo_channel.begin(), m_completed_per_pseudo_channel.end());
    for (size_t i = 0; i < m_milestone_waves.size(); i++) {
      if (s_milestone_cycles[i] == 0 && completed_waves >= static_cast<size_t>(m_milestone_waves[i])) {
        s_milestone_cycles[i] = completion;
      }
    }
  }
};

}  // namespace Ramulator
