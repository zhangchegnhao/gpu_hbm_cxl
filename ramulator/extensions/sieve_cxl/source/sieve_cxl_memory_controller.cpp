#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "ramulator/base/param.h"
#include "ramulator/controller/controller_base.h"

namespace Ramulator {

// Memory-only CXL service model. DRAMSpec supplies the transaction width and
// address levels; the CXL link itself is modeled as a FIFO with explicit
// bandwidth and access-latency parameters.
class SieveCXLMemoryController final : public ControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(
      IController, SieveCXLMemoryController, ControllerBase, "SieveCXLMemory")

 private:
  double m_bandwidth_bytes_per_second = 0.0;
  int64_t m_access_latency_ps = 0;
  int m_tick_ps = 0;
  int64_t m_issue_interval_ps = 0;
  Clk_t m_next_issue_ps = 0;

  size_t s_cxl_requests_served = 0;
  size_t s_cxl_queue_wait_cycles = 0;
  size_t s_cxl_max_queue_wait_cycles = 0;
  size_t s_cxl_request_residence_cycles = 0;
  size_t s_cxl_max_request_residence_cycles = 0;
  size_t s_cxl_link_busy_cycles = 0;
  size_t s_cxl_completion_cycles = 0;

 public:
  void init() override {
    init_base();
    RAMULATOR_PARSE_PARAM(
        m_bandwidth_bytes_per_second, double, "cxl_bandwidth_bytes_per_second").required();
    RAMULATOR_PARSE_PARAM(m_access_latency_ps, int64_t, "cxl_access_latency_ps").required();
    if (m_bandwidth_bytes_per_second <= 0.0 || m_access_latency_ps < 0) {
      throw std::runtime_error("CXL bandwidth must be positive and latency nonnegative");
    }
    m_tick_ps = m_device.m_spec->get_timing_value("tCK_ps");
    const int tx_bytes = m_device.m_spec->get_tx_bytes();
    m_issue_interval_ps = static_cast<int64_t>(
        (static_cast<double>(tx_bytes) * 1.0e12 / m_bandwidth_bytes_per_second) + 0.999999);
    if (m_issue_interval_ps <= 0) {
      m_issue_interval_ps = 1;
    }
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    setup_base(frontend, memory_system);
    m_stats.add("cxl_requests_served", s_cxl_requests_served);
    m_stats.add("cxl_queue_wait_cycles", s_cxl_queue_wait_cycles);
    m_stats.add("cxl_max_queue_wait_cycles", s_cxl_max_queue_wait_cycles);
    m_stats.add("cxl_request_residence_cycles", s_cxl_request_residence_cycles);
    m_stats.add("cxl_max_request_residence_cycles", s_cxl_max_request_residence_cycles);
    m_stats.add("cxl_link_busy_cycles", s_cxl_link_busy_cycles);
    m_stats.add("cxl_completion_cycles", s_cxl_completion_cycles);
  }

  void tick() override {
    tick_prologue();
    const Clk_t current_ps = m_clk * m_tick_ps;
    if (m_read_buffer.size() > 0 && current_ps >= m_next_issue_ps) {
      auto request = m_read_buffer.begin();
      const Clk_t queue_wait = m_clk - request->arrive;
      const Clk_t service_cycles = std::max<Clk_t>(
          1, (m_access_latency_ps + m_tick_ps - 1) / m_tick_ps);
      const Clk_t depart = m_clk + service_cycles;
      request->depart = depart;
      m_pending.push_back(*request);
      m_read_buffer.remove(request);

      s_num_read_reqs_served++;
      s_cxl_requests_served++;
      s_cxl_queue_wait_cycles += static_cast<size_t>(queue_wait);
      s_cxl_max_queue_wait_cycles = std::max(
          s_cxl_max_queue_wait_cycles, static_cast<size_t>(queue_wait));
      const size_t residence = static_cast<size_t>(depart - request->arrive);
      s_cxl_request_residence_cycles += residence;
      s_cxl_max_request_residence_cycles = std::max(
          s_cxl_max_request_residence_cycles, residence);
      s_cxl_completion_cycles = std::max(
          s_cxl_completion_cycles, static_cast<size_t>(depart));
      const int64_t next_ps = std::max(
          m_next_issue_ps + m_issue_interval_ps, current_ps + m_issue_interval_ps);
      m_next_issue_ps = next_ps;
      s_cxl_link_busy_cycles += static_cast<size_t>(
          (m_issue_interval_ps + m_tick_ps - 1) / m_tick_ps);
    }
  }
};

}  // namespace Ramulator
