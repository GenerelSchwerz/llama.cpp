#include "moe-cache.cuh"
#include "common.cuh"

ggml_cuda_moe_graph_plan::ggml_cuda_moe_graph_plan() :
    owner_(nullptr), graph_key_(nullptr), coverage_nodes_(nullptr), registry_generation_(0), graph_uid_(0), execution_semantic_key_(0), execution_certificate_(), coverage_epoch_(0), coverage_mmid_fingerprint_(0), graph_node_count_(0), coverage_mmid_count_(0), outcome_(GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR), n_groups_(0), n_nodes_(0), initialized_(false), inventory_complete_(false), unknown_reusable_(false) {
    for (auto & index : coverage_diagnostics_.first_node_index) {
        index = UINT32_MAX;
    }
    for (auto & index : coverage_diagnostics_.first_group_index) {
        index = UINT32_MAX;
    }
    for (auto & index : coverage_diagnostics_.first_bank_index) {
        index = UINT32_MAX;
    }
}

void ggml_cuda_moe_graph_plan::reset() {
    groups_.clear();
    mmid_inventory_.clear();
    prefill_add_id_witnesses_.clear();
    for (auto & entry : nodes_) {
        entry.node = nullptr;
    }
    owner_ = nullptr;
    graph_key_ = nullptr;
    coverage_nodes_ = nullptr;
    registry_generation_ = 0;
    graph_uid_ = 0;
    execution_semantic_key_ = 0;
    execution_certificate_ = {};
    coverage_epoch_ = 0;
    coverage_mmid_fingerprint_ = 0;
    graph_node_count_ = 0;
    coverage_mmid_count_ = 0;
    outcome_ = GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR;
    n_groups_ = 0;
    n_nodes_ = 0;
    coverage_diagnostics_ = {};
    for (auto & index : coverage_diagnostics_.first_node_index) {
        index = UINT32_MAX;
    }
    for (auto & index : coverage_diagnostics_.first_group_index) {
        index = UINT32_MAX;
    }
    for (auto & index : coverage_diagnostics_.first_bank_index) {
        index = UINT32_MAX;
    }
    initialized_ = false;
    inventory_complete_ = false;
    unknown_reusable_ = false;
}

static uint32_t ggml_cuda_moe_graph_node_hash(const ggml_tensor * node) {
    uint64_t value = reinterpret_cast<uintptr_t>(node);
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33;
    return static_cast<uint32_t>(value);
}

bool ggml_cuda_moe_graph_plan::insert(
        const ggml_tensor * node,
        uint32_t group_record,
        uint32_t role,
        uint32_t bank_index,
        uint32_t slot_index) {
    if (node == nullptr || group_record >= n_groups_ || n_nodes_ == MAX_NODE_BINDINGS) {
        return false;
    }
    uint32_t index = ggml_cuda_moe_graph_node_hash(node) & (NODE_TABLE_SIZE - 1);
    for (uint32_t probe = 0; probe < NODE_TABLE_SIZE; ++probe) {
        auto & entry = nodes_[index];
        if (entry.node == nullptr) {
            entry = {node, group_record, role, bank_index, slot_index};
            ++n_nodes_;
            return true;
        }
        if (entry.node == node) {
            return false;
        }
        index = (index + 1) & (NODE_TABLE_SIZE - 1);
    }
    return false;
}

const ggml_cuda_moe_graph_plan::node_entry * ggml_cuda_moe_graph_plan::find(const ggml_tensor * node) const {
    if (!initialized_ || node == nullptr) {
        return nullptr;
    }
    uint32_t index = ggml_cuda_moe_graph_node_hash(node) & (NODE_TABLE_SIZE - 1);
    for (uint32_t probe = 0; probe < NODE_TABLE_SIZE; ++probe) {
        const auto & entry = nodes_[index];
        if (entry.node == nullptr) {
            return nullptr;
        }
        if (entry.node == node) {
            return &entry;
        }
        index = (index + 1) & (NODE_TABLE_SIZE - 1);
    }
    return nullptr;
}

uint32_t ggml_cuda_moe_graph_plan::size() const {
    return initialized_ ? n_groups_ : 0;
}

uint64_t ggml_cuda_moe_graph_plan::registry_generation() const {
    return initialized_ ? registry_generation_ : 0;
}

uint64_t ggml_cuda_moe_graph_plan::graph_uid() const {
    return initialized_ ? graph_uid_ : 0;
}

int32_t ggml_cuda_moe_graph_plan::graph_node_count() const {
    return initialized_ ? graph_node_count_ : 0;
}

ggml_cuda_moe_graph_outcome ggml_cuda_moe_graph_plan::outcome() const {
    return initialized_ ? outcome_ : GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR;
}

bool ggml_cuda_moe_graph_plan::has_certified_complete_mmid_inventory() const {
    return initialized_ && unknown_reusable_ && inventory_complete_ && coverage_epoch_ != 0 &&
        coverage_nodes_ != nullptr && coverage_mmid_fingerprint_ != 0 &&
        coverage_mmid_count_ == coverage_diagnostics_.cached_mmid &&
        mmid_inventory_.size() == coverage_mmid_count_;
}

const ggml_cuda_moe_graph_coverage_diagnostics & ggml_cuda_moe_graph_plan::coverage_diagnostics() const {
    return coverage_diagnostics_;
}

ggml_cuda_moe_graph_execution::ggml_cuda_moe_graph_execution() :
        plan_(nullptr), owner_(nullptr), n_groups_(0), dispatch_mode_(GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY), dispatch_active_(false) {
}

ggml_cuda_moe_graph_execution::~ggml_cuda_moe_graph_execution() {
    reset();
}

void ggml_cuda_moe_graph_execution::reset() {
    if (dispatch_active_ && owner_ != nullptr) {
        (void) owner_->finish_graph_dispatch(this);
    }
    plan_lease_.reset();
    plan_ = nullptr;
    owner_ = nullptr;
    n_groups_ = 0;
    dispatch_mode_ = GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY;
    dispatch_active_ = false;
}

void ggml_cuda_moe_graph_execution::retain(const std::shared_ptr<const ggml_cuda_moe_graph_plan> & plan) {
    plan_lease_ = plan;
    plan_ = plan.get();
}

bool ggml_cuda_moe_graph_execution::find(const ggml_tensor * node, ggml_cuda_moe_graph_binding * binding) const {
    if (plan_ == nullptr) {
        return false;
    }
    const auto * entry = plan_->find(node);
    if (entry == nullptr || entry->group_record >= n_groups_) {
        return false;
    }
    if (binding != nullptr) {
        binding->key = groups_[entry->group_record].key;
        binding->role = entry->role;
        binding->bank_index = entry->bank_index;
        binding->slot_index = entry->slot_index;
    }
    return true;
}

bool ggml_cuda_moe_graph_execution::rejects_cached_mmid(const ggml_tensor * node) const {
    const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
    const bool cached = source != nullptr && source->buffer != nullptr &&
        ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer));
    const bool legacy = plan_ != nullptr &&
        (plan_->outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY ||
            plan_->outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    return cached && plan_ != nullptr && !legacy &&
        plan_->find(node) == nullptr;
}

ggml_cuda_moe_graph_group_dispatch * ggml_cuda_moe_graph_execution::find_group(
        const ggml_tensor * node,
        ggml_cuda_moe_graph_binding * binding) {
    if (plan_ == nullptr) {
        return nullptr;
    }
    const auto * entry = plan_->find(node);
    if (entry == nullptr || entry->group_record >= n_groups_) {
        return nullptr;
    }
    if (binding != nullptr) {
        binding->key = groups_[entry->group_record].key;
        binding->role = entry->role;
        binding->bank_index = entry->bank_index;
        binding->slot_index = entry->slot_index;
    }
    return &groups_[entry->group_record];
}

const ggml_cuda_moe_group_call_lease * ggml_cuda_moe_graph_execution::find_authority(const ggml_tensor * node) const {
    if (plan_ == nullptr || !dispatch_active_) {
        return nullptr;
    }
    const auto * entry = plan_->find(node);
    if (entry == nullptr || entry->group_record >= n_groups_) {
        return nullptr;
    }
    const auto & authority = groups_[entry->group_record].authority;
    return authority ? &authority : nullptr;
}

bool ggml_cuda_moe_graph_execution::resolve_streams(ggml_cuda_moe_graph_stream_resolver resolver, void * data) {
    if (plan_ == nullptr || resolver == nullptr || dispatch_active_) {
        return false;
    }
    for (uint32_t record_index = 0; record_index < n_groups_; ++record_index) {
        auto & dispatch = groups_[record_index];
        dispatch.stream = nullptr;
        if (plan_->outcome_ != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
            continue;
        }
        const auto & record = plan_->groups_[record_index];
        dispatch.stream = resolver(data, record.ids_root.tensor);
        if (dispatch.stream == nullptr) {
            continue;
        }
        for (const ggml_tensor * node : record.nodes) {
            if (node == nullptr) {
                continue;
            }
            cudaStream_t stream = resolver(data, node);
            if (stream == nullptr || (dispatch.stream != nullptr && dispatch.stream != stream)) {
                dispatch.stream = nullptr;
                break;
            }
            dispatch.stream = stream;
        }
    }
    return true;
}

bool ggml_cuda_moe_graph_execution::has_stream_grouped_candidate() const {
    if (plan_ == nullptr || dispatch_active_) {
        return false;
    }
    for (uint32_t record_index = 0; record_index < n_groups_; ++record_index) {
        const auto & dispatch = groups_[record_index];
        if (plan_->outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && dispatch.stream != nullptr) {
            return true;
        }
    }
    return false;
}

bool ggml_cuda_moe_graph_execution::has_coherent_grouped_streams() const {
    if (plan_ == nullptr || dispatch_active_ || plan_->outcome_ != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED || n_groups_ == 0) {
        return false;
    }
    for (uint32_t record_index = 0; record_index < n_groups_; ++record_index) {
        if (groups_[record_index].stream == nullptr) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_moe_graph_execution::has_explicit_grouped_strategies() const {
    if (plan_ == nullptr || dispatch_active_ || plan_->outcome_ != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED || n_groups_ == 0) {
        return false;
    }
    for (uint32_t record_index = 0; record_index < n_groups_; ++record_index) {
        const uint32_t strategy = groups_[record_index].strategy;
        if (strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT &&
                strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_moe_graph_execution::allows_graph_capture() const {
    if (!has_explicit_grouped_strategies()) {
        return false;
    }
    for (uint32_t record_index = 0; record_index < n_groups_; ++record_index) {
        if (groups_[record_index].strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_moe_graph_execution::requires_dispatch() const {
    return plan_ != nullptr && (n_groups_ != 0 ||
        (plan_->outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && plan_->coverage_diagnostics_.cached_mmid != 0));
}

bool ggml_cuda_moe_graph_execution::has_prefill_resident_witnesses() const {
    return plan_ != nullptr && !plan_->prefill_add_id_witnesses_.empty();
}

ggml_cuda_moe_graph_outcome ggml_cuda_moe_graph_execution::outcome() const {
    return plan_ != nullptr ? plan_->outcome_ : GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR;
}

ggml_cuda_moe_graph_dispatch_mode ggml_cuda_moe_graph_execution::dispatch_mode() const {
    return dispatch_mode_;
}

uint32_t ggml_cuda_moe_graph_execution::size() const {
    return plan_ != nullptr ? n_groups_ : 0;
}

bool ggml_cuda_moe_required_grouped_plan_ready(
        const ggml_cuda_moe_graph_plan & plan,
        const ggml_cuda_moe_graph_execution & execution) {
    return plan.has_certified_complete_mmid_inventory() &&
        plan.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
        execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
        execution.requires_dispatch() && execution.has_explicit_grouped_strategies();
}

ggml_cuda_moe_group_call_lease::ggml_cuda_moe_group_call_lease() noexcept = default;

ggml_cuda_moe_group_call_lease::~ggml_cuda_moe_group_call_lease() {
    if (owner_ != nullptr) {
        owner_->end_group_call(*this);
    }
}

ggml_cuda_moe_group_call_lease::ggml_cuda_moe_group_call_lease(ggml_cuda_moe_group_call_lease && other) noexcept :
        owner_(other.owner_),
        candidate_generation_(other.candidate_generation_),
        authority_epoch_(other.authority_epoch_),
        group_index_(other.group_index_),
        authority_(other.authority_),
        execution_domain_(other.execution_domain_),
        row_semantics_(other.row_semantics_),
        prefill_resident_certified_(other.prefill_resident_certified_) {
    other.owner_ = nullptr;
}

ggml_cuda_moe_group_call_lease & ggml_cuda_moe_group_call_lease::operator=(ggml_cuda_moe_group_call_lease && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (owner_ != nullptr) {
        owner_->end_group_call(*this);
    }
    owner_ = other.owner_;
    candidate_generation_ = other.candidate_generation_;
    authority_epoch_ = other.authority_epoch_;
    group_index_ = other.group_index_;
    authority_ = other.authority_;
    execution_domain_ = other.execution_domain_;
    row_semantics_ = other.row_semantics_;
    prefill_resident_certified_ = other.prefill_resident_certified_;
    other.owner_ = nullptr;
    return *this;
}

ggml_cuda_moe_group_call_lease::operator bool() const noexcept {
    return owner_ != nullptr;
}

ggml_cuda_moe_group_authority ggml_cuda_moe_group_call_lease::authority() const noexcept {
    return authority_;
}

bool ggml_cuda_moe_group_call_lease::legacy_telemetry_is_decode(bool fallback) const noexcept {
    if (owner_ == nullptr || execution_domain_ < GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
            execution_domain_ > GGML_GRAPH_EXECUTION_DOMAIN_MTP) {
        return fallback;
    }
    if (row_semantics_ == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL) {
        return false;
    }
    if (row_semantics_ == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ||
            row_semantics_ == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE) {
        return true;
    }
    return fallback;
}

ggml_cuda_moe_legacy_operation_lease::ggml_cuda_moe_legacy_operation_lease() noexcept = default;

ggml_cuda_moe_legacy_operation_lease::~ggml_cuda_moe_legacy_operation_lease() {
    if (owner_ != nullptr) {
        owner_->end_legacy_operation(*this);
    }
}

ggml_cuda_moe_legacy_operation_lease::ggml_cuda_moe_legacy_operation_lease(
        ggml_cuda_moe_legacy_operation_lease && other) noexcept : owner_(other.owner_) {
    other.owner_ = nullptr;
}

ggml_cuda_moe_legacy_operation_lease & ggml_cuda_moe_legacy_operation_lease::operator=(
        ggml_cuda_moe_legacy_operation_lease && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (owner_ != nullptr) {
        owner_->end_legacy_operation(*this);
    }
    owner_ = other.owner_;
    other.owner_ = nullptr;
    return *this;
}

ggml_cuda_moe_legacy_operation_lease::operator bool() const noexcept {
    return owner_ != nullptr;
}

ggml_cuda_moe_legacy_cache_lease::ggml_cuda_moe_legacy_cache_lease() noexcept = default;

ggml_cuda_moe_legacy_cache_lease::~ggml_cuda_moe_legacy_cache_lease() {
    if (owner_ != nullptr) {
        owner_->release_legacy_cache(*this);
    }
}

ggml_cuda_moe_legacy_cache_lease::ggml_cuda_moe_legacy_cache_lease(ggml_cuda_moe_legacy_cache_lease && other) noexcept :
        owner_(other.owner_),
        record_(other.record_),
        cache_(other.cache_),
        acquisition_(other.acquisition_) {
    other.owner_ = nullptr;
    other.record_ = nullptr;
    other.cache_ = nullptr;
    other.acquisition_ = {};
}

ggml_cuda_moe_legacy_cache_lease & ggml_cuda_moe_legacy_cache_lease::operator=(ggml_cuda_moe_legacy_cache_lease && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (owner_ != nullptr) {
        owner_->release_legacy_cache(*this);
    }
    owner_ = other.owner_;
    record_ = other.record_;
    cache_ = other.cache_;
    acquisition_ = other.acquisition_;
    other.owner_ = nullptr;
    other.record_ = nullptr;
    other.cache_ = nullptr;
    other.acquisition_ = {};
    return *this;
}

ggml_cuda_moe_legacy_cache_lease::operator bool() const noexcept {
    return owner_ != nullptr;
}

const ggml_cuda_moe_legacy_acquisition & ggml_cuda_moe_legacy_cache_lease::acquisition() const noexcept {
    return acquisition_;
}

ggml_cuda_moe_cache * ggml_cuda_moe_legacy_cache_lease::get() const noexcept {
    return cache_;
}

ggml_cuda_moe_grouped_host_staging_lease::ggml_cuda_moe_grouped_host_staging_lease() noexcept = default;

ggml_cuda_moe_grouped_host_staging_lease::~ggml_cuda_moe_grouped_host_staging_lease() {
    if (owner_ != nullptr) {
        owner_->release_grouped_host_staging(*this);
    }
}

ggml_cuda_moe_grouped_host_staging_lease::ggml_cuda_moe_grouped_host_staging_lease(
        ggml_cuda_moe_grouped_host_staging_lease && other) noexcept :
        owner_(other.owner_),
        transaction_(other.transaction_),
        cache_(other.cache_),
        capability_(other.capability_),
        bank_index_(other.bank_index_) {
    other.owner_ = nullptr;
    other.transaction_ = {};
    other.cache_ = nullptr;
    other.capability_ = nullptr;
    other.bank_index_ = UINT32_MAX;
}

ggml_cuda_moe_grouped_host_staging_lease & ggml_cuda_moe_grouped_host_staging_lease::operator=(
        ggml_cuda_moe_grouped_host_staging_lease && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (owner_ != nullptr) {
        owner_->release_grouped_host_staging(*this);
    }
    owner_ = other.owner_;
    transaction_ = other.transaction_;
    cache_ = other.cache_;
    capability_ = other.capability_;
    bank_index_ = other.bank_index_;
    other.owner_ = nullptr;
    other.transaction_ = {};
    other.cache_ = nullptr;
    other.capability_ = nullptr;
    other.bank_index_ = UINT32_MAX;
    return *this;
}

ggml_cuda_moe_grouped_host_staging_lease::operator bool() const noexcept {
    return owner_ != nullptr && cache_ != nullptr && capability_ != nullptr;
}

ggml_cuda_moe_cache * ggml_cuda_moe_grouped_host_staging_lease::get() const noexcept {
    return cache_;
}

const ggml_cuda_moe_graph_capability_witness & ggml_cuda_moe_grouped_host_staging_lease::capability() const noexcept {
    GGML_ASSERT(capability_ != nullptr);
    return *capability_;
}
