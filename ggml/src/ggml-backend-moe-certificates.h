#pragma once

#include "ggml.h"

static inline bool ggml_backend_sched_execution_certificate_valid(const struct ggml_graph_execution_certificate * certificate) {
    static_assert(sizeof(struct ggml_graph_execution_certificate) == 96, "unexpected graph execution certificate size");

    if (certificate == nullptr ||
            certificate->magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ||
            certificate->abi_version != GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION ||
            certificate->struct_size != sizeof(*certificate) ||
            (certificate->flags & ~GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0 ||
            certificate->domain < GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
            certificate->domain > GGML_GRAPH_EXECUTION_DOMAIN_MTP ||
            certificate->row_semantics < GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ||
            certificate->row_semantics > GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE ||
            certificate->n_rows == 0 || certificate->n_sequences == 0 ||
            certificate->owner_namespace == 0 || certificate->owner_generation == 0 ||
            certificate->source_graph_uid != 0 || certificate->split_graph_uid != 0) {
        return false;
    }

    for (size_t i = 0; i < sizeof(certificate->reserved)/sizeof(certificate->reserved[0]); ++i) {
        if (certificate->reserved[i] != 0) {
            return false;
        }
    }

    return true;
}

static inline bool ggml_backend_sched_hybrid_certificate_supported(
        const ggml_graph_execution_certificate & certificate, bool source) {
    const bool main = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    const bool required = certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    if (certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL) {
        return main || (source && required && certificate.n_sequences <= certificate.n_rows);
    }
    if (certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT) {
        return certificate.n_sequences == certificate.n_rows && (source || certificate.n_rows == 1) &&
            (main ? certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE : source && required);
    }
    return main && required && certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE &&
        certificate.n_rows > 1 && certificate.n_sequences < certificate.n_rows;
}

static inline struct ggml_graph_execution_certificate ggml_backend_sched_split_certificate(
        uint64_t source_graph_uid, uint64_t split_graph_uid, struct ggml_graph_execution_certificate certificate) {
    if (certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC) { return {}; }
    certificate.source_graph_uid = source_graph_uid;
    certificate.split_graph_uid = split_graph_uid;
    return certificate;
}
