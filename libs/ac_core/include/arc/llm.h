/**
 * @file llm.h
 * @brief ArC LLM API - Internal Interface
 *
 * LLM abstraction using arena allocation.
 * Supports thinking/reasoning models and streaming.
 * This is an internal interface used by agents.
 */

#ifndef ARC_LLM_H
#define ARC_LLM_H

#include "arena.h"
#include "error.h"
#include "message.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * LLM Handle (opaque)
 *============================================================================*/

typedef struct ac_llm ac_llm_t;

/*============================================================================
 * LLM Capabilities
 *============================================================================*/

typedef enum {
    AC_LLM_CAP_THINKING     = (1 << 0),  /**< Supports thinking (Anthropic) */
    AC_LLM_CAP_REASONING    = (1 << 1),  /**< Supports reasoning (OpenAI) */
    AC_LLM_CAP_STREAMING    = (1 << 2),  /**< Supports streaming */
    AC_LLM_CAP_STATEFUL     = (1 << 3),  /**< Supports stateful mode (OpenAI Responses) */
    AC_LLM_CAP_TOOLS        = (1 << 4),  /**< Supports tool/function calling */
    AC_LLM_CAP_VISION       = (1 << 5),  /**< Supports vision/images */
} ac_llm_capability_t;

/*============================================================================
 * Stream Event Types
 *============================================================================*/

typedef enum {
    AC_STREAM_MESSAGE_START,       /**< Message started */
    AC_STREAM_CONTENT_BLOCK_START, /**< Content block started */
    AC_STREAM_DELTA,               /**< Content delta */
    AC_STREAM_CONTENT_BLOCK_STOP,  /**< Content block finished */
    AC_STREAM_MESSAGE_DELTA,       /**< Message-level update */
    AC_STREAM_MESSAGE_STOP,        /**< Message finished */
    AC_STREAM_ERROR,               /**< Error occurred */
} ac_stream_event_type_t;

typedef enum {
    AC_DELTA_THINKING,     /**< thinking_delta (Anthropic) */
    AC_DELTA_TEXT,         /**< text_delta */
    AC_DELTA_INPUT_JSON,   /**< input_json_delta (tool arguments) */
    AC_DELTA_SIGNATURE,    /**< signature_delta (Anthropic) */
    AC_DELTA_REASONING,    /**< reasoning delta (OpenAI) */
} ac_delta_type_t;

/*============================================================================
 * Stream Event Structure
 *============================================================================*/

typedef struct {
    ac_stream_event_type_t type;   /**< Event type */
    
    /* Block info (for CONTENT_BLOCK_START/DELTA) */
    int block_index;               /**< Block index in response */
    ac_block_type_t block_type;    /**< Block type */
    
    /* Delta content (for DELTA events) */
    ac_delta_type_t delta_type;    /**< Delta type */
    const char* delta;             /**< Delta text */
    size_t delta_len;              /**< Delta length */
    
    /* Tool use info */
    const char* tool_id;           /**< Tool call ID */
    const char* tool_name;         /**< Tool/function name */
    
    /* Message level info (for MESSAGE_DELTA/STOP) */
    const char* stop_reason;       /**< Stop reason */
    int output_tokens;             /**< Output token count */
    
    /* Error info (for ERROR events) */
    const char* error_type;        /**< Error type */
    const char* error_msg;         /**< Error message */
} ac_stream_event_t;

/**
 * @brief Stream callback function
 *
 * Called for each streaming event.
 *
 * @param event     Stream event
 * @param user_data User context
 * @return 0 to continue, non-zero to abort
 */
typedef int (*ac_stream_callback_t)(
    const ac_stream_event_t* event,
    void* user_data
);

/*============================================================================
 * Thinking Configuration
 *============================================================================*/

typedef struct {
    int enabled;             /**< Enable thinking mode */
    int budget_tokens;       /**< Thinking token budget (min 1024 for Anthropic) */
} ac_thinking_config_t;

/*============================================================================
 * Stateful Configuration (OpenAI Responses API)
 *============================================================================*/

typedef struct {
    int store;               /**< Enable stateful mode (store: true) */
    const char* response_id; /**< Previous response ID for chaining */
    int include_encrypted;   /**< Include encrypted reasoning items */
} ac_stateful_config_t;

/*============================================================================
 * LLM Parameters
 *============================================================================*/

/**
 * @brief LLM configuration parameters
 *
 * Note: System instructions should be included in the message history as
 * AC_ROLE_SYSTEM message, not configured here. The LLM layer is a passthrough
 * that handles API communication, while the Agent layer manages instructions.
 */
typedef struct {
    /*========== Provider Selection ==========*/
    const char* provider;           /**< Provider name: "openai", "anthropic", etc. */
    const char* compatible;         /**< Compatibility mode: "openai" for OpenAI-compatible */

    /*========== LLM Configuration ==========*/
    const char* model;              /**< Model name (required) */
    const char* api_key;            /**< API key (required) */
    const char* api_base;           /**< API base URL (optional) */
    // const char* instructions;       /**< System instructions (optional) */

    /*========== Generation Parameters ==========*/
    float temperature;              /**< Sampling temperature (0.0-2.0, default: 0.7) */
    float top_p;                    /**< Nucleus sampling (0.0-1.0) */
    int max_tokens;                 /**< Max tokens to generate (0 = no limit) */
    int timeout_ms;                 /**< Request timeout in ms (default: 60000) */
    
    /*========== Thinking/Reasoning (v2) ==========*/
    ac_thinking_config_t thinking;  /**< Thinking configuration */
    
    /*========== Stateful Mode (v2) ==========*/
    ac_stateful_config_t stateful;  /**< Stateful configuration */
    
    /*========== Streaming (v2) ==========*/
    int stream;                     /**< Enable streaming mode */
} ac_llm_params_t;

/*============================================================================
 * LLM API
 *============================================================================*/

/**
 * @brief Create LLM with arena
 *
 * Creates an LLM client using the provided arena for all memory allocations.
 * All memory is freed when the arena is destroyed.
 *
 * @param arena   Arena for memory allocation
 * @param params  LLM parameters
 * @return LLM handle, NULL on error
 */
ac_llm_t* ac_llm_create(arena_t* arena, const ac_llm_params_t* params);

/**
 * @brief Chat with LLM (simple, text-only)
 *
 * Sends message history to the LLM and returns the text response.
 * The response is allocated from the LLM's arena.
 *
 * @param llm       LLM handle
 * @param messages  Message history (linked list)
 * @return Response text (allocated from arena), NULL on error
 */
char* ac_llm_chat(ac_llm_t* llm, const ac_message_t* messages);

/**
 * @brief Chat with LLM with tool support
 *
 * Sends message history with optional tools to the LLM.
 * Returns structured response that may include tool calls.
 *
 * @param llm       LLM handle
 * @param messages  Message history (linked list)
 * @param tools     JSON array of tool definitions (NULL for no tools)
 * @param response  Output response structure (caller must call ac_chat_response_free)
 * @return ARC_OK on success
 */
arc_err_t ac_llm_chat_with_tools(
    ac_llm_t* llm,
    const ac_message_t* messages,
    const char* tools,
    ac_chat_response_t* response
);

/**
 * @brief Chat with LLM with streaming support (v2)
 *
 * Sends message history with optional tools to the LLM.
 * Invokes callback for each streaming event.
 *
 * @param llm       LLM handle
 * @param messages  Message history (linked list)
 * @param tools     JSON array of tool definitions (NULL for no tools)
 * @param callback  Streaming callback (called for each event)
 * @param user_data User context passed to callback
 * @param response  Optional: accumulated final response (caller must free)
 * @return ARC_OK on success
 */
arc_err_t ac_llm_chat_stream(
    ac_llm_t* llm,
    const ac_message_t* messages,
    const char* tools,
    ac_stream_callback_t callback,
    void* user_data,
    ac_chat_response_t* response
);

/**
 * @brief Update LLM parameters
 *
 * Updates LLM parameters (e.g., for stateful response chaining).
 * Only certain fields can be updated after creation.
 *
 * @param llm     LLM handle
 * @param params  New parameters (only specific fields are used)
 * @return ARC_OK on success
 */
arc_err_t ac_llm_update_params(ac_llm_t* llm, const ac_llm_params_t* params);

/**
 * @brief Get LLM capabilities
 *
 * @param llm  LLM handle
 * @return Capability bitmask
 */
uint32_t ac_llm_get_capabilities(ac_llm_t* llm);

/**
 * @brief Cleanup LLM resources
 *
 * Cleanup provider private data (HTTP client, etc).
 * The LLM structure itself is in arena and will be freed with arena_destroy.
 * This should be called before destroying the arena.
 *
 * @param llm  LLM handle
 */
void ac_llm_cleanup(ac_llm_t* llm);

#ifdef __cplusplus
}
#endif

#endif /* ARC_LLM_H */
