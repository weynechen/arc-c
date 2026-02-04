/**
 * @file message.h
 * @brief ArC Message Structure - Internal Interface
 *
 * Message structure for conversation history with content block support.
 * Supports thinking/reasoning blocks from models like Claude and GPT.
 * Messages are stored in agent's arena.
 */

#ifndef ARC_MESSAGE_H
#define ARC_MESSAGE_H

#include "arena.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Message Role
 *============================================================================*/

typedef enum {
    AC_ROLE_SYSTEM,
    AC_ROLE_USER,
    AC_ROLE_ASSISTANT,
    AC_ROLE_TOOL
} ac_role_t;

/*============================================================================
 * Content Block Types (v2)
 *============================================================================*/

/**
 * @brief Content block type for structured responses
 */
typedef enum {
    AC_BLOCK_TEXT,              /**< Plain text content */
    AC_BLOCK_THINKING,          /**< Thinking content with signature (Anthropic) */
    AC_BLOCK_REDACTED_THINKING, /**< Redacted/encrypted thinking (Anthropic) */
    AC_BLOCK_REASONING,         /**< Reasoning content (OpenAI) */
    AC_BLOCK_TOOL_USE,          /**< Tool/function call request */
    AC_BLOCK_TOOL_RESULT,       /**< Tool/function call result */
} ac_block_type_t;

/*============================================================================
 * Content Block Structure (v2)
 *============================================================================*/

/**
 * @brief Content block for structured message content
 *
 * Content blocks represent different types of content in a message.
 * For thinking models, blocks must be preserved and passed back unmodified.
 */
typedef struct ac_content_block {
    ac_block_type_t type;       /**< Block type */
    
    /* Type-specific data (use based on type) */
    char* text;                 /**< Text content (TEXT, THINKING) */
    char* signature;            /**< Signature for THINKING blocks (must preserve) */
    char* data;                 /**< Encrypted data for REDACTED_THINKING */
    
    /* Tool use fields */
    char* id;                   /**< Tool call ID (TOOL_USE, TOOL_RESULT) */
    char* name;                 /**< Function name (TOOL_USE) */
    char* input;                /**< JSON arguments (TOOL_USE) */
    int is_error;               /**< Error flag (TOOL_RESULT) */
    
    struct ac_content_block* next;  /**< Linked list */
} ac_content_block_t;

/*============================================================================
 * Tool Call Structure
 *============================================================================*/

/**
 * @brief Tool call from LLM response (legacy, use ac_content_block_t for v2)
 */
typedef struct ac_tool_call {
    char* id;                        /* Tool call ID (e.g., "call_abc123") */
    char* name;                      /* Function name */
    char* arguments;                 /* JSON arguments string */
    struct ac_tool_call* next;       /* Linked list for multiple tool calls */
} ac_tool_call_t;

/*============================================================================
 * Chat Response Structure
 *============================================================================*/

/**
 * @brief LLM chat completion response
 */
typedef struct {
    /* Response ID (for stateful APIs like OpenAI Responses) */
    char* id;                        /**< Response ID */
    
    /* Content blocks (v2) - ordered sequence, must preserve for multi-turn */
    ac_content_block_t* blocks;      /**< Content block list */
    int block_count;                 /**< Number of blocks */
    
    /* Legacy fields (backward compatibility) */
    char* content;                   /**< Text response (may be NULL if tool_calls) */
    ac_tool_call_t* tool_calls;      /**< Tool calls list (may be NULL) */
    int tool_call_count;             /**< Number of tool calls */

    /* Usage info */
    int input_tokens;                /**< Input tokens (v2 naming) */
    int output_tokens;               /**< Output tokens (v2 naming) */
    int thinking_tokens;             /**< Thinking tokens (Anthropic) */
    int reasoning_tokens;            /**< Reasoning tokens (OpenAI) */
    int cache_creation_tokens;       /**< Cache creation tokens */
    int cache_read_tokens;           /**< Cache read tokens */
    
    /* Legacy usage (backward compatibility) */
    int prompt_tokens;               /**< Alias for input_tokens */
    int completion_tokens;           /**< Alias for output_tokens */
    int total_tokens;                /**< Total tokens */

    /* Finish reason */
    char* finish_reason;             /**< "stop", "tool_calls", "length", etc. */
    char* stop_reason;               /**< Alias (Anthropic naming) */
} ac_chat_response_t;

/*============================================================================
 * Message Structure
 *============================================================================*/

typedef struct ac_message {
    ac_role_t role;
    
    /* Simple mode (backward compatible) */
    char* content;                   /**< Message content (stored in arena) */
    
    /* Content block mode (v2) */
    ac_content_block_t* blocks;      /**< Content blocks (for thinking models) */
    
    /* Tool related */
    char* tool_call_id;              /**< For AC_ROLE_TOOL: which tool call this responds to */
    ac_tool_call_t* tool_calls;      /**< For AC_ROLE_ASSISTANT: tool calls (legacy) */
    
    struct ac_message* next;         /**< Linked list */
} ac_message_t;

/*============================================================================
 * Message API
 *============================================================================*/

/**
 * @brief Create a message in arena
 *
 * @param arena   Arena for allocation
 * @param role    Message role
 * @param content Message content
 * @return New message, NULL on error
 */
ac_message_t* ac_message_create(arena_t* arena, ac_role_t role, const char* content);

/**
 * @brief Create a tool result message in arena
 *
 * @param arena        Arena for allocation
 * @param tool_call_id Tool call ID
 * @param content      Tool result content
 * @return New message, NULL on error
 */
ac_message_t* ac_message_create_tool_result(
    arena_t* arena,
    const char* tool_call_id,
    const char* content
);

/**
 * @brief Append message to list
 *
 * @param list     Pointer to list head
 * @param message  Message to append
 */
void ac_message_append(ac_message_t** list, ac_message_t* message);

/**
 * @brief Count messages in list
 *
 * @param list  Message list
 * @return Number of messages
 */
size_t ac_message_count(const ac_message_t* list);

/**
 * @brief Get role string
 *
 * @param role  Role enum
 * @return Role string ("system", "user", "assistant", "tool")
 */
const char* ac_role_to_string(ac_role_t role);

/*============================================================================
 * Tool Call API
 *============================================================================*/

/**
 * @brief Create a tool call in arena
 */
ac_tool_call_t* ac_tool_call_create(
    arena_t* arena,
    const char* id,
    const char* name,
    const char* arguments
);

/**
 * @brief Append tool call to list
 */
void ac_tool_call_append(ac_tool_call_t** list, ac_tool_call_t* call);

/**
 * @brief Create assistant message with tool calls
 */
ac_message_t* ac_message_create_with_tool_calls(
    arena_t* arena,
    const char* content,
    ac_tool_call_t* tool_calls
);

/*============================================================================
 * Content Block API (v2)
 *============================================================================*/

/**
 * @brief Get block type name
 */
const char* ac_block_type_to_string(ac_block_type_t type);

/**
 * @brief Create a text content block in arena
 */
ac_content_block_t* ac_block_create_text(arena_t* arena, const char* text);

/**
 * @brief Create a thinking content block in arena (Anthropic)
 *
 * @param arena     Arena for allocation
 * @param thinking  Thinking content
 * @param signature Signature (must be preserved for multi-turn)
 */
ac_content_block_t* ac_block_create_thinking(
    arena_t* arena,
    const char* thinking,
    const char* signature
);

/**
 * @brief Create a redacted thinking block in arena (Anthropic)
 *
 * @param arena  Arena for allocation
 * @param data   Encrypted data (must be preserved)
 */
ac_content_block_t* ac_block_create_redacted(arena_t* arena, const char* data);

/**
 * @brief Create a tool use content block in arena
 */
ac_content_block_t* ac_block_create_tool_use(
    arena_t* arena,
    const char* id,
    const char* name,
    const char* input
);

/**
 * @brief Create a tool result content block in arena
 */
ac_content_block_t* ac_block_create_tool_result(
    arena_t* arena,
    const char* tool_use_id,
    const char* content,
    int is_error
);

/**
 * @brief Append block to list
 */
void ac_block_append(ac_content_block_t** list, ac_content_block_t* block);

/**
 * @brief Count blocks in list
 */
size_t ac_block_count(const ac_content_block_t* list);

/**
 * @brief Free block list (heap allocated, not arena)
 */
void ac_block_free_list(ac_content_block_t* list);

/*============================================================================
 * Chat Response API
 *============================================================================*/

/**
 * @brief Initialize response structure
 */
void ac_chat_response_init(ac_chat_response_t* response);

/**
 * @brief Free response contents (not the struct itself)
 */
void ac_chat_response_free(ac_chat_response_t* response);

/**
 * @brief Parse JSON response into ac_chat_response_t
 *
 * @param json_str  Raw JSON response from LLM API
 * @param response  Output response structure
 * @return ARC_OK on success
 */
arc_err_t ac_chat_response_parse(const char* json_str, ac_chat_response_t* response);

/**
 * @brief Check if response has tool calls
 */
static inline int ac_chat_response_has_tool_calls(const ac_chat_response_t* resp) {
    return resp && resp->tool_calls && resp->tool_call_count > 0;
}

/*============================================================================
 * Response Helper Functions (v2)
 *============================================================================*/

/**
 * @brief Get combined text content from response
 *
 * Combines all AC_BLOCK_TEXT blocks into a single string.
 * Returns response->content if no blocks present.
 *
 * @param resp  Response structure
 * @return Text content (do not free, owned by response)
 */
const char* ac_response_text(const ac_chat_response_t* resp);

/**
 * @brief Get combined thinking content from response
 *
 * Combines all AC_BLOCK_THINKING blocks.
 *
 * @param resp  Response structure
 * @return Thinking content or NULL
 */
const char* ac_response_thinking(const ac_chat_response_t* resp);

/**
 * @brief Check if response has thinking blocks
 */
static inline int ac_response_has_thinking(const ac_chat_response_t* resp) {
    if (!resp || !resp->blocks) return 0;
    for (ac_content_block_t* b = resp->blocks; b; b = b->next) {
        if (b->type == AC_BLOCK_THINKING || b->type == AC_BLOCK_REDACTED_THINKING) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Create message from response (for multi-turn conversations)
 *
 * Converts response to an assistant message, preserving all content blocks
 * including thinking/signature for proper multi-turn handling.
 *
 * @param arena  Arena for allocation
 * @param resp   Response structure
 * @return New message with all blocks preserved
 */
ac_message_t* ac_message_from_response(arena_t* arena, const ac_chat_response_t* resp);

#ifdef __cplusplus
}
#endif

#endif /* ARC_MESSAGE_H */
