/**
 * @file message.c
 * @brief Message implementation
 */

#include "arc/message.h"
#include "arc/log.h"
#include <string.h>

/*============================================================================
 * Role Helper
 *============================================================================*/

const char* ac_role_to_string(ac_role_t role) {
    switch (role) {
        case AC_ROLE_SYSTEM:    return "system";
        case AC_ROLE_USER:      return "user";
        case AC_ROLE_ASSISTANT: return "assistant";
        case AC_ROLE_TOOL:      return "tool";
        default:                return "unknown";
    }
}

/*============================================================================
 * Message Creation
 *============================================================================*/

ac_message_t* ac_message_create(arena_t* arena, ac_role_t role, const char* content) {
    if (!arena || !content) {
        AC_LOG_ERROR("Invalid arguments to ac_message_create");
        return NULL;
    }

    ac_message_t* msg = (ac_message_t*)arena_alloc(arena, sizeof(ac_message_t));
    if (!msg) {
        AC_LOG_ERROR("Failed to allocate message from arena");
        return NULL;
    }

    msg->role = role;
    msg->content = arena_strdup(arena, content);
    msg->blocks = NULL;
    msg->tool_call_id = NULL;
    msg->tool_calls = NULL;
    msg->next = NULL;

    if (!msg->content) {
        AC_LOG_ERROR("Failed to duplicate message content");
        return NULL;
    }

    return msg;
}

ac_message_t* ac_message_create_tool_result(
    arena_t* arena,
    const char* tool_call_id,
    const char* content
) {
    if (!arena || !tool_call_id || !content) {
        AC_LOG_ERROR("Invalid arguments to ac_message_create_tool_result");
        return NULL;
    }

    ac_message_t* msg = (ac_message_t*)arena_alloc(arena, sizeof(ac_message_t));
    if (!msg) {
        AC_LOG_ERROR("Failed to allocate message from arena");
        return NULL;
    }

    msg->role = AC_ROLE_TOOL;
    msg->content = arena_strdup(arena, content);
    msg->blocks = NULL;
    msg->tool_call_id = arena_strdup(arena, tool_call_id);
    msg->tool_calls = NULL;
    msg->next = NULL;

    if (!msg->content || !msg->tool_call_id) {
        AC_LOG_ERROR("Failed to duplicate message strings");
        return NULL;
    }

    return msg;
}

/*============================================================================
 * Message List Operations
 *============================================================================*/

void ac_message_append(ac_message_t** list, ac_message_t* message) {
    if (!list || !message) {
        return;
    }

    if (!*list) {
        *list = message;
        return;
    }

    ac_message_t* tail = *list;
    while (tail->next) {
        tail = tail->next;
    }
    tail->next = message;
}

size_t ac_message_count(const ac_message_t* list) {
    size_t count = 0;
    const ac_message_t* curr = list;

    while (curr) {
        count++;
        curr = curr->next;
    }

    return count;
}

/*============================================================================
 * Tool Call Operations
 *============================================================================*/

ac_tool_call_t* ac_tool_call_create(
    arena_t* arena,
    const char* id,
    const char* name,
    const char* arguments
) {
    if (!arena || !id || !name) {
        AC_LOG_ERROR("Invalid arguments to ac_tool_call_create");
        return NULL;
    }

    ac_tool_call_t* call = (ac_tool_call_t*)arena_alloc(arena, sizeof(ac_tool_call_t));
    if (!call) {
        AC_LOG_ERROR("Failed to allocate tool call from arena");
        return NULL;
    }

    call->id = arena_strdup(arena, id);
    call->name = arena_strdup(arena, name);
    call->arguments = arguments ? arena_strdup(arena, arguments) : NULL;
    call->next = NULL;

    if (!call->id || !call->name) {
        AC_LOG_ERROR("Failed to duplicate tool call strings");
        return NULL;
    }

    return call;
}

void ac_tool_call_append(ac_tool_call_t** list, ac_tool_call_t* call) {
    if (!list || !call) {
        return;
    }

    if (!*list) {
        *list = call;
        return;
    }

    ac_tool_call_t* tail = *list;
    while (tail->next) {
        tail = tail->next;
    }
    tail->next = call;
}

ac_message_t* ac_message_create_with_tool_calls(
    arena_t* arena,
    const char* content,
    ac_tool_call_t* tool_calls
) {
    if (!arena) {
        AC_LOG_ERROR("Invalid arena in ac_message_create_with_tool_calls");
        return NULL;
    }

    ac_message_t* msg = (ac_message_t*)arena_alloc(arena, sizeof(ac_message_t));
    if (!msg) {
        AC_LOG_ERROR("Failed to allocate message from arena");
        return NULL;
    }

    msg->role = AC_ROLE_ASSISTANT;
    msg->content = content ? arena_strdup(arena, content) : NULL;
    msg->tool_call_id = NULL;
    msg->tool_calls = tool_calls;
    msg->blocks = NULL;
    msg->next = NULL;

    return msg;
}

/*============================================================================
 * Content Block Operations (v2)
 *============================================================================*/

const char* ac_block_type_to_string(ac_block_type_t type) {
    switch (type) {
        case AC_BLOCK_TEXT:              return "text";
        case AC_BLOCK_THINKING:          return "thinking";
        case AC_BLOCK_REDACTED_THINKING: return "redacted_thinking";
        case AC_BLOCK_REASONING:         return "reasoning";
        case AC_BLOCK_TOOL_USE:          return "tool_use";
        case AC_BLOCK_TOOL_RESULT:       return "tool_result";
        default:                         return "unknown";
    }
}

ac_content_block_t* ac_block_create_text(arena_t* arena, const char* text) {
    if (!arena || !text) {
        AC_LOG_ERROR("Invalid arguments to ac_block_create_text");
        return NULL;
    }

    ac_content_block_t* block = (ac_content_block_t*)arena_alloc(arena, sizeof(ac_content_block_t));
    if (!block) {
        AC_LOG_ERROR("Failed to allocate content block from arena");
        return NULL;
    }

    memset(block, 0, sizeof(ac_content_block_t));
    block->type = AC_BLOCK_TEXT;
    block->text = arena_strdup(arena, text);

    if (!block->text) {
        AC_LOG_ERROR("Failed to duplicate text content");
        return NULL;
    }

    return block;
}

ac_content_block_t* ac_block_create_thinking(
    arena_t* arena,
    const char* thinking,
    const char* signature
) {
    if (!arena || !thinking) {
        AC_LOG_ERROR("Invalid arguments to ac_block_create_thinking");
        return NULL;
    }

    ac_content_block_t* block = (ac_content_block_t*)arena_alloc(arena, sizeof(ac_content_block_t));
    if (!block) {
        AC_LOG_ERROR("Failed to allocate content block from arena");
        return NULL;
    }

    memset(block, 0, sizeof(ac_content_block_t));
    block->type = AC_BLOCK_THINKING;
    block->text = arena_strdup(arena, thinking);
    block->signature = signature ? arena_strdup(arena, signature) : NULL;

    if (!block->text) {
        AC_LOG_ERROR("Failed to duplicate thinking content");
        return NULL;
    }

    return block;
}

ac_content_block_t* ac_block_create_redacted(arena_t* arena, const char* data) {
    if (!arena || !data) {
        AC_LOG_ERROR("Invalid arguments to ac_block_create_redacted");
        return NULL;
    }

    ac_content_block_t* block = (ac_content_block_t*)arena_alloc(arena, sizeof(ac_content_block_t));
    if (!block) {
        AC_LOG_ERROR("Failed to allocate content block from arena");
        return NULL;
    }

    memset(block, 0, sizeof(ac_content_block_t));
    block->type = AC_BLOCK_REDACTED_THINKING;
    block->data = arena_strdup(arena, data);

    if (!block->data) {
        AC_LOG_ERROR("Failed to duplicate redacted data");
        return NULL;
    }

    return block;
}

ac_content_block_t* ac_block_create_tool_use(
    arena_t* arena,
    const char* id,
    const char* name,
    const char* input
) {
    if (!arena || !id || !name) {
        AC_LOG_ERROR("Invalid arguments to ac_block_create_tool_use");
        return NULL;
    }

    ac_content_block_t* block = (ac_content_block_t*)arena_alloc(arena, sizeof(ac_content_block_t));
    if (!block) {
        AC_LOG_ERROR("Failed to allocate content block from arena");
        return NULL;
    }

    memset(block, 0, sizeof(ac_content_block_t));
    block->type = AC_BLOCK_TOOL_USE;
    block->id = arena_strdup(arena, id);
    block->name = arena_strdup(arena, name);
    block->input = input ? arena_strdup(arena, input) : NULL;

    if (!block->id || !block->name) {
        AC_LOG_ERROR("Failed to duplicate tool use strings");
        return NULL;
    }

    return block;
}

ac_content_block_t* ac_block_create_tool_result(
    arena_t* arena,
    const char* tool_use_id,
    const char* content,
    int is_error
) {
    if (!arena || !tool_use_id || !content) {
        AC_LOG_ERROR("Invalid arguments to ac_block_create_tool_result");
        return NULL;
    }

    ac_content_block_t* block = (ac_content_block_t*)arena_alloc(arena, sizeof(ac_content_block_t));
    if (!block) {
        AC_LOG_ERROR("Failed to allocate content block from arena");
        return NULL;
    }

    memset(block, 0, sizeof(ac_content_block_t));
    block->type = AC_BLOCK_TOOL_RESULT;
    block->id = arena_strdup(arena, tool_use_id);
    block->text = arena_strdup(arena, content);
    block->is_error = is_error;

    if (!block->id || !block->text) {
        AC_LOG_ERROR("Failed to duplicate tool result strings");
        return NULL;
    }

    return block;
}

void ac_block_append(ac_content_block_t** list, ac_content_block_t* block) {
    if (!list || !block) {
        return;
    }

    if (!*list) {
        *list = block;
        return;
    }

    ac_content_block_t* tail = *list;
    while (tail->next) {
        tail = tail->next;
    }
    tail->next = block;
}

size_t ac_block_count(const ac_content_block_t* list) {
    size_t count = 0;
    const ac_content_block_t* curr = list;

    while (curr) {
        count++;
        curr = curr->next;
    }

    return count;
}

/*============================================================================
 * Response Helper Functions (v2)
 *============================================================================*/

const char* ac_response_text(const ac_chat_response_t* resp) {
    if (!resp) return NULL;
    
    /* If blocks exist, combine AC_BLOCK_TEXT blocks */
    if (resp->blocks) {
        /* For now, return first text block content */
        /* TODO: combine multiple text blocks if needed */
        for (ac_content_block_t* b = resp->blocks; b; b = b->next) {
            if (b->type == AC_BLOCK_TEXT && b->text) {
                return b->text;
            }
        }
        return NULL;
    }
    
    /* Fall back to legacy content field */
    return resp->content;
}

const char* ac_response_thinking(const ac_chat_response_t* resp) {
    if (!resp || !resp->blocks) return NULL;
    
    /* Return first thinking block content */
    for (ac_content_block_t* b = resp->blocks; b; b = b->next) {
        if (b->type == AC_BLOCK_THINKING && b->text) {
            return b->text;
        }
    }
    
    return NULL;
}

ac_message_t* ac_message_from_response(arena_t* arena, const ac_chat_response_t* resp) {
    if (!arena || !resp) {
        AC_LOG_ERROR("Invalid arguments to ac_message_from_response");
        return NULL;
    }

    ac_message_t* msg = (ac_message_t*)arena_alloc(arena, sizeof(ac_message_t));
    if (!msg) {
        AC_LOG_ERROR("Failed to allocate message from arena");
        return NULL;
    }

    memset(msg, 0, sizeof(ac_message_t));
    msg->role = AC_ROLE_ASSISTANT;

    /* Copy content blocks if present (preserve order for thinking models) */
    if (resp->blocks) {
        ac_content_block_t* last_block = NULL;
        
        for (ac_content_block_t* src = resp->blocks; src; src = src->next) {
            ac_content_block_t* dst = (ac_content_block_t*)arena_alloc(arena, sizeof(ac_content_block_t));
            if (!dst) {
                AC_LOG_ERROR("Failed to allocate content block");
                return NULL;
            }
            
            memset(dst, 0, sizeof(ac_content_block_t));
            dst->type = src->type;
            
            /* Copy type-specific fields */
            if (src->text) dst->text = arena_strdup(arena, src->text);
            if (src->signature) dst->signature = arena_strdup(arena, src->signature);
            if (src->data) dst->data = arena_strdup(arena, src->data);
            if (src->id) dst->id = arena_strdup(arena, src->id);
            if (src->name) dst->name = arena_strdup(arena, src->name);
            if (src->input) dst->input = arena_strdup(arena, src->input);
            dst->is_error = src->is_error;
            
            if (!msg->blocks) {
                msg->blocks = dst;
            } else {
                last_block->next = dst;
            }
            last_block = dst;
        }
    }
    
    /* Also set legacy content field for backward compatibility */
    if (resp->content) {
        msg->content = arena_strdup(arena, resp->content);
    } else {
        /* Try to extract text from blocks */
        const char* text = ac_response_text(resp);
        if (text) {
            msg->content = arena_strdup(arena, text);
        }
    }
    
    /* Copy legacy tool calls if present */
    if (resp->tool_calls) {
        ac_tool_call_t* last_call = NULL;
        
        for (ac_tool_call_t* src = resp->tool_calls; src; src = src->next) {
            ac_tool_call_t* dst = ac_tool_call_create(arena, src->id, src->name, src->arguments);
            if (!dst) {
                AC_LOG_ERROR("Failed to copy tool call");
                return NULL;
            }
            
            if (!msg->tool_calls) {
                msg->tool_calls = dst;
            } else {
                last_call->next = dst;
            }
            last_call = dst;
        }
    }

    return msg;
}
