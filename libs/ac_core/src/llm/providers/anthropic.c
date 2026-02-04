/**
 * @file anthropic.c
 * @brief Anthropic Claude API provider
 *
 * Supports Claude models via Anthropic's API.
 * Features:
 * - Extended thinking (thinking blocks with signature)
 * - Tool calling
 * - Content block parsing
 *
 * API documentation: https://docs.anthropic.com/
 */

#include "../llm_provider.h"
#include "../message/message_json.h"
#include "arc/message.h"
#include "arc/platform.h"
#include "arc/log.h"
#include "http_client.h"
#include "cJSON.h"
#include <string.h>
#include <stdio.h>

#define ANTHROPIC_API_VERSION "2023-06-01"
#define ANTHROPIC_THINKING_MIN_BUDGET 1024

/*============================================================================
 * HTTP Pool Integration (weak symbols for optional linking)
 *============================================================================*/

/* Weak declarations - resolved at link time if ac_hosted is linked */
__attribute__((weak)) int ac_http_pool_is_initialized(void);
__attribute__((weak)) arc_http_client_t *ac_http_pool_acquire(uint32_t timeout_ms);
__attribute__((weak)) void ac_http_pool_release(arc_http_client_t *client);

/**
 * @brief Check if HTTP pool is available and initialized
 */
static int http_pool_available(void) {
    return ac_http_pool_is_initialized && ac_http_pool_is_initialized();
}

/*============================================================================
 * Anthropic Provider Private Data
 *============================================================================*/

typedef struct {
    arc_http_client_t *http;  /**< Owned HTTP client (NULL if using pool) */
    int owns_http;               /**< 1 if we created the client, 0 if from pool */
} anthropic_priv_t;

/*============================================================================
 * Provider Implementation
 *============================================================================*/

static void* anthropic_create(const ac_llm_params_t* params) {
    if (!params) {
        return NULL;
    }

    anthropic_priv_t* priv = ARC_CALLOC(1, sizeof(anthropic_priv_t));
    if (!priv) {
        return NULL;
    }

    /* Check if HTTP pool is available */
    if (http_pool_available()) {
        /* Will acquire from pool on each request */
        priv->http = NULL;
        priv->owns_http = 0;
        AC_LOG_DEBUG("Anthropic provider initialized (using HTTP pool)");
    } else {
        /* Create own HTTP client */
        arc_http_client_config_t config = {
            .default_timeout_ms = 60000,  /* Default 60s timeout */
        };

        arc_err_t err = arc_http_client_create(&config, &priv->http);
        if (err != ARC_OK) {
            ARC_FREE(priv);
            return NULL;
        }
        priv->owns_http = 1;
        AC_LOG_DEBUG("Anthropic provider initialized (using own HTTP client)");
    }

    return priv;
}

static arc_err_t anthropic_chat(
    void* priv_data,
    const ac_llm_params_t* params,
    const ac_message_t* messages,
    const char* tools,
    ac_chat_response_t* response
) {
    if (!priv_data || !params || !response) {
        return ARC_ERR_INVALID_ARG;
    }

    anthropic_priv_t* priv = (anthropic_priv_t*)priv_data;
    arc_http_client_t* http = NULL;
    int from_pool = 0;

    /* Get HTTP client: from pool or owned */
    if (priv->owns_http) {
        http = priv->http;
    } else if (http_pool_available()) {
        http = ac_http_pool_acquire(params->timeout_ms > 0 ? params->timeout_ms : 60000);
        if (!http) {
            AC_LOG_ERROR("Anthropic: failed to acquire HTTP client from pool");
            return ARC_ERR_TIMEOUT;
        }
        from_pool = 1;
    } else {
        AC_LOG_ERROR("Anthropic: no HTTP client available");
        return ARC_ERR_NOT_INITIALIZED;
    }

    /* Build URL */
    const char* api_base = params->api_base ? params->api_base : "https://api.anthropic.com";
    char url[512];
    snprintf(url, sizeof(url), "%s/v1/messages", api_base);

    /* Build request JSON */
    cJSON* root = cJSON_CreateObject();
    if (!root) {
        if (from_pool) ac_http_pool_release(http);
        return ARC_ERR_NO_MEMORY;
    }

    cJSON_AddStringToObject(root, "model", params->model);
    cJSON_AddNumberToObject(root, "max_tokens", params->max_tokens > 0 ? params->max_tokens : 4096);

    /* Anthropic uses separate system field, not in messages */
    if (params->instructions) {
        cJSON_AddStringToObject(root, "system", params->instructions);
    }

    /* Thinking configuration */
    if (params->thinking.enabled) {
        cJSON* thinking = cJSON_CreateObject();
        cJSON_AddStringToObject(thinking, "type", "enabled");
        int budget = params->thinking.budget_tokens;
        if (budget < ANTHROPIC_THINKING_MIN_BUDGET) {
            budget = ANTHROPIC_THINKING_MIN_BUDGET;
        }
        cJSON_AddNumberToObject(thinking, "budget_tokens", budget);
        cJSON_AddItemToObject(root, "thinking", thinking);
    }

    /* Messages array (skip system messages) */
    cJSON* msgs_arr = cJSON_AddArrayToObject(root, "messages");

    for (const ac_message_t* msg = messages; msg; msg = msg->next) {
        /* Skip system messages (handled separately) */
        if (msg->role == AC_ROLE_SYSTEM) {
            continue;
        }

        /* Use Anthropic-specific format for messages with content blocks */
        cJSON* msg_obj = ac_message_to_json_anthropic(msg);
        if (msg_obj) {
            cJSON_AddItemToArray(msgs_arr, msg_obj);
        }
    }

    /* Tools */
    if (tools && strlen(tools) > 0) {
        cJSON* tools_arr = cJSON_Parse(tools);
        if (tools_arr) {
            cJSON_AddItemToObject(root, "tools", tools_arr);
        }
    }

    char* body = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!body) {
        if (from_pool) ac_http_pool_release(http);
        return ARC_ERR_NO_MEMORY;
    }

    AC_LOG_DEBUG("Anthropic request to %s: %s", url, body);

    /* Build headers */
    arc_http_header_t* headers = NULL;
    arc_http_header_append(&headers,
        arc_http_header_create("Content-Type", "application/json; charset=utf-8"));
    arc_http_header_append(&headers,
        arc_http_header_create("x-api-key", params->api_key));
    arc_http_header_append(&headers,
        arc_http_header_create("anthropic-version", ANTHROPIC_API_VERSION));

    /* Make HTTP request */
    arc_http_request_t req = {
        .url = url,
        .method = ARC_HTTP_POST,
        .headers = headers,
        .body = body,
        .body_len = strlen(body),
        .timeout_ms = params->timeout_ms > 0 ? params->timeout_ms : 60000,
        .verify_ssl = 1,
    };

    arc_http_response_t http_resp = {0};
    arc_err_t err = arc_http_request(http, &req, &http_resp);

    /* Cleanup */
    arc_http_header_free(headers);
    cJSON_free(body);

    if (err != ARC_OK) {
        AC_LOG_ERROR("Anthropic HTTP request failed: %d", err);
        arc_http_response_free(&http_resp);
        if (from_pool) ac_http_pool_release(http);
        return err;
    }

    if (http_resp.status_code != 200) {
        AC_LOG_ERROR("Anthropic HTTP %d: %s", http_resp.status_code,
            http_resp.body ? http_resp.body : "");
        arc_http_response_free(&http_resp);
        if (from_pool) ac_http_pool_release(http);
        return ARC_ERR_HTTP;
    }

    /* Parse response using Anthropic-specific parser */
    AC_LOG_DEBUG("Anthropic response: %s", http_resp.body);

    err = ac_chat_response_parse_anthropic(http_resp.body, response);

    arc_http_response_free(&http_resp);

    /* Release HTTP client back to pool */
    if (from_pool) ac_http_pool_release(http);

    if (err != ARC_OK) {
        AC_LOG_ERROR("Failed to parse Anthropic response");
        return err;
    }

    AC_LOG_DEBUG("Anthropic chat completed: blocks=%d, content=%s",
                 response->block_count,
                 response->content ? "yes" : "no");
    return ARC_OK;
}

static void anthropic_cleanup(void* priv_data) {
    if (!priv_data) {
        return;
    }

    anthropic_priv_t* priv = (anthropic_priv_t*)priv_data;

    /* Only destroy HTTP client if we own it (not from pool) */
    if (priv->owns_http && priv->http) {
        arc_http_client_destroy(priv->http);
    }

    ARC_FREE(priv);

    AC_LOG_DEBUG("Anthropic provider cleaned up");
}

/*============================================================================
 * Provider Registration
 *============================================================================*/

const ac_llm_ops_t anthropic_ops = {
    .name = "anthropic",
    .capabilities = AC_LLM_CAP_THINKING | AC_LLM_CAP_TOOLS | AC_LLM_CAP_STREAMING,
    .create = anthropic_create,
    .chat = anthropic_chat,
    .chat_stream = NULL,  /* TODO: Implement streaming */
    .cleanup = anthropic_cleanup,
};

AC_PROVIDER_REGISTER(anthropic, &anthropic_ops);
