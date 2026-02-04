/**
 * @file chat_stream.c
 * @brief Streaming chat demo with thinking support
 *
 * Demonstrates:
 * - Streaming LLM responses (real-time token output)
 * - Extended thinking mode (Claude thinking blocks)
 * - Direct LLM API usage (without Agent abstraction)
 *
 * Usage:
 *   1. Create .env file with ANTHROPIC_API_KEY=sk-xxx
 *   2. Run ./chat_stream
 *
 * Environment variables:
 *   ANTHROPIC_API_KEY  - Required: Anthropic API key
 *   ANTHROPIC_MODEL    - Optional: Model name (default: claude-sonnet-4-5-20250514)
 *   ENABLE_THINKING    - Optional: Enable thinking mode (default: 0)
 *   THINKING_BUDGET    - Optional: Thinking token budget (default: 10000)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <arc.h>
#include <arc/llm.h>
#include <arc/arena.h>
#include <arc/env.h>

#define MAX_INPUT_LEN 4096
#define DEFAULT_MODEL "claude-sonnet-4-5-20250514"

static volatile int g_running = 1;
static int g_thinking_mode = 0;
static int g_show_thinking = 1;

/* ANSI color codes */
#define COLOR_RESET    "\033[0m"
#define COLOR_THINKING "\033[36m"   /* Cyan for thinking */
#define COLOR_TEXT     "\033[0m"    /* Default for text */
#define COLOR_INFO     "\033[33m"   /* Yellow for info */
#define COLOR_PROMPT   "\033[32m"   /* Green for prompt */

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
    printf("\n[Interrupted]\n");
}

static void print_usage(void) {
    printf("\nCommands:\n");
    printf("  /help      - Show this help\n");
    printf("  /thinking  - Toggle thinking mode\n");
    printf("  /show      - Toggle showing thinking content\n");
    printf("  /quit      - Exit\n\n");
}

/**
 * @brief Stream callback - called for each streaming event
 */
static int stream_callback(const ac_stream_event_t* event, void* user_data) {
    (void)user_data;
    
    switch (event->type) {
        case AC_STREAM_MESSAGE_START:
            /* Message started */
            break;
            
        case AC_STREAM_CONTENT_BLOCK_START:
            if (event->block_type == AC_BLOCK_THINKING && g_show_thinking) {
                printf("%s[thinking] ", COLOR_THINKING);
                fflush(stdout);
            } else if (event->block_type == AC_BLOCK_TEXT) {
                printf("%s", COLOR_TEXT);
            } else if (event->block_type == AC_BLOCK_TOOL_USE) {
                printf("%s[tool: %s] ", COLOR_INFO, event->tool_name ? event->tool_name : "?");
                fflush(stdout);
            }
            break;
            
        case AC_STREAM_DELTA:
            if (event->delta && event->delta_len > 0) {
                if (event->delta_type == AC_DELTA_THINKING) {
                    if (g_show_thinking) {
                        printf("%.*s", (int)event->delta_len, event->delta);
                        fflush(stdout);
                    }
                } else if (event->delta_type == AC_DELTA_TEXT) {
                    printf("%.*s", (int)event->delta_len, event->delta);
                    fflush(stdout);
                } else if (event->delta_type == AC_DELTA_INPUT_JSON) {
                    /* Tool input JSON delta - can show if needed */
                }
            }
            break;
            
        case AC_STREAM_CONTENT_BLOCK_STOP:
            if (event->block_type == AC_BLOCK_THINKING && g_show_thinking) {
                printf("%s\n", COLOR_RESET);
            } else if (event->block_type == AC_BLOCK_TOOL_USE) {
                printf("%s\n", COLOR_RESET);
            }
            break;
            
        case AC_STREAM_MESSAGE_DELTA:
            /* Message level update (stop_reason, usage) */
            break;
            
        case AC_STREAM_MESSAGE_STOP:
            printf("%s\n", COLOR_RESET);
            break;
            
        case AC_STREAM_ERROR:
            printf("\n%s[Error: %s]%s\n", COLOR_INFO, 
                   event->error_msg ? event->error_msg : "Unknown", COLOR_RESET);
            return -1;  /* Abort */
            
        default:
            break;
    }
    
    return 0;  /* Continue */
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    /* Load environment */
    ac_env_load_verbose(NULL);

    /* Get API key */
    const char *api_key = ac_env_require("ANTHROPIC_API_KEY");
    if (!api_key) {
        ac_env_print_help("chat_stream");
        return 1;
    }

    /* Get optional settings */
    const char *model = ac_env_get("ANTHROPIC_MODEL", DEFAULT_MODEL);
    const char *base_url = ac_env_get("ANTHROPIC_BASE_URL", NULL);
    g_thinking_mode = atoi(ac_env_get("ENABLE_THINKING", "0"));
    int thinking_budget = atoi(ac_env_get("THINKING_BUDGET", "10000"));

    /* Setup signal handler */
    signal(SIGINT, signal_handler);

    /* Create arena for memory management */
    arena_t *arena = arena_create(1024 * 1024);  /* 1MB arena */
    if (!arena) {
        fprintf(stderr, "Failed to create arena\n");
        return 1;
    }

    /* Create LLM with streaming support */
    ac_llm_params_t llm_params = {
        .provider = "anthropic",
        .model = model,
        .api_key = api_key,
        .api_base = base_url,  
        .instructions = "You are a helpful assistant. Be concise and clear.",
        .max_tokens = 4096,
        .timeout_ms = 120000,  /* 2 minutes for streaming */
        .thinking = {
            .enabled = g_thinking_mode,
            .budget_tokens = thinking_budget,
        },
        .stream = 1,
    };

    ac_llm_t *llm = ac_llm_create(arena, &llm_params);
    if (!llm) {
        fprintf(stderr, "Failed to create LLM\n");
        arena_destroy(arena);
        return 1;
    }

    printf("\n=== ArC Streaming Chat Demo ===\n");
    printf("Model: %s\n", model);
    printf("Provider: anthropic\n");
    printf("Thinking mode: %s\n", g_thinking_mode ? "ON" : "OFF");
    if (g_thinking_mode) {
        printf("Thinking budget: %d tokens\n", thinking_budget);
    }
    printf("Type /help for commands, /quit to exit\n\n");

    char input[MAX_INPUT_LEN];
    ac_message_t *messages = NULL;

    while (g_running) {
        printf("%sYou: %s", COLOR_PROMPT, COLOR_RESET);
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) {
            break;
        }

        /* Remove trailing newline */
        size_t len = strlen(input);
        if (len > 0 && input[len - 1] == '\n') {
            input[--len] = '\0';
        }

        /* Skip empty input */
        if (len == 0) {
            continue;
        }

        /* Handle commands */
        if (input[0] == '/') {
            if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
                break;
            } else if (strcmp(input, "/help") == 0) {
                print_usage();
                continue;
            } else if (strcmp(input, "/thinking") == 0) {
                g_thinking_mode = !g_thinking_mode;
                llm_params.thinking.enabled = g_thinking_mode;
                ac_llm_update_params(llm, &llm_params);
                printf("[Thinking mode: %s]\n", g_thinking_mode ? "ON" : "OFF");
                continue;
            } else if (strcmp(input, "/show") == 0) {
                g_show_thinking = !g_show_thinking;
                printf("[Show thinking: %s]\n", g_show_thinking ? "ON" : "OFF");
                continue;
            } else {
                printf("[Unknown command: %s]\n", input);
                continue;
            }
        }

        /* Add user message */
        ac_message_t *user_msg = ac_message_create(arena, AC_ROLE_USER, input);
        ac_message_append(&messages, user_msg);

        printf("%sAssistant: %s", COLOR_PROMPT, COLOR_RESET);
        fflush(stdout);

        /* Call LLM with streaming */
        ac_chat_response_t response = {0};
        arc_err_t err = ac_llm_chat_stream(llm, messages, NULL, stream_callback, NULL, &response);

        if (err != ARC_OK) {
            printf("[Error: %s]\n", ac_strerror(err));
        } else {
            /* Add assistant response to history */
            ac_message_t *assistant_msg = ac_message_from_response(arena, &response);
            if (assistant_msg) {
                ac_message_append(&messages, assistant_msg);
            }
            
            /* Show usage if available */
            if (response.output_tokens > 0) {
                printf("%s[tokens: %d]%s\n", COLOR_INFO, response.output_tokens, COLOR_RESET);
            }
        }

        ac_chat_response_free(&response);
        printf("\n");
    }

    /* Cleanup */
    ac_llm_cleanup(llm);
    arena_destroy(arena);

    printf("Goodbye!\n");
    return 0;
}
