# LLM API v2 设计文档

## 概述

本文档描述 `ac_core` 中 LLM API 的 v2 版本设计，主要解决两个核心问题：

1. **思考识别 (Thinking)** - 支持思考模型的思考内容输出
2. **流式输出 (Streaming)** - 支持实时流式响应

## 设计目标

- 统一抽象层包含所有可能的字段，Provider 根据实际情况选用
- 支持多种 Provider：Anthropic、OpenAI (Chat Completions & Responses)、DeepSeek、Kimi 等
- 保持向后兼容简单场景
- 支持有状态和无状态两种模式

## 核心约束

根据 Anthropic 和 OpenAI 官方文档的要求：

1. **thinking 块必须原封不动传回** - 包括 `signature` 字段
2. **响应是多种内容块的序列** - thinking → text → tool_use 等
3. **不能重新排列或修改** - 必须保持原始顺序
4. **流式事件粒度** - thinking_delta、text_delta、signature_delta 等

---

## 数据结构设计

### 1. 内容块类型

```c
typedef enum {
    AC_BLOCK_TEXT,              // 普通文本内容
    AC_BLOCK_THINKING,          // 思考内容（含签名）
    AC_BLOCK_REDACTED_THINKING, // 被编辑的思考（加密）
    AC_BLOCK_REASONING,         // OpenAI 推理内容
    AC_BLOCK_TOOL_USE,          // 工具调用请求
    AC_BLOCK_TOOL_RESULT,       // 工具调用结果
} ac_block_type_t;
```

### 2. 内容块结构

```c
typedef struct ac_content_block {
    ac_block_type_t type;
    
    union {
        /* AC_BLOCK_TEXT */
        struct {
            char* text;
        } text;
        
        /* AC_BLOCK_THINKING (Anthropic) */
        struct {
            char* thinking;      // 思考内容
            char* signature;     // 必须原样保留的签名
        } thinking;
        
        /* AC_BLOCK_REDACTED_THINKING (Anthropic) */
        struct {
            char* data;          // 加密数据，必须原样保留
        } redacted;
        
        /* AC_BLOCK_REASONING (OpenAI) */
        struct {
            char* content;       // 推理内容（可能为空/加密）
            char* encrypted;     // 加密的推理内容
            char** summary;      // 摘要数组
            int summary_count;
        } reasoning;
        
        /* AC_BLOCK_TOOL_USE */
        struct {
            char* id;            // 工具调用 ID
            char* name;          // 函数名
            char* input;         // JSON 参数
        } tool_use;
        
        /* AC_BLOCK_TOOL_RESULT */
        struct {
            char* tool_use_id;   // 对应的工具调用 ID
            char* content;       // 结果内容
            int is_error;        // 是否是错误
        } tool_result;
    };
    
    struct ac_content_block* next;
} ac_content_block_t;
```

### 3. 消息结构

保持向后兼容，同时支持新的内容块模式：

```c
typedef struct ac_message {
    ac_role_t role;              // SYSTEM, USER, ASSISTANT
    
    /* 简单模式（向后兼容） */
    char* content;               // 简单文本内容
    
    /* 内容块模式（v2） */
    ac_content_block_t* blocks;  // 内容块链表
    
    /* 工具调用（保持兼容） */
    char* tool_call_id;          // TOOL 消息的调用 ID
    ac_tool_call_t* tool_calls;  // ASSISTANT 的工具调用（已弃用，使用 blocks）
    
    struct ac_message* next;
} ac_message_t;
```

### 4. 响应结构

```c
typedef struct {
    /* 响应 ID（用于有状态模式） */
    char* id;                    // 响应 ID
    
    /* 内容块（v2 核心） */
    ac_content_block_t* blocks;  // 完整的内容块序列
    int block_count;
    
    /* 向后兼容 */
    char* content;               // 合并的文本内容（便捷访问）
    ac_tool_call_t* tool_calls;  // 工具调用列表
    int tool_call_count;
    
    /* Token 使用统计 */
    int input_tokens;
    int output_tokens;
    int thinking_tokens;         // 思考 token（Anthropic）
    int reasoning_tokens;        // 推理 token（OpenAI）
    int cache_creation_tokens;
    int cache_read_tokens;
    
    /* 结束原因 */
    char* stop_reason;           // "end_turn", "tool_use", "max_tokens"
} ac_chat_response_t;
```

---

## LLM 参数设计

### 统一参数结构

```c
typedef struct {
    /*========== 基础配置 ==========*/
    const char* provider;        // Provider 名称
    const char* compatible;      // 兼容模式
    const char* model;           // 模型名称
    const char* api_key;         // API 密钥
    const char* api_base;        // API 基础 URL
    const char* instructions;    // 系统指令
    
    /*========== 生成参数 ==========*/
    float temperature;           // 采样温度
    float top_p;                 // 核采样
    int max_tokens;              // 最大生成 token
    int timeout_ms;              // 请求超时
    
    /*========== 思考/推理配置 ==========*/
    struct {
        int enabled;             // 是否启用
        int budget_tokens;       // 思考预算（最小 1024）
    } thinking;
    
    /*========== 有状态配置（OpenAI Responses） ==========*/
    struct {
        int store;               // 是否存储（启用有状态）
        const char* response_id; // 前一个响应 ID（链式调用）
        int include_encrypted;   // 是否包含加密推理
    } stateful;
    
    /*========== 流式配置 ==========*/
    int stream;                  // 是否启用流式
} ac_llm_params_t;
```

---

## 流式设计

### 事件类型

```c
typedef enum {
    AC_STREAM_MESSAGE_START,       // 消息开始
    AC_STREAM_CONTENT_BLOCK_START, // 内容块开始
    AC_STREAM_DELTA,               // 内容增量
    AC_STREAM_CONTENT_BLOCK_STOP,  // 内容块结束
    AC_STREAM_MESSAGE_DELTA,       // 消息级更新
    AC_STREAM_MESSAGE_STOP,        // 消息结束
    AC_STREAM_ERROR,               // 错误
} ac_stream_event_type_t;
```

### 增量类型

```c
typedef enum {
    AC_DELTA_THINKING,     // thinking_delta
    AC_DELTA_TEXT,         // text_delta
    AC_DELTA_INPUT_JSON,   // input_json_delta（工具参数）
    AC_DELTA_SIGNATURE,    // signature_delta
    AC_DELTA_REASONING,    // reasoning content delta
} ac_delta_type_t;
```

### 流式事件

```c
typedef struct {
    ac_stream_event_type_t type;
    
    /* 块信息 */
    int block_index;
    ac_block_type_t block_type;
    
    /* 增量内容 */
    ac_delta_type_t delta_type;
    const char* delta;
    size_t delta_len;
    
    /* 工具调用相关 */
    const char* tool_id;
    const char* tool_name;
    
    /* 消息级别更新 */
    const char* stop_reason;
    int output_tokens;
    
    /* 错误信息 */
    const char* error_type;
    const char* error_msg;
} ac_stream_event_t;
```

### 流式回调

```c
typedef int (*ac_stream_callback_t)(
    const ac_stream_event_t* event,
    void* user_data
);
```

---

## Provider 接口

### 操作接口

```c
typedef struct ac_llm_ops {
    const char* name;
    uint32_t capabilities;       // 能力位图
    
    /* 生命周期 */
    void* (*create)(const ac_llm_params_t* params);
    void (*cleanup)(void* priv);
    
    /* 同步接口 */
    arc_err_t (*chat)(
        void* priv,
        const ac_llm_params_t* params,
        const ac_message_t* messages,
        const char* tools,
        ac_chat_response_t* response
    );
    
    /* 流式接口 */
    arc_err_t (*chat_stream)(
        void* priv,
        const ac_llm_params_t* params,
        const ac_message_t* messages,
        const char* tools,
        ac_stream_callback_t callback,
        void* user_data,
        ac_chat_response_t* response
    );
} ac_llm_ops_t;
```

### 能力位图

```c
typedef enum {
    AC_LLM_CAP_THINKING     = (1 << 0),  // 支持思考
    AC_LLM_CAP_REASONING    = (1 << 1),  // 支持推理
    AC_LLM_CAP_STREAMING    = (1 << 2),  // 支持流式
    AC_LLM_CAP_STATEFUL     = (1 << 3),  // 支持有状态
    AC_LLM_CAP_TOOLS        = (1 << 4),  // 支持工具调用
    AC_LLM_CAP_VISION       = (1 << 5),  // 支持视觉
} ac_llm_capability_t;
```

---

## API 函数

### 核心 API

```c
/* 创建/销毁 */
ac_llm_t* ac_llm_create(arena_t* arena, const ac_llm_params_t* params);
void ac_llm_cleanup(ac_llm_t* llm);

/* 同步调用 */
arc_err_t ac_llm_chat(
    ac_llm_t* llm,
    const ac_message_t* messages,
    const char* tools,
    ac_chat_response_t* response
);

/* 流式调用 */
arc_err_t ac_llm_chat_stream(
    ac_llm_t* llm,
    const ac_message_t* messages,
    const char* tools,
    ac_stream_callback_t callback,
    void* user_data,
    ac_chat_response_t* response
);
```

### 便捷函数

```c
/* 响应访问 */
const char* ac_response_text(const ac_chat_response_t* resp);
const char* ac_response_thinking(const ac_chat_response_t* resp);
int ac_response_has_tool_calls(const ac_chat_response_t* resp);

/* 消息构建 */
ac_message_t* ac_message_create_text(arena_t* arena, ac_role_t role, const char* text);
ac_message_t* ac_message_from_response(arena_t* arena, const ac_chat_response_t* resp);

/* 内容块操作 */
ac_content_block_t* ac_block_create_text(arena_t* arena, const char* text);
ac_content_block_t* ac_block_create_thinking(arena_t* arena, const char* thinking, const char* signature);
ac_content_block_t* ac_block_create_tool_use(arena_t* arena, const char* id, const char* name, const char* input);
```

---

## Provider 实现要点

### Anthropic Provider

- 请求：添加 `thinking: { type: "enabled", budget_tokens: N }`
- 响应：解析 `content[]` 数组中的 `thinking`、`text`、`tool_use` 块
- 流式：处理 `thinking_delta`、`text_delta`、`signature_delta` 事件
- 多轮：将 thinking 块（含 signature）原样放入历史消息

### OpenAI Chat Completions Provider

- 思考：从 `content` 中解析 `<think>...</think>` 标签（DeepSeek）
- 工具：标准 `tool_calls` 处理
- 流式：标准 SSE 事件

### OpenAI Responses Provider

- 请求：使用 `input` 替代 `messages`，支持 `previous_response_id`
- 响应：解析 `output[]` 中的 `reasoning`、`message`、`function_call` 项
- 有状态：设置 `store: true` 并保存 `response.id`
- 加密推理：设置 `include: ["reasoning.encrypted_content"]`

---

## 迁移指南

### 从 v1 迁移

v1 代码：
```c
char* content = ac_llm_chat(llm, messages);
```

v2 代码（兼容模式）：
```c
ac_chat_response_t resp = {0};
ac_llm_chat(llm, messages, NULL, &resp);
const char* content = ac_response_text(&resp);
ac_chat_response_free(&resp);
```

### 使用思考功能

```c
ac_llm_params_t params = {
    .model = "claude-sonnet-4-5",
    .thinking = { .enabled = 1, .budget_tokens = 10000 },
    // ...
};

ac_chat_response_t resp = {0};
ac_llm_chat(llm, messages, NULL, &resp);

// 访问思考内容
const char* thinking = ac_response_thinking(&resp);
const char* text = ac_response_text(&resp);

// 构建历史消息（保留所有块）
ac_message_t* assistant = ac_message_from_response(arena, &resp);
```

### 使用流式输出

```c
int on_stream(const ac_stream_event_t* event, void* ctx) {
    if (event->type == AC_STREAM_DELTA) {
        switch (event->delta_type) {
            case AC_DELTA_THINKING:
                printf("[thinking] %.*s", (int)event->delta_len, event->delta);
                break;
            case AC_DELTA_TEXT:
                printf("%.*s", (int)event->delta_len, event->delta);
                break;
        }
    }
    return 0;  // 继续，非零中止
}

ac_llm_chat_stream(llm, messages, NULL, on_stream, NULL, &final_resp);
```

---

## 实现计划

| 阶段 | 任务 | 状态 |
|------|------|------|
| P0 | 定义新数据结构 (`ac_content_block_t` 等) | ✅ 已完成 |
| P0 | 更新 `ac_llm_params_t` | ✅ 已完成 |
| P0 | 实现 Anthropic Provider 思考支持 | ✅ 已完成 |
| P1 | 实现流式事件结构 | ✅ 已完成（定义） |
| P1 | 实现 SSE 解析器 | 待实现 |
| P1 | 实现流式回调机制 | ✅ 已完成（接口） |
| P2 | OpenAI Provider 流式支持 | 待实现 |
| P2 | OpenAI Responses Provider | 待实现 |
| P3 | DeepSeek `<think>` 标签解析 | 待实现 |

---

## 参考资料

- [Anthropic Extended Thinking](https://platform.claude.com/docs/zh-CN/build-with-claude/extended-thinking)
- [OpenAI Responses API Migration](https://platform.openai.com/docs/guides/migrate-to-responses)
- [Kimi K2 Thinking Model](https://platform.moonshot.cn/docs/guide/use-kimi-k2-thinking-model)
