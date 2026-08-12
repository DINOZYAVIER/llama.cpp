#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "base64.hpp"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <climits>
#include <clocale>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <unordered_set>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;
static std::mutex agl_attempt_mutex;
static std::string agl_active_attempt;
static std::atomic<bool> agl_cancel_requested = false;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// satisfies -Wmissing-declarations
int llama_server(int argc, char ** argv);

static int agl_private_inventory() {
#if defined(_WIN32)
    return -1;
#else
    const char * value = std::getenv("AGL_LLAMA_SERVER_INVENTORY_FD");
    if (value == nullptr) {
        return -1;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 3 || parsed > INT_MAX) {
        return 2;
    }
    common_init();
    llama_backend_init();
    ggml_backend_load_all();
    json devices = json::array();
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(ggml_backend_dev_get(index), &props);
        const char * kind = "unknown";
        switch (props.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   kind = "cpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   kind = "discrete_gpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  kind = "integrated_gpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: kind = "accelerator"; break;
            case GGML_BACKEND_DEVICE_TYPE_META:  kind = "metadata"; break;
        }
        devices.push_back({
            {"identity", props.name == nullptr ? "" : props.name},
            {"description", props.description == nullptr ? "" : props.description},
            {"native_device_id", props.device_id == nullptr ? "" : props.device_id},
            {"kind", kind},
            {"available_pool_bytes", props.memory_free},
            {"physical_pool_bytes", props.memory_total},
        });
    }
    const std::string payload = safe_json_to_str({
        {"schema", "agentlibre.llama-inventory/v1"},
        {"llama_cpp_commit", llama_commit()},
        {"devices", std::move(devices)},
    }) + "\n";
    size_t offset = 0;
    while (offset < payload.size()) {
        const ssize_t written = ::write(static_cast<int>(parsed), payload.data() + offset, payload.size() - offset);
        if (written <= 0) {
            llama_backend_free();
            return 3;
        }
        offset += static_cast<size_t>(written);
    }
    llama_backend_free();
    return 0;
#endif
}

static server_http_res_ptr agl_private_generate(
        const server_http_context::handler_t & generate,
        const server_http_req & req) {
    constexpr size_t AGL_MAX_MEDIA_BYTES = 64 * 1024 * 1024;
    constexpr size_t AGL_MAX_MEDIA_PARTS = 64;

    json body;
    if (req.files.empty()) {
        body = json::parse(req.body);
    } else {
        if (req.files.size() > AGL_MAX_MEDIA_PARTS) {
            throw std::invalid_argument("AGL media part count exceeds 64");
        }
        const json envelope = json::parse(req.body);
        if (!envelope.is_object() || envelope.size() != 1 || !envelope.contains("request") ||
            !envelope.at("request").is_string()) {
            throw std::invalid_argument("AGL multipart body must contain one request field");
        }
        body = json::parse(envelope.at("request").get<std::string>());
    }

    const std::string attempt_id = json_value(body, "agl_attempt_id", std::string());
    if (attempt_id.empty() || attempt_id.size() > 128) {
        throw std::invalid_argument("AGL generation omitted a bounded attempt identity");
    }
    body.erase("agl_attempt_id");
    {
        std::lock_guard<std::mutex> lock(agl_attempt_mutex);
        if (!agl_active_attempt.empty()) {
            throw std::invalid_argument("AGL private engine already has an active attempt");
        }
        agl_active_attempt = attempt_id;
        agl_cancel_requested.store(false);
    }
    struct attempt_guard {
        std::string id;
        ~attempt_guard() {
            std::lock_guard<std::mutex> lock(agl_attempt_mutex);
            if (agl_active_attempt == id) {
                agl_active_attempt.clear();
                agl_cancel_requested.store(false);
            }
        }
    } guard {attempt_id};

    auto & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("AGL messages must be an array");
    }
    std::unordered_set<std::string> used;
    size_t total_media_bytes = 0;
    for (auto & message : messages) {
        if (!message.is_object() || !message.contains("content")) {
            continue;
        }
        auto & content = message["content"];
        if (!content.is_array()) {
            continue;
        }
        for (auto & part : content) {
            if (!part.is_object()) {
                throw std::invalid_argument("AGL message content part must be an object");
            }
            const std::string type = json_value(part, "type", std::string());
            if (type == "text") {
                if (!part.contains("text") || !part.at("text").is_string()) {
                    throw std::invalid_argument("AGL text content part is invalid");
                }
                continue;
            }
            if (type != "agl_attachment") {
                throw std::invalid_argument("private AGL generation rejects URL and path media");
            }
            const std::string id = json_value(part, "id", std::string());
            const std::string media_type = json_value(part, "media_type", std::string());
            const uint64_t byte_length = json_value(part, "byte_length", UINT64_C(0));
            if (id.empty() || !used.insert(id).second) {
                throw std::invalid_argument("AGL attachment identity is empty or duplicated");
            }
            const auto found = req.files.find(id);
            if (found == req.files.end()) {
                throw std::invalid_argument("AGL attachment bytes are missing");
            }
            const auto & file = found->second;
            if ((media_type != "image/png" && media_type != "image/jpeg") ||
                file.content_type != media_type || file.filename != id ||
                byte_length != file.data.size()) {
                throw std::invalid_argument("AGL attachment metadata does not match multipart bytes");
            }
            if (file.data.size() > AGL_MAX_MEDIA_BYTES - total_media_bytes) {
                throw std::invalid_argument("AGL media bytes exceed 64 MiB");
            }
            total_media_bytes += file.data.size();
            const auto * data = reinterpret_cast<const char *>(file.data.data());
            part = {
                {"type", "image_url"},
                {"image_url", {
                    {"url", "data:" + media_type + ";base64," + base64::encode(data, file.data.size())},
                }},
            };
        }
    }
    if (used.size() != req.files.size()) {
        throw std::invalid_argument("AGL multipart contains an unreferenced file");
    }

    const std::function<bool()> should_stop = [&req]() {
        return agl_cancel_requested.load() || req.should_stop();
    };
    const server_http_req normalized {
        req.params,
        req.headers,
        req.path,
        req.query_string,
        body.dump(),
        {},
        should_stop,
    };
    return generate(normalized);
}

static server_http_res_ptr agl_private_control(const server_http_req & req) {
    const json body = json::parse(req.body);
    const std::string attempt_id = json_value(body, "attempt_id", std::string());
    const std::string action = json_value(body, "action", std::string());
    if (attempt_id.empty() || action != "cancel" || body.size() != 2) {
        throw std::invalid_argument("invalid AGL control request");
    }
    {
        std::lock_guard<std::mutex> lock(agl_attempt_mutex);
        if (agl_active_attempt != attempt_id) {
            throw std::invalid_argument("AGL control attempt is not active");
        }
        agl_cancel_requested.store(true);
    }
    auto response = std::make_unique<server_http_res>();
    response->data = safe_json_to_str({
        {"schema", "agentlibre.llama-cancel/v1"},
        {"attempt_id", attempt_id},
        {"acknowledged", true},
    });
    return response;
}

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (const int inventory = agl_private_inventory(); inventory >= 0) {
        return inventory;
    }

    // own arguments required by this example
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // router server never loads a model and must not touch the GPU
    // skip device enumeration so the CUDA primary context stays uncreated
    const bool is_router_server = params.model.path.empty();
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
    }

    if (ctx_http.is_private_inherited()) {
        if (is_router_server) {
            SRV_ERR("%s", "router mode is unavailable on the AGL inherited listener\n");
            return 1;
        }
        ctx_http.get ("/agl/v1/readiness",     ex_wrapper(routes.get_props));
        ctx_http.post("/agl/v1/generate",      ex_wrapper([&routes](const server_http_req & req) {
            return agl_private_generate(routes.post_chat_completions, req);
        }));
        ctx_http.post("/agl/v1/control",       ex_wrapper(agl_private_control));
        ctx_http.post("/agl/v1/slot/:id_slot", ex_wrapper(routes.post_slots));
    } else {
    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    // Supports both new ui_mcp_proxy and deprecated webui_mcp_proxy fields
    if (params.ui_mcp_proxy || params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    }
    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // load the model
        SRV_INF("%s", "loading model\n");

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        SRV_INF("router server is listening on %s\n", ctx_http.listening_address.c_str());
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            json model_info = routes.get_model_info();
            monitor_thread = server_models::setup_child_server(shutdown_handler, model_info);
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
