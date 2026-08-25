const std = @import("std");
const image_attachments = @import("../core/images/image_attachments.zig");
const stream_provider = @import("../core/agent/stream_provider.zig");
const model_tool_schema = @import("../core/tooling/model_tool_schema.zig");
const gateway_client = @import("client.zig");
const io_mod = @import("../core/shared/io.zig");
const types = @import("../core/shared/types.zig");
const gateway_provider = @import("../core/gateway/gateway_provider.zig");
const model_catalog = @import("../core/gateway/model_catalog.zig");
const collections = @import("../core/shared/collections.zig");

const Allocator = std.mem.Allocator;

const env_base_url = "FX_OPENAI_BASE_URL";
const base_url_env_fallback = "OPENAI_BASE_URL";
const env_api_key = "FX_OPENAI_API_KEY";
const api_key_env_fallback = "OPENAI_API_KEY";

const chat_completions_path = "/chat/completions";
const models_path = "/models";

const max_sse_line_bytes: usize = 1024 * 1024;
const max_sse_aggregate_bytes: usize = 64 * 1024 * 1024;
const max_error_body_bytes: usize = 256 * 1024;
const transfer_buffer_bytes: usize = 256 * 1024;
const connect_timeout_ms: i64 = 30_000;
const max_tool_calls: usize = 128;
const max_tool_arguments_bytes: usize = 4 * 1024 * 1024;

pub const agent_stream_provider = stream_provider.Provider{
    .stream_fn = streamCompletion,
};

pub fn configuredBaseUrl() ?[]const u8 {
    return baseUrlOverride(io_mod.getenv(env_base_url)) orelse
        baseUrlOverride(io_mod.getenv(base_url_env_fallback));
}

fn baseUrlOverride(url: ?[]const u8) ?[]const u8 {
    const value = url orelse return null;
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    return if (trimmed.len > 0) trimmed else null;
}

fn requiredBaseUrl() ![]const u8 {
    return configuredBaseUrl() orelse error.MissingOpenAiCompatBaseUrl;
}

pub fn configuredApiKey() ?[]const u8 {
    return apiKeyOverride(io_mod.getenv(env_api_key)) orelse
        apiKeyOverride(io_mod.getenv(api_key_env_fallback));
}

fn apiKeyOverride(url: ?[]const u8) ?[]const u8 {
    const value = url orelse return null;
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    return if (trimmed.len > 0) trimmed else null;
}

fn joinEndpoint(base: []const u8, path: []const u8) ![]u8 {
    const trimmed = std.mem.trimEnd(u8, base, "/");
    return std.fmt.allocPrint(std.heap.c_allocator, "{s}{s}", .{ trimmed, path });
}

pub fn chatCompletionsUrl() ![]u8 {
    return joinEndpoint(try requiredBaseUrl(), chat_completions_path);
}

fn modelsUrl() ![]u8 {
    return joinEndpoint(try requiredBaseUrl(), models_path);
}

fn validateModel(model: []const u8) !void {
    if (model.len == 0 or model.len > 256) return error.InvalidOpenAiCompatModel;
    for (model) |byte| {
        if (byte <= 0x20 or byte == 0x7f) return error.InvalidOpenAiCompatModel;
    }
}

pub fn buildRequest(
    alloc: Allocator,
    request: stream_provider.RequestData,
) ![]u8 {
    try validateModel(request.model);
    if (request.budget) |budget| {
        if (budget.cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
        _ = budget.deadline;
    }

    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\"model\":");
    try std.json.Stringify.value(request.model, .{}, writer);
    try writer.writeAll(",\"stream\":true");

    const has_tools = request.tools.advertised_names.len > 0 or
        request.tools.additional_functions.len > 0 or
        request.tools.selected_dynamic.len > 0;
    if (has_tools) {
        try writer.writeAll(",\"tools\":");
        try writeToolsArray(alloc, writer, request.tools);
    }

    try writer.writeAll(",\"tool_choice\":");
    if (!has_tools and request.tool_choice == .none) {
        try writer.writeAll("\"none\"");
    } else if (has_tools) {
        try std.json.Stringify.value(request.tool_choice.label(), .{}, writer);
    } else {
        try writer.writeAll("\"auto\"");
    }

    try writer.writeAll(",\"messages\":");
    try buildMessages(alloc, writer, request.messages, request.verified_images);

    if (request.provider_options.reasoning) |effort| {
        if (!effort.isDefault()) {
            try writer.writeAll(",\"reasoning_effort\":");
            try std.json.Stringify.value(effort.label(), .{}, writer);
        }
    }
    if (request.max_output_tokens) |limit| {
        try writer.writeAll(",\"max_tokens\":");
        try writer.print("{d}", .{limit});
    }
    if (request.response_format) |format| {
        if (format.schema != .object) return error.InvalidStructuredResponseSchema;
        try writer.writeAll(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":");
        try std.json.Stringify.value(format.name, .{}, writer);
        try writer.writeAll(",\"description\":");
        try std.json.Stringify.value(format.description, .{}, writer);
        try writer.writeAll(",\"schema\":");
        try std.json.Stringify.value(format.schema, .{}, writer);
        try writer.writeAll(",\"strict\":true}}");
    }

    try writer.writeByte('}');
    return out.toOwnedSlice();
}

fn buildMessages(
    alloc: Allocator,
    writer: *std.Io.Writer,
    messages: []const types.ChatMessage,
    verified_images: ?[]const image_attachments.VerifiedSnapshot,
) !void {
    try writer.writeByte('[');
    for (messages, 0..) |message, index| {
        if (index > 0) try writer.writeByte(',');
        switch (message.role) {
            .system => {
                try writer.writeAll("{\"role\":\"system\",\"content\":");
                try std.json.Stringify.value(message.content orelse "", .{}, writer);
                try writer.writeByte('}');
            },
            .user => {
                const attach_images = verified_images != null and index == messages.len - 1;
                if (!attach_images) {
                    try writer.writeAll("{\"role\":\"user\",\"content\":");
                    try std.json.Stringify.value(message.content orelse "", .{}, writer);
                    try writer.writeByte('}');
                    continue;
                }
                try writer.writeAll("{\"role\":\"user\",\"content\":[");
                var parts: usize = 0;
                if (message.content) |content| {
                    if (content.len > 0) {
                        try writer.writeAll("{\"type\":\"text\",\"text\":");
                        try std.json.Stringify.value(content, .{}, writer);
                        try writer.writeByte('}');
                        parts += 1;
                    }
                }
                for (verified_images.?) |image| {
                    if (parts > 0) try writer.writeByte(',');
                    try writeSnapshotImagePart(writer, alloc, image);
                    parts += 1;
                }
                try writer.writeByte(']');
                try writer.writeByte('}');
            },
            .assistant => {
                try writer.writeAll("{\"role\":\"assistant\",\"content\":");
                try std.json.Stringify.value(message.content orelse null, .{}, writer);
                if (message.tool_calls.len > 0) {
                    try writer.writeAll(",\"tool_calls\":[");
                    for (message.tool_calls, 0..) |call, call_index| {
                        if (call_index > 0) try writer.writeByte(',');
                        try writer.writeAll("{\"id\":");
                        try std.json.Stringify.value(call.id, .{}, writer);
                        try writer.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
                        try std.json.Stringify.value(call.name, .{}, writer);
                        try writer.writeAll(",\"arguments\":");
                        try std.json.Stringify.value(call.arguments_json, .{}, writer);
                        try writer.writeAll("}}");
                    }
                    try writer.writeByte(']');
                }
                try writer.writeByte('}');
            },
            .tool => {
                try writer.writeAll("{\"role\":\"tool\",\"tool_call_id\":");
                try std.json.Stringify.value(message.tool_call_id orelse "", .{}, writer);
                try writer.writeAll(",\"content\":");
                try std.json.Stringify.value(message.content orelse "", .{}, writer);
                try writer.writeByte('}');
            },
        }
    }
    try writer.writeByte(']');
}

fn writeSnapshotImagePart(
    writer: *std.Io.Writer,
    alloc: Allocator,
    image: image_attachments.VerifiedSnapshot,
) !void {
    const encoded_len = std.base64.standard.Encoder.calcSize(image.bytes.len);
    const encoded = try alloc.alloc(u8, encoded_len);
    defer alloc.free(encoded);
    _ = std.base64.standard.Encoder.encode(encoded, image.bytes);
    try writer.writeAll("{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:");
    try writer.writeAll(image.media_type);
    try writer.writeAll(";base64,");
    try writer.writeAll(encoded);
    try writer.writeAll("\"}}");
}

fn writeToolsArray(
    alloc: Allocator,
    writer: *std.Io.Writer,
    tools: stream_provider.ToolSelection,
) !void {
    try writer.writeByte('[');
    var first = true;
    for (tools.advertised_names) |name| {
        const function = tools.advertisedFunction(name) orelse continue;
        if (!first) try writer.writeByte(',');
        first = false;
        try writeFunctionTool(alloc, writer, function);
    }
    for (tools.additional_functions) |function| {
        if (toolNameSelected(tools.advertised_names, function.name)) continue;
        if (!first) try writer.writeByte(',');
        first = false;
        try writeFunctionTool(alloc, writer, function);
    }
    for (tools.selected_dynamic) |tool| {
        if (toolNameSelected(tools.advertised_names, tool.name)) continue;
        if (!first) try writer.writeByte(',');
        first = false;
        try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
        try std.json.Stringify.value(tool.name, .{}, writer);
        try writer.writeAll(",\"description\":");
        try std.json.Stringify.value(tool.description, .{}, writer);
        try writer.writeAll(",\"parameters\":");
        try std.json.Stringify.value(tool.input_schema, .{}, writer);
        try writer.writeAll("}}");
    }
    try writer.writeByte(']');
}

fn writeFunctionTool(
    alloc: Allocator,
    writer: *std.Io.Writer,
    function: model_tool_schema.FunctionSchema,
) !void {
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
    try std.json.Stringify.value(function.name, .{}, writer);
    try writer.writeAll(",\"description\":");
    try model_tool_schema.writeCappedDescriptionJsonString(alloc, writer, function.description);
    try writer.writeAll(",\"parameters\":");
    try model_tool_schema.writeObjectSchema(alloc, writer, function.input_schema);
    try writer.writeAll("}}");
}

fn toolNameSelected(names: []const []const u8, expected: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, expected)) return true;
    return false;
}

fn streamCompletion(
    _: ?*anyopaque,
    alloc: Allocator,
    request: stream_provider.ModelRequest,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    const payload = try buildRequest(alloc, request.data());
    defer alloc.free(payload);
    var result = streamPrepared(alloc, request, payload) catch |err| {
        if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
        if (requestDeadlineExpired(request)) return error.Timeout;
        request.attempt_evidence.network_failure = gateway_client.networkFailureEvidence(err, request.delivery.load());
        return err;
    };
    if (requestDeadlineExpired(request)) {
        result.deinit(alloc);
        return error.Timeout;
    }
    return result;
}

fn requestDeadlineExpired(request: stream_provider.ModelRequest) bool {
    const deadline = request.deadline orelse return false;
    const now = std.Io.Clock.Timestamp.now(io_mod.getIo(), .awake);
    return !std.Io.Clock.Timestamp.compare(now, .lt, deadline);
}

const OpenedRequest = struct {
    request: ?std.http.Client.Request,

    pub fn deinit(self: *OpenedRequest, _: Allocator) void {
        if (self.request) |*request| request.deinit();
        self.request = null;
    }

    pub fn take(self: *OpenedRequest) std.http.Client.Request {
        const request = self.request.?;
        self.request = null;
        return request;
    }
};

const OpenRequestOperation = struct {
    client: *std.http.Client,
    uri: std.Uri,
    auth_header: ?[]const u8,

    pub fn run(self: *@This()) !OpenedRequest {
        return .{ .request = try self.client.request(.POST, self.uri, .{
            .headers = .{
                .content_type = .{ .override = "application/json" },
                .authorization = if (self.auth_header) |value|
                    .{ .override = value }
                else
                    .default,
                .accept_encoding = .omit,
                .user_agent = .{ .override = gateway_client.user_agent },
            },
            .extra_headers = &.{.{ .name = "accept", .value = "text/event-stream" }},
            .keep_alive = false,
            .redirect_behavior = .unhandled,
        }) };
    }
};

pub fn streamPrepared(
    alloc: Allocator,
    request: stream_provider.ModelRequest,
    payload: []const u8,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    const url = try chatCompletionsUrl();
    defer alloc.free(url);
    const uri = try std.Uri.parse(url);

    var auth_header: ?[]u8 = null;
    defer if (auth_header) |value| alloc.free(value);
    if (configuredApiKey()) |key| {
        auth_header = try std.fmt.allocPrint(alloc, "Bearer {s}", .{key});
    }

    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();
    var open_operation = OpenRequestOperation{
        .client = &client,
        .uri = uri,
        .auth_header = if (auth_header) |value| value else null,
    };
    var connect_deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromMilliseconds(connect_timeout_ms),
    });
    if (request.deadline) |deadline| {
        if (std.Io.Clock.Timestamp.compare(deadline, .lt, connect_deadline)) {
            connect_deadline = deadline;
        }
    }
    try request.admission.admit();
    var opened = try gateway_client.runBoundedHttpOperation(
        OpenedRequest,
        alloc,
        request.cancel_flag,
        connect_deadline,
        &open_operation,
    );
    var http_request = opened.take();
    defer http_request.deinit();
    var cancel_watch_done = std.atomic.Value(bool).init(false);
    const cancel_watcher = if (http_request.connection) |connection| blk: {
        if (request.deadline) |deadline|
            break :blk try gateway_client.spawnHttpCancelWatcherBounded(
                &cancel_watch_done,
                request.cancel_flag,
                deadline,
                connection.stream_writer.stream,
            );
        break :blk try gateway_client.spawnHttpCancelWatcher(
            &cancel_watch_done,
            request.cancel_flag,
            connection.stream_writer.stream,
        );
    } else null;
    defer {
        cancel_watch_done.store(true, .seq_cst);
        if (cancel_watcher) |thread| thread.join();
    }
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    http_request.transfer_encoding = .{ .content_length = payload.len };
    var send_buffer: [8192]u8 = undefined;
    request.delivery.markPossiblySent();
    var body_writer = try http_request.sendBodyUnflushed(&send_buffer);
    try body_writer.writer.writeAll(payload);
    try body_writer.end();
    if (http_request.connection) |connection| try connection.flush();
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    var response = try http_request.receiveHead(&.{});
    if (response.head.status != .ok) {
        var transfer: [16 * 1024]u8 = undefined;
        const reader = response.reader(&transfer);
        const bounded_body = reader.allocRemaining(alloc, .limited(max_error_body_bytes + 1)) catch |err| switch (err) {
            error.StreamTooLong => try alloc.dupe(u8, "OpenAI-compatible endpoint returned an oversized error body"),
            else => return err,
        };
        const body = if (bounded_body.len > max_error_body_bytes) body: {
            alloc.free(bounded_body);
            break :body try alloc.dupe(u8, "OpenAI-compatible endpoint returned an oversized error body");
        } else bounded_body;
        return .{ .failed = .{
            .kind = failureKind(response.head.status),
            .detail = body,
            .ownership = .owned,
        } };
    }

    var transfer_buffer: [transfer_buffer_bytes]u8 = undefined;
    const reader = response.reader(&transfer_buffer);
    var events = request.events;
    const completion = try consumeSse(
        alloc,
        reader,
        &events,
        request.cancel_flag,
        request.content_capture_limit,
    );
    return .{ .completed = .{
        .completion = completion,
        .usage = .{ .immediate = null },
        .ownership = .owned,
    } };
}

const SseReader = struct {
    pending_line: std.ArrayList(u8) = .empty,
    aggregate_bytes: usize = 0,

    const Line = struct {
        bytes: []const u8,
        wire_bytes: usize,
    };

    fn deinit(self: *SseReader, alloc: Allocator) void {
        self.pending_line.deinit(alloc);
    }

    fn release(self: *SseReader) void {
        self.pending_line.clearRetainingCapacity();
    }

    fn next(self: *SseReader, alloc: Allocator, reader: anytype) !?[]const u8 {
        while (true) {
            const line = try self.readLine(alloc, reader) orelse return null;
            self.aggregate_bytes += line.wire_bytes;
            if (self.aggregate_bytes > max_sse_aggregate_bytes) return error.OpenAiCompatResourceLimitExceeded;
            const trimmed = std.mem.trim(u8, line.bytes, " \t\r");
            if (trimmed.len == 0 or trimmed[0] == ':') {
                self.release();
                continue;
            }
            if (!std.mem.startsWith(u8, trimmed, "data:")) {
                self.release();
                continue;
            }
            const data = std.mem.trim(u8, trimmed["data:".len..], " \t");
            if (std.mem.eql(u8, data, "[DONE]")) return null;
            return data;
        }
    }

    fn readLine(self: *SseReader, alloc: Allocator, reader: anytype) !?Line {
        while (true) {
            const fragment = reader.takeDelimiter('\n') catch |err| switch (err) {
                error.StreamTooLong => {
                    const buffered = reader.buffered();
                    if (buffered.len == 0) return error.OpenAiCompatSseReadStalled;
                    if (buffered.len > max_sse_line_bytes - self.pending_line.items.len) {
                        return error.OpenAiCompatSseEventTooLarge;
                    }
                    try self.pending_line.appendSlice(alloc, buffered);
                    reader.tossBuffered();
                    continue;
                },
                error.ReadFailed => return error.ReadFailed,
            } orelse {
                if (self.pending_line.items.len > 0) {
                    return .{
                        .bytes = self.pending_line.items,
                        .wire_bytes = self.pending_line.items.len,
                    };
                }
                return null;
            };
            if (fragment.len > max_sse_line_bytes - self.pending_line.items.len) {
                return error.OpenAiCompatSseEventTooLarge;
            }
            if (self.pending_line.items.len == 0) {
                return .{
                    .bytes = fragment,
                    .wire_bytes = fragment.len + 1,
                };
            }
            try self.pending_line.appendSlice(alloc, fragment);
            return .{
                .bytes = self.pending_line.items,
                .wire_bytes = self.pending_line.items.len + 1,
            };
        }
    }
};

const PendingToolCall = struct {
    id: std.ArrayList(u8) = .empty,
    name: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    started: bool = false,
};

const StreamAccumulator = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    tool_calls: std.ArrayList(PendingToolCall) = .empty,
    finish_reason: ?types.ProviderFinishReason = null,
    input_tokens: ?u64 = null,
    output_tokens: ?u64 = null,
    saw_error: bool = false,
    content_full: bool = false,

    fn deinit(self: *StreamAccumulator, alloc: Allocator) void {
        self.content.deinit(alloc);
        self.reasoning.deinit(alloc);
        for (self.tool_calls.items) |*call| {
            call.id.deinit(alloc);
            call.name.deinit(alloc);
            call.arguments.deinit(alloc);
        }
        self.tool_calls.deinit(alloc);
    }

    fn toolAt(self: *StreamAccumulator, alloc: Allocator, index: usize) !*PendingToolCall {
        while (self.tool_calls.items.len <= index) {
            try self.tool_calls.append(alloc, .{});
        }
        return &self.tool_calls.items[index];
    }
};

fn consumeSse(
    alloc: Allocator,
    reader: anytype,
    events: *stream_provider.EventSink,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) !types.ModelCompletion {
    var acc: StreamAccumulator = .{};
    defer acc.deinit(alloc);
    var sse: SseReader = .{};
    defer sse.deinit(alloc);

    while (try sse.next(alloc, reader)) |json_text| {
        defer sse.release();
        if (cancel_flag.load(.seq_cst)) return error.Cancelled;
        try applyJson(alloc, json_text, events, &acc, content_capture_limit);
        if (acc.saw_error) return error.OpenAiCompatStreamError;
    }

    return finalize(alloc, &acc);
}

fn applyJson(
    alloc: Allocator,
    json_text: []const u8,
    events: *stream_provider.EventSink,
    acc: *StreamAccumulator,
    content_capture_limit: ?usize,
) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, json_text, .{}) catch
        return error.InvalidOpenAiCompatSseEvent;
    defer parsed.deinit();
    const value = parsed.value;
    if (value != .object) return;
    const object = value.object;

    if (object.get("error") != null) {
        acc.saw_error = true;
        return;
    }

    if (object.get("usage")) |usage_value| {
        if (usage_value == .object) {
            const usage = usage_value.object;
            if (usage.get("prompt_tokens")) |t| {
                if (t == .integer) acc.input_tokens = @intCast(t.integer);
            }
            if (usage.get("completion_tokens")) |t| {
                if (t == .integer) acc.output_tokens = @intCast(t.integer);
            }
        }
    }

    const choices = object.get("choices") orelse return;
    if (choices != .array or choices.array.items.len == 0) return;
    const choice = choices.array.items[0];
    if (choice != .object) return;

    if (choice.object.get("finish_reason")) |reason| {
        if (reason == .string) {
            acc.finish_reason = types.ProviderFinishReason.parse_legacy(reason.string) orelse .other;
        } else if (reason == .null) {
            acc.finish_reason = .stop;
        }
    }

    const delta = choice.object.get("delta") orelse return;
    if (delta != .object) return;
    const delta_object = delta.object;

    if (delta_object.get("content")) |content| {
        if (content == .string and !acc.content_full) {
            const chunk = content.string;
            const remaining = if (content_capture_limit) |limit|
                limit - @min(acc.content.items.len, limit)
            else
                chunk.len;
            if (remaining == 0) {
                acc.content_full = true;
            } else {
                const take = @min(chunk.len, remaining);
                acc.content.appendSlice(alloc, chunk[0..take]) catch return error.OutOfMemory;
                events.emit(.{ .content_delta = chunk[0..take] });
                if (take < chunk.len) acc.content_full = true;
            }
        }
    }
    if (delta_object.get("reasoning")) |content| {
        if (content == .string) {
            const chunk = content.string;
            acc.reasoning.appendSlice(alloc, chunk) catch return error.OutOfMemory;
            events.emit(.{ .reasoning_delta = chunk });
        }
    } else if (delta_object.get("reasoning_content")) |content| {
        if (content == .string) {
            const chunk = content.string;
            acc.reasoning.appendSlice(alloc, chunk) catch return error.OutOfMemory;
            events.emit(.{ .reasoning_delta = chunk });
        }
    }

    if (delta_object.get("tool_calls")) |tool_calls| {
        if (tool_calls != .array) return;
        for (tool_calls.array.items) |tool_call| {
            if (tool_call != .object) continue;
            const call_object = tool_call.object;
            const index = indexFromValue(call_object.get("index")) orelse continue;
            const call = try acc.toolAt(alloc, index);
            if (call_object.get("id")) |id_value| {
                if (id_value == .string and call.id.items.len == 0) {
                    call.id.appendSlice(alloc, id_value.string) catch return error.OutOfMemory;
                }
            }
            const function = call_object.get("function") orelse continue;
            if (function != .object) continue;
            const function_object = function.object;
            if (function_object.get("name")) |name_value| {
                if (name_value == .string and call.name.items.len == 0) {
                    call.name.appendSlice(alloc, name_value.string) catch return error.OutOfMemory;
                }
            }
            if (function_object.get("arguments")) |args_value| {
                if (args_value == .string) {
                    const chunk = args_value.string;
                    call.arguments.appendSlice(alloc, chunk) catch return error.OutOfMemory;
                    if (chunk.len > 0) events.emit(.{ .tool_input_delta = chunk });
                }
            }
            if (!call.started and call.name.items.len > 0) {
                call.started = true;
                events.emit(.{ .tool_started = .{ .id = call.id.items, .name = call.name.items, .label = null } });
            }
        }
    }
}

fn indexFromValue(value: ?std.json.Value) ?usize {
    const resolved = value orelse return null;
    if (resolved != .integer) return null;
    const n = resolved.integer;
    if (n < 0 or n > 4096) return null;
    return @intCast(n);
}

fn finalize(
    alloc: Allocator,
    acc: *StreamAccumulator,
) !types.ModelCompletion {
    const tool_calls = try buildToolCalls(alloc, acc);
    errdefer freeToolCallSlice(alloc, tool_calls);
    const content = if (acc.content.items.len > 0)
        try alloc.dupe(u8, acc.content.items)
    else
        null;
    var completion = types.ModelCompletion{
        .content = content,
        .tool_calls = tool_calls,
        .finish_reason = acc.finish_reason,
        .usage = .{
            .input_tokens = acc.input_tokens,
            .output_tokens = acc.output_tokens,
        },
    };
    _ = &completion;
    return completion;
}

fn buildToolCalls(alloc: Allocator, acc: *StreamAccumulator) ![]types.ToolCall {
    var result: std.ArrayList(types.ToolCall) = .empty;
    errdefer {
        for (result.items) |call| {
            alloc.free(@constCast(call.id));
            alloc.free(@constCast(call.name));
            alloc.free(@constCast(call.arguments_json));
        }
        result.deinit(alloc);
    }
    for (acc.tool_calls.items) |*call| {
        if (result.items.len >= max_tool_calls) break;
        if (call.name.items.len == 0) continue;
        if (call.arguments.items.len > max_tool_arguments_bytes) return error.OpenAiCompatToolArgumentsTooLarge;
        const id = if (call.id.items.len > 0)
            try alloc.dupe(u8, call.id.items)
        else
            try std.fmt.allocPrint(alloc, "call_{d}", .{result.items.len});
        errdefer alloc.free(id);
        const name = try alloc.dupe(u8, call.name.items);
        errdefer alloc.free(name);
        const arguments_json = try alloc.dupe(u8, call.arguments.items);
        errdefer alloc.free(arguments_json);
        try result.append(alloc, .{
            .id = id,
            .name = name,
            .arguments_json = arguments_json,
            .argument_integrity = try types.ToolArgumentIntegrity.classifySerialized(alloc, arguments_json),
        });
    }
    return result.toOwnedSlice(alloc);
}

fn freeToolCallSlice(alloc: Allocator, calls: []types.ToolCall) void {
    for (calls) |call| {
        alloc.free(@constCast(call.id));
        alloc.free(@constCast(call.name));
        alloc.free(@constCast(call.arguments_json));
    }
    alloc.free(calls);
}

fn failureKind(status: std.http.Status) stream_provider.FailureKind {
    return switch (status) {
        .bad_request => .invalid_request,
        .unauthorized => .unauthorized,
        .forbidden => .forbidden,
        .payload_too_large => .request_too_large,
        .too_many_requests => .rate_limited,
        .internal_server_error => .server_error,
        .bad_gateway => .bad_gateway,
        .service_unavailable => .unavailable,
        .gateway_timeout => .gateway_timeout,
        else => .provider_error,
    };
}

pub const cli_model_catalog_provider = gateway_provider.CliModelCatalogProvider{
    .fetch_fn = fetchCliModelCatalog,
};

pub const model_catalog_provider = model_catalog.Provider{
    .fetch_fn = fetchModelCatalog,
};

fn fetchModelsJson(alloc: Allocator) !gateway_client.GetResult {
    const url = try modelsUrl();
    defer alloc.free(url);
    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();

    var auth_header: ?[]u8 = null;
    defer if (auth_header) |value| alloc.free(value);
    if (configuredApiKey()) |key| {
        auth_header = try std.fmt.allocPrint(alloc, "Bearer {s}", .{key});
    }

    var headers: std.http.Client.Request.Headers = .{};
    headers.user_agent = .{ .override = gateway_client.user_agent };
    headers.accept_encoding = .omit;
    if (auth_header) |value| {
        headers.authorization = .{ .override = value };
    }

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    const result = try client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .headers = headers,
        .response_writer = &out.writer,
        .redirect_behavior = .unhandled,
    });
    return .{ .status = result.status, .body = try out.toOwnedSlice() };
}

fn fetchCliModelCatalog(
    _: ?*anyopaque,
    alloc: Allocator,
    input: gateway_provider.CliModelCatalogInput,
) gateway_provider.CliModelCatalogResult {
    const access = model_catalog.AccessMetadata.init(input.access);
    var result = fetchModelsJson(alloc) catch {
        return .{ .failure = .{
            .access = access,
            .anonymous_fallback_used = false,
            .failure = .{ .category = .transport },
        } };
    };
    defer result.deinit(alloc);
    if (result.status != .ok) {
        return .{ .failure = .{
            .access = access,
            .anonymous_fallback_used = false,
            .failure = model_catalog.failureForHttpStatus(result.status),
        } };
    }
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, result.body, .{}) catch {
        return .{ .failure = .{
            .access = access,
            .anonymous_fallback_used = false,
            .failure = .{ .category = .malformed_response },
        } };
    };
    defer parsed.deinit();
    var ids: std.ArrayList([]u8) = .empty;
    errdefer collections.freeStringList(alloc, &ids);
    const data_list = modelData(parsed.value) orelse return .{ .failure = .{
        .access = access,
        .anonymous_fallback_used = false,
        .failure = .{ .category = .malformed_response },
    } };
    for (data_list) |item| {
        const id_value = item.object.get("id") orelse continue;
        if (id_value != .string) continue;
        const owned = alloc.dupe(u8, id_value.string) catch batch: {
            collections.freeStringList(alloc, &ids);
            break :batch null;
        };
        if (owned) |value| {
            ids.append(alloc, value) catch {
                alloc.free(value);
                collections.freeStringList(alloc, &ids);
                return .{ .failure = .{
                    .access = access,
                    .anonymous_fallback_used = false,
                    .failure = .{ .category = .resource_exhausted },
                } };
            };
        } else {
            return .{ .failure = .{
                .access = access,
                .anonymous_fallback_used = false,
                .failure = .{ .category = .resource_exhausted },
            } };
        }
    }
    return .{ .loaded = .{
        .ids = ids,
        .provenance = .{ .access = access },
    } };
}

fn fetchModelCatalog(
    _: ?*anyopaque,
    alloc: Allocator,
    _: model_catalog.FetchInput,
) Allocator.Error!model_catalog.ProviderResult {
    var result = fetchModelsJson(alloc) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .failure = .{ .category = .transport } },
    };
    defer result.deinit(alloc);
    if (result.status != .ok) return .{ .failure = model_catalog.failureForHttpStatus(result.status) };

    var parsed = std.json.parseFromSlice(std.json.Value, alloc, result.body, .{}) catch
        return .{ .failure = .{ .category = .malformed_response } };
    defer parsed.deinit();
    const data_list = modelData(parsed.value) orelse
        return .{ .failure = .{ .category = .malformed_response } };

    var entries: std.ArrayList(model_catalog.ModelCatalogEntry) = .empty;
    errdefer model_catalog.freeModelCatalog(alloc, &entries);
    for (data_list) |item| {
        const id_value = item.object.get("id") orelse continue;
        if (id_value != .string) continue;
        const id = try alloc.dupe(u8, id_value.string);
        errdefer alloc.free(id);
        const model_type = try alloc.dupe(u8, "language");
        errdefer alloc.free(model_type);
        try entries.append(alloc, .{
            .id = id,
            .model_type = model_type,
            .has_tool_use = true,
            .has_vision = true,
            .has_file_input = true,
        });
    }
    return .{ .catalog = entries };
}

fn modelData(value: std.json.Value) ?[]const std.json.Value {
    if (value != .object) return null;
    const data = value.object.get("data") orelse return null;
    if (data != .array) return null;
    for (data.array.items) |item| {
        if (item != .object) return null;
    }
    return data.array.items;
}

test "request serializes chat completions with tools and output limit" {
    const read_file_schema = model_tool_schema.FunctionSchema{
        .name = "read_file",
        .description = "Read a file",
        .input_schema = .{},
    };
    const messages = [_]types.ChatMessage{
        .{ .role = .system, .content = "Be concise." },
        .{ .role = .user, .content = "Read it." },
    };
    const body = try buildRequest(std.testing.allocator, .{
        .model = "openai/gpt-5.6",
        .messages = &messages,
        .tools = .{ .additional_functions = &.{read_file_schema} },
        .tool_choice = .auto,
        .provider_options = .{},
        .max_output_tokens = 4096,
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.find(u8, body, "\"model\":\"openai/gpt-5.6\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"stream\":true") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"type\":\"function\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"name\":\"read_file\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"tool_choice\":\"auto\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"max_tokens\":4096") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"role\":\"system\"") != null);
}

test "request serializes user content with attached snapshot image" {
    var image_bytes = [_]u8{ 1, 2, 3, 4, 5 };
    const snapshot = image_attachments.VerifiedSnapshot{
        .bytes = &image_bytes,
        .media_type = "image/png",
    };
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "what is this?" },
    };
    const body = try buildRequest(std.testing.allocator, .{
        .model = "model",
        .messages = &messages,
        .verified_images = &.{snapshot},
        .tool_choice = .auto,
        .provider_options = .{},
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.find(u8, body, "\"type\":\"image_url\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"type\":\"text\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "data:image/png;base64,") != null);
}

test "SSE reducer accumulates text reasoning tool calls and usage" {
    const Capture = struct {
        content: std.ArrayList(u8) = .empty,
        reasoning: std.ArrayList(u8) = .empty,
        saw_read_file: bool = false,

        fn emit(raw: *anyopaque, event: stream_provider.Event) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            switch (event) {
                .content_delta => |value| self.content.appendSlice(std.testing.allocator, value) catch unreachable,
                .reasoning_delta => |value| self.reasoning.appendSlice(std.testing.allocator, value) catch unreachable,
                .tool_started => |start| if (std.mem.eql(u8, start.name, "read_file")) {
                    self.saw_read_file = true;
                },
                else => {},
            }
        }
    };
    var capture: Capture = .{};
    defer capture.content.deinit(std.testing.allocator);
    defer capture.reasoning.deinit(std.testing.allocator);
    var events = stream_provider.EventSink{ .context = &capture, .emit_fn = Capture.emit };

    const sse_text =
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"README.md\\\"}\"}}]}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4}}\n\n" ++
        "data: [DONE]\n\n";

    var reader: std.Io.Reader = .fixed(sse_text);
    var cancelled = std.atomic.Value(bool).init(false);
    const completion = try consumeSse(
        std.testing.allocator,
        &reader,
        &events,
        &cancelled,
        null,
    );
    defer {
        if (completion.content) |value| std.testing.allocator.free(@constCast(value));
        types.freeToolCallSlice(std.testing.allocator, @constCast(completion.tool_calls));
    }

    try std.testing.expectEqualStrings("Hello", capture.content.items);
    try std.testing.expectEqualStrings("thinking", capture.reasoning.items);
    try std.testing.expect(capture.saw_read_file);
    try std.testing.expectEqualStrings("Hello", completion.content.?);
    try std.testing.expectEqual(@as(usize, 1), completion.tool_calls.len);
    try std.testing.expectEqualStrings("call_1", completion.tool_calls[0].id);
    try std.testing.expectEqualStrings("read_file", completion.tool_calls[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"README.md\"}", completion.tool_calls[0].arguments_json);
    try std.testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
    try std.testing.expectEqual(@as(?u64, 10), completion.usage.input_tokens);
    try std.testing.expectEqual(@as(?u64, 4), completion.usage.output_tokens);
}
