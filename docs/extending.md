# Extend

## Extending API Support

This section describes how to add support for a new backend API in **Trace-Replayer**.


### Steps

1. Create a new file under:

```text
src/apis/xxx_api.rs
```

2. Define a struct for the new API.
3. Implement the `LLMApi` trait:

   * **`request_json_body`**: Construct the HTTP request body
   * **`parse_response`** parses the HTTP response and receives `request_start`,
     the monotonic timestamp captured immediately before the HTTP send.
4. In `client.rs`, instantiate
   `spawn_request_loop_with_timestamp<T>` using the new API struct as the
   generic parameter.
