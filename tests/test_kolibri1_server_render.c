/* Kolibri-1 server-renderer parity driver.
 *
 * Reads a request-shaped JSON object on stdin:
 *   {"messages": [...], "tools": [...]?}
 * parses it with the REAL server request parsers and renders it through
 * render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_KOLIBRI, ...),
 * printing the rendered prompt text to stdout. tests/test_kolibri1_chat.py
 * byte-compares that output against the released Jinja template's render of
 * the same conversation. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

static char *render_driver_read_stdin(void) {
    buf b = {0};
    char chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), stdin)) > 0)
        buf_append(&b, chunk, n);
    return buf_take(&b);
}

int main(int argc, char **argv) {
    ds4_think_mode mode = DS4_THINK_HIGH;
    int live_tail_start = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--effort") && i + 1 < argc) {
            const char *e = argv[++i];
            if (!strcmp(e, "none")) mode = DS4_THINK_NONE;
            else if (!strcmp(e, "low")) mode = DS4_THINK_LOW;
            else if (!strcmp(e, "medium")) mode = DS4_THINK_MEDIUM;
            else if (!strcmp(e, "high")) mode = DS4_THINK_HIGH;
            else {
                fprintf(stderr, "render driver: unknown effort %s\n", e);
                return 2;
            }
        } else if (!strcmp(argv[i], "--live-tail") && i + 1 < argc) {
            live_tail_start = atoi(argv[++i]);
        } else {
            fprintf(stderr, "render driver: unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    char *body = render_driver_read_stdin();
    const char *p = body ? body : "";
    chat_msgs msgs = {0};
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    json_ws(&p);
    if (*p != '{') {
        fprintf(stderr, "render driver: body must be a JSON object\n");
        return 2;
    }
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        json_ws(&p);
        if (!strcmp(key, "messages")) {
            if (!parse_messages(&p, &msgs)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tools")) {
            if (!parse_tools_value(&p, &schemas, &orders)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') {
            p++;
            json_ws(&p);
        }
    }
    if (*p != '}') goto bad;

    if (live_tail_start >= 0) {
        char *tail = render_live_tool_tail_for_syntax(
            SERVER_MODEL_SYNTAX_KOLIBRI, &msgs, live_tail_start, &orders, mode);
        if (!tail) goto bad;
        fputs(tail, stdout);
        return 0;
    }

    char *rendered = render_chat_prompt_text_for_syntax(
        SERVER_MODEL_SYNTAX_KOLIBRI, &msgs, schemas, &orders, mode);
    if (!rendered) goto bad;
    fputs(rendered, stdout);
    return 0;
bad:
    fprintf(stderr, "render driver: malformed request body\n");
    return 2;
}
