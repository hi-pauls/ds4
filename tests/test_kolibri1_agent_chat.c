#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

/* Kolibri-1 agent-transcript parity driver.
 *
 * Builds the exact transcript the ds4 agent assembles for a tool-calling
 * round (system block with effort + tools, user turn, replayed assistant
 * turn with two tool calls, two grouped tool results, fresh assistant
 * prefix) and prints the token ids, one per line.
 *
 * Modes:
 *   --tools-section   print agent_kolibri_tools_section(false) and exit
 *   --rules           print agent_kolibri_rules_text(false) and exit
 *   <model>           build the transcript and print ids
 *
 * tests/test_kolibri1_chat.py renders the equivalent conversation with the
 * released jinja template (tools= parsed from --tools-section) and compares
 * ids against the HuggingFace tokenizers encoding.
 */

static const char AGENT_SYSTEM_TEXT[] =
    "You are a coding agent running in a local workspace.";
static const char AGENT_USER_TEXT[] =
    "List the files here, then read the first one.";
static const char AGENT_ASSISTANT_TEXT[] =
    "<think>\n\n</think>\n\nChecking the directory first.\n"
    "<tool_call>\n"
    "{\"name\": \"list\", \"arguments\": {\"path\": \".\"}}\n"
    "</tool_call>\n"
    "<tool_call>\n"
    "{\"name\": \"read\", \"arguments\": {\"max_lines\": 5, \"path\": \"README.md\"}}\n"
    "</tool_call>";
static const char AGENT_TOOL_RESULT_1[] = "a\nb\nc";
static const char AGENT_TOOL_RESULT_2[] = "line1\nline2";

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--tools-section")) {
        char *tools = agent_kolibri_tools_section(false);
        fwrite(tools, 1, strlen(tools), stdout);
        free(tools);
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--rules")) {
        char *rules = agent_kolibri_rules_text(false);
        fwrite(rules, 1, strlen(rules), stdout);
        free(rules);
        return 0;
    }
    const char *model = argc > 1 ? argv[1] : "gguf/Kolibri-1-mini.gguf";
    ds4_engine_options opt = {.model_path = model, .backend = default_backend(),
                              .context_size = 4096, .power_percent = 100};
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "test_kolibri1_agent_chat: engine open failed for %s\n",
                model);
        return 2;
    }

    char *tools = agent_kolibri_tools_section(false);
    char *rules = agent_kolibri_rules_text(false);
    char content[4096];
    snprintf(content, sizeof(content), "%s%s", AGENT_SYSTEM_TEXT, rules);
    free(rules);

    ds4_tokens t = {0};
    ds4_chat_begin(e, &t);
    ds4_chat_append_system_effort_tools(e, &t, content, DS4_THINK_HIGH, tools);
    free(tools);
    ds4_chat_append_message(e, &t, "user", AGENT_USER_TEXT);

    /* The assistant prefix opens the generation turn; the replayed tokens are
     * what a thinking-enabled model would emit, then the turn end closes it. */
    ds4_chat_append_assistant_prefix(e, &t, DS4_THINK_HIGH);
    ds4_tokenize_rendered_chat(e, AGENT_ASSISTANT_TEXT, &t);
    ds4_chat_append_assistant_turn_end(e, &t);

    ds4_chat_append_message(e, &t, "tool", AGENT_TOOL_RESULT_1);
    ds4_chat_append_message(e, &t, "tool", AGENT_TOOL_RESULT_2);
    ds4_chat_append_assistant_prefix(e, &t, DS4_THINK_HIGH);

    for (int i = 0; i < t.len; i++) printf("%d\n", t.v[i]);
    ds4_tokens_free(&t);
    ds4_engine_close(e);
    return 0;
}
