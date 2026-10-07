/* No generation: drive the CLI token printer directly.  Plain chat hides the
 * tool-call wire format behind a dim placeholder (like the agent and server);
 * a marker inside thinking stays visible prose, and raw prompts keep every
 * byte. */
#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>

static char *capture(bool format_thinking, bool hide_tool_calls, bool use_color,
                     bool start_in_think,
                     const char *const *chunks, size_t count) {
    FILE *fp = tmpfile();
    assert(fp);
    token_printer printer = {
        .fp = fp,
        .format_thinking = format_thinking,
        .hide_tool_calls = hide_tool_calls,
        .in_think = start_in_think,
        .use_color = use_color,
        .last_output_newline = true,
    };
    for (size_t i = 0; i < count; i++)
        token_printer_write_text(&printer, chunks[i], strlen(chunks[i]));
    token_printer_finish(&printer);

    long size = ftell(fp);
    assert(size >= 0);
    rewind(fp);
    char *buf = malloc((size_t)size + 1);
    assert(buf);
    size_t got = fread(buf, 1, (size_t)size, fp);
    buf[got] = '\0';
    fclose(fp);
    return buf;
}

static void check(const char *label, char *got, const char *want) {
    if (strcmp(got, want)) {
        fprintf(stderr, "%s:\n  got:  [%s]\n  want: [%s]\n", label, got, want);
        assert(0);
    }
    free(got);
}

int main(void) {
    {
        const char *chunks[] = {
            "before\n",
            "<tool_call>\n{\"name\": \"list\", \"arguments\": {\"path\": \".\"}}\n",
            "</tool_call>\nafter\n",
        };
        check("chat call hidden", capture(true, true, false, false, chunks, 3),
              "before\n[tool call]\n\nafter\n");
    }
    {
        /* both markers split across token chunks */
        const char *chunks[] = {
            "a<to", "ol_call>{\"name\":\"x\"}</tool", "_call>b",
        };
        check("split markers", capture(true, true, false, false, chunks, 3),
              "a\n[tool call]\nb");
    }
    {
        /* inside thinking the marker is prose, and think tags still format */
        const char *chunks[] = {
            "<think>plan <tool_call> note</tool_call>", "</think>done",
        };
        check("in-think marker is prose",
              capture(true, true, false, true, chunks, 2),
              "plan <tool_call> note</tool_call>\ndone");
    }
    {
        /* no-think chat hides the stanza too */
        const char *chunks[] = { "x<tool_call>{}</tool_call>y" };
        check("no-think chat", capture(false, true, false, false, chunks, 1),
              "x\n[tool call]\ny");
    }
    {
        /* an orphan close marker outside a call stays raw */
        const char *chunks[] = { "a</tool_call>b" };
        check("orphan closer", capture(true, true, false, false, chunks, 1),
              "a</tool_call>b");
    }
    {
        /* raw prompts (hide off) keep every byte, markers included */
        const char *chunks[] = { "a<tool_call>{}</tool_call>b" };
        check("raw prompt", capture(false, false, false, false, chunks, 1),
              "a<tool_call>{}</tool_call>b");
    }
    {
        /* the placeholder picks up the dim style when color is enabled */
        const char *chunks[] = { "<tool_call>{}</tool_call>" };
        check("dim placeholder", capture(true, true, true, false, chunks, 1),
              "\x1b[90m[tool call]\x1b[0m\n");
    }
    puts("CLI token printer: PASS");
    return 0;
}
