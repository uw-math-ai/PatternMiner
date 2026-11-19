# Direct LLM baselines

Appendix C gives each of GPT-6.2 Astra and GPT-6.2 Sol one 15-minute attempt on each of the five Stanley problems. Each Codex CLI session began with a fresh context, a separate workspace, the statement of one problem and the same procedural instructions. The agents could use Python and its standard library, choose subquestions, test examples and attempt proofs. The instructions excluded the manuscript results, PatternMiner workflow, earlier outputs, online retrieval, subagents and predictor training. The 900 seconds included computation and time waiting for model and tool responses.

Eight of ten initial attempts finished within the allowance. All ten yielded reports. The [Table 22 data](table-22.csv) give the elapsed times and selected mathematical outcomes reported in the paper. After the outputs were frozen, separate agents reviewed 66 claim records. They accepted the arguments for 53 within their stated domains, classified nine as finite evidence and left four unproved in the timed submissions.

The two Astra attempts that reached 900 seconds received fresh 30-minute retries. The Kronecker retry finished in 29 min 13 s and gave a block formula under a row-gap condition. The immanant retry finished in 29 min 40 s and gave a positivity formula for matrices of lower bandwidth at most two with nonzero second-subdiagonal entries in nonconsecutive rows.

The table and counts here are transcribed from Appendix C. The [matched Schubert projection comparison](../schubert-structure-constants/ablations/B/matched-projection/README.md) used GPT-6.1 Sol and has separate prompts and query logs.
