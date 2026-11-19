# Golden Knowledge sample data

`golden_trios.json` contains nine authored representative examples. No real Golden Knowledge bucket or historical expert-created corpus was supplied. These examples are **sample data**, not approved analyst work, measured warehouse results or an independent evaluation benchmark.

Golden Knowledge is a production design capability. This small seeded set and lexical retrieval are an optional prototype demonstration. The production lifecycle is described in [Architecture section 8](../docs/ARCHITECTURE.md#8-golden-knowledge-life-cycle).

## Prototype format

The file is a JSON array. Each object has:

| Field | Meaning |
|---|---|
| `id` | Stable sample identifier, such as `sample-001` |
| `provenance` | `sample`; explicitly distinguishes samples from historical expert data |
| `question` | Representative natural-language analysis question |
| `sql` | BigQuery Standard SQL template using the allowed bare table names |
| `report` | Illustrative analyst-report guidance: interpretation, report structure and possible follow-ups; not a claim that an analyst performed the work |
| `tags` | Topic keywords used by the prototype's lexical retriever |

The sample reports describe analysis methods rather than asserting fabricated business findings. Some describe follow-up calculations that require additional queries; the accompanying SQL is a starting point, not evidence for every suggested insight. Dates, periods and metric windows must be adapted to the current question. All factual results must come from tool execution.

Samples contain no stored brand-specific results. The SQL guard applies the signed-in user's brand scope before execution. The same guards apply to SQL inspired by an example as to any other generated SQL. The prompt labels retrieved examples as sample data.

The production format adds approval, lineage, scope and version metadata; those controls are designed in the HLD and are not implemented by this local JSON loader. The seed set is not a substitute for held-out, independently reviewed evaluation cases.
