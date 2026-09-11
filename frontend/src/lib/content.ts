/**
 * One content model, five design languages.
 *
 * Every design renders this exact object. That is the whole point of the
 * comparison: if one direction looks better because it quietly got shorter
 * copy or fewer metrics, the comparison is measuring the copy edit rather than
 * the design.
 */

export type Metric = {
  label: string;
  value: string;
  caption: string;
};

export type PipelineStage = {
  id: string;
  title: string;
  detail: string;
};

export type DemoMessage =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; citations: Citation[] }
  | { role: "refusal"; text: string };

export type Citation = {
  source: string;
  locator: string;
  score: number;
};

/**
 * The value a figure carries until it has actually been measured.
 *
 * Named rather than an inline em dash because the placeholder flag below is
 * derived from it: a cell is a proof exactly while it still holds this.
 */
const PROOF = "—";

export const content = {
  brand: {
    name: "NeuroNest",
    tagline: "Retrieval-augmented generation, measured.",
  },

  nav: [
    { label: "Overview", href: "#overview" },
    { label: "How it works", href: "#pipeline" },
    { label: "Results", href: "#results" },
    { label: "Docs", href: "#docs" },
  ],

  hero: {
    eyebrow: "Grounded question answering",
    headline: "Every answer cited, or refused.",
    subhead:
      "NeuroNest indexes what you give it and answers strictly from the passages it retrieved. When your corpus does not contain the answer, it declines instead of inventing one.",
    primaryCta: "Ingest a document",
    secondaryCta: "See the numbers",
  },

  /**
   * The second exchange is the one that matters. Any RAG demo can show a
   * confident cited answer; refusing an adjacent-but-absent question is the
   * behaviour almost nobody ships, so it belongs on the landing page rather
   * than buried in the docs.
   */
  demo: {
    placeholder: "Ask about anything in your corpus",
    suggestions: [
      "What does the paper claim about scaling?",
      "Summarise section 3",
      "Who funded this research?",
    ],
    messages: [
      { role: "user", text: "What does the paper claim about scaling laws?" },
      {
        role: "assistant",
        text: "Loss scales as a power law with model size, dataset size and compute, and the three must grow together — increasing parameters while holding data fixed hits a floor rather than continuing to improve.",
        citations: [
          { source: "scaling-laws.pdf", locator: "p. 4, ¶2", score: 0.87 },
          { source: "scaling-laws.pdf", locator: "p. 11, ¶1", score: 0.81 },
        ],
      },
      { role: "user", text: "What was the training run's electricity bill?" },
      {
        role: "refusal",
        text: "Not in the corpus. The indexed documents cover compute budgets in FLOPs but never cost in currency or energy, so there is nothing here to answer from.",
      },
    ] satisfies DemoMessage[],
  },

  pipeline: [
    { id: "load", title: "Load", detail: "PDF, Markdown and plain text. Scanned PDFs are rejected, not silently indexed as empty." },
    { id: "chunk", title: "Chunk", detail: "Deterministic splits with stable ids, so the same input always produces the same chunks." },
    { id: "embed", title: "Embed", detail: "Local sentence-transformers, cached on disk by model and content hash." },
    { id: "index", title: "Index", detail: "Persistent Chroma with upsert by chunk id — re-ingesting updates instead of duplicating." },
    { id: "retrieve", title: "Retrieve", detail: "Top-k with scores and a threshold. An empty result is a valid outcome, not an error." },
    { id: "generate", title: "Generate", detail: "Answers only from retrieved passages, with citations. Empty retrieval never reaches the model." },
  ] satisfies PipelineStage[],

  results: {
    heading: "Measured, not asserted",
    blurb:
      "Retrieval quality and generation quality are scored separately against 100 hand-labelled questions written before any tuning happened. Most RAG failures are retrieval failures blamed on the model; these numbers tell the two apart.",
    retrieval: [
      { label: "Recall@1", value: "0.429", caption: "labelled passage ranked first" },
      { label: "Recall@5", value: "0.729", caption: "labelled passage in the top five" },
      { label: "Recall@10", value: "0.800", caption: "labelled passage in the top ten" },
      { label: "MRR", value: "0.551", caption: "mean reciprocal rank" },
    ] satisfies Metric[],
    retrievalNote:
      "Over the 70 answerable questions: all-MiniLM-L6-v2, 1000/200 chunking, k=10, no threshold. The ten questions whose answer needs two separate passages score 0.400 at Recall@10, and a hand audit found six more where retrieval was right but the label named a different passage stating the same fact. Both are in the README, with the result file every figure is read from.",
    generation: [
      { label: "Faithfulness", value: PROOF, caption: "claims supported by sources" },
      { label: "Refusal accuracy", value: PROOF, caption: "on 20 unanswerable questions" },
      { label: "Judge agreement", value: PROOF, caption: "LLM judge vs. hand-checked sample" },
      { label: "p95 latency", value: PROOF, caption: "retrieval and generation, split" },
    ] satisfies Metric[],
    generationNote:
      "Not run yet. These four stay empty rather than plausible until the judge has been written and checked against a hand-scored sample.",
  },

  evaluation: {
    heading: "How the evaluation set was built",
    points: [
      { n: "70", label: "answerable questions", detail: "spread across the corpus, each labelled with the chunk that actually contains the answer" },
      { n: "20", label: "unanswerable questions", detail: "adjacent subjects deliberately absent from the corpus, to test refusal" },
      { n: "10", label: "multi-chunk questions", detail: "answers that require joining two separate passages" },
    ],
    note: "Labelled before tuning. A set written afterwards only asks questions the current configuration already answers.",
  },

  nonGoals: [
    "No multi-tenancy — one corpus, one index",
    "No authentication or accounts",
    "English only",
    "No memory between questions",
  ],

  footer: {
    repo: "github.com/DeepanshuSagore/neuronest",
    stack: ["Python 3.12", "FastAPI", "LangChain", "ChromaDB", "Docker"],
  },
} as const;

export type Content = typeof content;

/**
 * True while any published figure is still a proof rather than a measurement.
 *
 * Derived from the table rather than set by hand. Retrieval has been measured
 * and generation has not, and a boolean somebody has to remember to flip is
 * exactly how a disclaimer outlives the thing it was disclaiming.
 */
export const METRICS_ARE_PLACEHOLDER = [
  ...content.results.retrieval,
  ...content.results.generation,
].some((metric) => metric.value === PROOF);

export const MEASURED_COUNT = [
  ...content.results.retrieval,
  ...content.results.generation,
].filter((metric) => metric.value !== PROOF).length;
