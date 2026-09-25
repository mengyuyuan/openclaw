/** Terminal identity rules used to reconcile live and durable assistant projections. */

import { asNullableRecord as readRecord } from "@openclaw/normalization-core/record-coerce";
import { stableStringify } from "@openclaw/normalization-core/stable-stringify";
import {
  hasDisplayableSessionMessage,
  readSessionMessageDisplayContent,
  projectSessionTerminalReplyMessage,
} from "./session-projection-message-content.js";
import {
  readSessionMessageIdentity,
  sameAssistantPersistenceReceipt,
  sameTranscriptIdentity,
  type SessionMessageIdentity,
} from "./session-projection-message-identity.js";

/**
 * The class of rows the #148297 relaxation newly admits as finals: persisted
 * with the run's terminal tool stop reason while carrying no tool-call
 * content. Both adoption directions and every position rule scope to this
 * predicate, so pre-existing match semantics stay untouched.
 */
export function isToolUsePersistedFinalRow(message: unknown): boolean {
  const record = readRecord(message);
  return record?.["stopReason"] === "toolUse" && !isSessionProjectionToolContinuation(message);
}

type TerminalProjectionEntry = {
  message: unknown;
  identity: SessionMessageIdentity | null;
  afterSequence?: number | null;
  live: boolean;
};

type TerminalProjectionRun = {
  message?: unknown;
  status: string;
  acceptedFinalMessageIdentities?: readonly string[];
};

/** Tool-bearing assistant rows are continuations even without a tool stop reason. */
export function isSessionProjectionToolContinuation(message: unknown): boolean {
  const record = readRecord(message);
  if (
    Array.isArray(record?.content) &&
    record.content.some((block) => {
      const type = readRecord(block)?.type;
      return type === "toolCall" || type === "toolUse" || type === "functionCall";
    })
  ) {
    return true;
  }
  // A tool stop reason alone marks a continuation only when the row carries no
  // displayable text: a run's selected final answer is persisted with the
  // run's terminal stop reason even when the row itself is pure text, and such
  // a final must still reconcile with its unkeyed live projection (#148297).
  return record?.stopReason === "toolUse" && !hasDisplayableSessionMessage(message);
}

function readPersistedFinalIdentity(message: unknown): string | null {
  const identity = readSessionMessageIdentity(message);
  if (identity?.externalSource) {
    return `import:${identity.role}:${identity.externalSource}`;
  }
  if (identity?.id && !identity.isImported) {
    return `id:${identity.role}:${identity.id}`;
  }
  if (identity?.sequence !== null && identity?.sequence !== undefined) {
    return `seq:${identity.role}:${identity.sequence}`;
  }
  if (identity?.role === "assistant" && !identity.isImported && identity.idempotencyKey) {
    return `key:assistant:${identity.idempotencyKey}`;
  }
  return null;
}

function hasCompatiblePersistedFinalIdentity(currentMessage: unknown, incomingMessage: unknown) {
  const current = readSessionMessageIdentity(currentMessage);
  const incoming = readSessionMessageIdentity(incomingMessage);
  if (!current || !incoming || current.role !== incoming.role) {
    return false;
  }
  if (current.isImported || incoming.isImported) {
    if (!current.isImported || !incoming.isImported) {
      return false;
    }
    if (current.externalSource && incoming.externalSource) {
      return current.externalSource === incoming.externalSource;
    }
    return (
      current.sequence !== null &&
      incoming.sequence !== null &&
      current.sequence === incoming.sequence
    );
  }
  if (sameAssistantPersistenceReceipt(current, incoming)) {
    return true;
  }
  if (current.id && incoming.id) {
    return current.id === incoming.id;
  }
  return (
    current.sequence !== null &&
    incoming.sequence !== null &&
    current.sequence === incoming.sequence
  );
}

export function readFinalContentIdentity(message: unknown): string | null {
  const terminalMessage = projectSessionTerminalReplyMessage(message);
  const display = readSessionMessageDisplayContent(terminalMessage);
  if (!display.text && !display.hasNonText) {
    return null;
  }
  const identity = readSessionMessageIdentity(message);
  const record = readRecord(terminalMessage);
  const metadata = readRecord(record?.["__openclaw"]);
  try {
    return `content:${stableStringify([
      identity?.role ?? "assistant",
      display.text,
      display.hasNonText ? (record?.content ?? null) : null,
      metadata?.media ?? null,
      identity?.isImported
        ? [
            metadata?.importedFrom ?? null,
            metadata?.cliSessionId ?? null,
            metadata?.externalId ?? null,
          ]
        : null,
    ])}`;
  } catch {
    return null;
  }
}

function hasTerminalStopReason(message: unknown): boolean {
  const stopReason = readRecord(message)?.stopReason;
  return (
    stopReason === "stop" ||
    stopReason === "length" ||
    stopReason === "error" ||
    stopReason === "aborted" ||
    stopReason === "end_turn"
  );
}

/**
 * Shared position checks over one snapshot: indexes are built in a single sweep
 * and reused across every candidate row of a reconciliation pass. Snapshot
 * promotion and live replay reconciliation both scope to these rules.
 */
type RunSnapshotChecks = {
  hasCompletedRunSnapshotContext: (entry: TerminalProjectionEntry) => boolean;
  isLastSameRunAssistantRow: (entry: TerminalProjectionEntry) => boolean;
};

function createRunSnapshotChecks(
  snapshot: readonly TerminalProjectionEntry[],
  runId: string | null,
): RunSnapshotChecks {
  let snapshotIndexes: Map<TerminalProjectionEntry, number> | undefined;
  let firstUserIndex = -1;
  let lastAssistantIndex = -1;
  const ensureRunIndexes = (): Map<TerminalProjectionEntry, number> => {
    if (snapshotIndexes) {
      return snapshotIndexes;
    }
    const indexes = new Map<TerminalProjectionEntry, number>();
    if (runId) {
      snapshot.forEach((candidate, index) => {
        if (candidate.identity?.runId === runId) {
          // Keep indexOf's first-occurrence identity when a snapshot repeats an entry.
          if (!indexes.has(candidate)) {
            indexes.set(candidate, index);
          }
          if (candidate.identity.role === "user" && firstUserIndex < 0) {
            firstUserIndex = index;
          } else if (candidate.identity.role === "assistant") {
            lastAssistantIndex = index;
          }
        }
      });
    }
    snapshotIndexes = indexes;
    return indexes;
  };
  const readSameRunIndex = (entry: TerminalProjectionEntry): number | undefined => {
    if (!runId || entry.identity?.runId !== runId) {
      return undefined;
    }
    return ensureRunIndexes().get(entry);
  };
  return {
    hasCompletedRunSnapshotContext: (entry: TerminalProjectionEntry): boolean => {
      const entryIndex = readSameRunIndex(entry);
      return (
        entryIndex !== undefined &&
        firstUserIndex >= 0 &&
        firstUserIndex < entryIndex &&
        lastAssistantIndex <= entryIndex
      );
    },
    // A toolUse-persisted row can only be the run's selected final when no later
    // same-run assistant row exists in the snapshot: a following assistant or
    // tool row proves the run continued past it, so the live terminal must stay.
    isLastSameRunAssistantRow: (entry: TerminalProjectionEntry): boolean => {
      const entryIndex = readSameRunIndex(entry);
      return (
        entryIndex !== undefined && lastAssistantIndex >= 0 && entryIndex >= lastAssistantIndex
      );
    },
  };
}

/** Terminal evidence shared by snapshot promotion and live replay reconciliation. */
function hasTerminalProjectionEvidence(
  entry: TerminalProjectionEntry,
  runId: string | null,
  checks: RunSnapshotChecks,
): boolean {
  const metadata = readRecord(readRecord(entry.message)?.["__openclaw"]);
  return (
    metadata?.runTerminal === true ||
    (entry.identity?.runId === runId &&
      (hasTerminalStopReason(entry.message) ||
        // A row persisted with the run's terminal tool stop reason and no
        // tool-call content is the run's selected final; it must reconcile
        // with its unkeyed live projection (#148297). Unmarked rows stay
        // separate — partial history must not adopt the terminal — and the
        // row must be the run's last assistant row in the snapshot, since
        // a later same-run row proves the run continued past it.
        (isToolUsePersistedFinalRow(entry.message) && checks.isLastSameRunAssistantRow(entry)))) ||
    checks.hasCompletedRunSnapshotContext(entry)
  );
}

/** Explicit terminal markers that later history cannot contradict. */
function hasExplicitTerminalEvidence(entry: TerminalProjectionEntry): boolean {
  const metadata = readRecord(readRecord(entry.message)?.["__openclaw"]);
  return metadata?.runTerminal === true || hasTerminalStopReason(entry.message);
}

/** Read stable persisted identity first, falling back to canonical display content. */
export function readSessionProjectionFinalMessageIdentity(message: unknown): string | null {
  if (!hasDisplayableSessionMessage(message)) {
    return null;
  }
  return readPersistedFinalIdentity(message) ?? readFinalContentIdentity(message);
}

/** Check whether a displayable terminal may recover a prior empty terminal. */
export function canRecoverSessionProjectionFinal(
  currentMessage: unknown,
  incomingMessage: unknown,
): boolean {
  if (hasDisplayableSessionMessage(currentMessage)) {
    return false;
  }
  const currentIdentity = readPersistedFinalIdentity(currentMessage);
  return (
    currentIdentity === null || hasCompatiblePersistedFinalIdentity(currentMessage, incomingMessage)
  );
}

/** Check whether a run has already accepted the same terminal reply. */
export function hasSessionProjectionAcceptedFinal(
  run: TerminalProjectionRun | undefined,
  message: unknown,
): boolean {
  const identity = readSessionProjectionFinalMessageIdentity(message);
  return Boolean(
    identity &&
    run &&
    (run.acceptedFinalMessageIdentities?.includes(identity) ||
      readSessionProjectionFinalMessageIdentity(run.message) === identity),
  );
}

/** Match an unsequenced live terminal to exactly one durable same-run terminal row. */
export function findUniqueSnapshotTerminalMatch(
  current: TerminalProjectionEntry,
  matches: readonly TerminalProjectionEntry[],
  run: TerminalProjectionRun | undefined,
  snapshot: readonly TerminalProjectionEntry[],
): { entry: TerminalProjectionEntry; inferred: boolean } | null {
  if (
    !current.live ||
    current.identity?.role !== "assistant" ||
    current.identity.id ||
    current.identity.sequence !== null ||
    !run ||
    run.status === "streaming" ||
    matches.length === 0
  ) {
    return null;
  }
  const terminalContent = readFinalContentIdentity(current.message);
  if (!terminalContent || readFinalContentIdentity(run.message) !== terminalContent) {
    return null;
  }
  const entry = findUniqueTerminalContentMatch(
    matches,
    terminalContent,
    snapshot,
    current.identity?.runId ?? null,
  );
  if (!entry) {
    return null;
  }
  return {
    entry,
    inferred: !hasExplicitTerminalEvidence(entry),
  };
}

/**
 * Pick the unique durable row that carries terminal evidence and matches the
 * replay content; shared by snapshot promotion and live reconciliation.
 */
function findUniqueTerminalContentMatch<T extends TerminalProjectionEntry>(
  matches: readonly T[],
  content: string,
  snapshot: readonly TerminalProjectionEntry[],
  runId: string | null,
): T | null {
  const checks = createRunSnapshotChecks(snapshot, runId);
  const candidates = matches.filter(
    (entry) =>
      hasTerminalProjectionEvidence(entry, runId, checks) &&
      readFinalContentIdentity(entry.message) === content,
  );
  return candidates.length === 1 ? (candidates[0] ?? null) : null;
}

/**
 * Match an unsequenced live replay to exactly one durable row sharing the same
 * terminal evidence and full display content as snapshot reconciliation.
 */
export function findUniqueLiveTerminalMatch<T extends TerminalProjectionEntry>(
  current: TerminalProjectionEntry,
  matches: readonly T[],
  snapshot: readonly TerminalProjectionEntry[],
): T | null {
  const content = readFinalContentIdentity(current.message);
  if (!content) {
    return null;
  }
  return findUniqueTerminalContentMatch(
    matches,
    content,
    snapshot,
    current.identity?.runId ?? null,
  );
}

/** Check whether ordinary single-match promotion needs terminal-content verification. */
export function isUnsequencedLiveTerminal(
  current: TerminalProjectionEntry,
  run: TerminalProjectionRun | undefined,
): boolean {
  return Boolean(
    current.live &&
    current.identity?.role === "assistant" &&
    !current.identity.id &&
    current.identity.sequence === null &&
    run &&
    run.status !== "streaming" &&
    readFinalContentIdentity(current.message) === readFinalContentIdentity(run.message),
  );
}

/**
 * Keep a position-inferred live match recoverable until later history confirms
 * it; shared with snapshot reconciliation's tentative recovery record.
 */
export function withTentativeRecovery<TRun extends { message?: unknown; status: string }>(
  run: TRun | undefined,
  entry: TerminalProjectionEntry,
  matched: TerminalProjectionEntry,
): TRun | null {
  if (
    !run ||
    !matched.identity ||
    !isUnsequencedLiveTerminal(entry, run) ||
    hasExplicitTerminalEvidence(matched)
  ) {
    return null;
  }
  return { ...run, inferredSnapshotTerminal: { entry, matchedIdentity: matched.identity } };
}

/** Compose a run's tentative terminal-recovery record from a checked match. */
export function withInferredTerminal<TRun extends { message?: unknown; status: string }>(
  run: TRun,
  entry: TerminalProjectionEntry,
  matchedIdentity: SessionMessageIdentity,
): TRun {
  return { ...run, inferredSnapshotTerminal: { entry, matchedIdentity } };
}

/**
 * Check whether a live unsequenced assistant reply is a distinct later reply of
 * the run rather than its immutable first final. Tentative recovery cannot
 * represent these entries, so suppression must not drop them silently.
 * Sequence-fenced tails reconcile against their later durable row instead.
 */
function isDistinctLaterLiveFinal(
  current: TerminalProjectionEntry,
  run: TerminalProjectionRun | undefined,
): boolean {
  const content = readFinalContentIdentity(current.message);
  return Boolean(
    current.live &&
    current.identity?.role === "assistant" &&
    !current.identity.id &&
    current.identity.sequence === null &&
    current.afterSequence === undefined &&
    run &&
    run.status !== "streaming" &&
    content !== null &&
    content !== readFinalContentIdentity(run.message),
  );
}

/**
 * Keep a later distinct final visible when tentative recovery cannot represent
 * it: without explicit terminal evidence, a contradicted position match would
 * have no recovery record to restore the suppressed reply.
 */
export function shouldKeepLaterFinalVisible(
  entry: TerminalProjectionEntry,
  matched: TerminalProjectionEntry,
  run: TerminalProjectionRun | undefined,
): boolean {
  return !hasExplicitTerminalEvidence(matched) && isDistinctLaterLiveFinal(entry, run);
}

/** A single candidate still goes through the ambiguous-replay evidence check. */
export function findAdoptableMatch<T extends TerminalProjectionEntry>(
  matches: readonly T[],
): T | undefined {
  const single = matches.length === 1 ? matches[0] : undefined;
  return single && !isSessionProjectionToolContinuation(single.message) ? single : undefined;
}

/** The adoption target for a live projection: exact identity, then replay selection. */
export function findLiveTranscriptTarget<T extends TerminalProjectionEntry>(
  matches: readonly T[],
  incoming: TerminalProjectionEntry,
  snapshot: readonly TerminalProjectionEntry[],
): T | null {
  const direct = matches.find((entry) => sameTranscriptIdentity(entry.identity, incoming.identity));
  return (
    direct ??
    findAdoptableMatch(matches) ??
    findUniqueLiveTerminalMatch(incoming, matches, snapshot)
  );
}

/** A tool continuation can neither be adopted as nor replace an unkeyed live final. */
export function isToolContinuationOwner(
  tool: TerminalProjectionEntry,
  target: TerminalProjectionEntry,
): boolean {
  return isSessionProjectionToolContinuation(tool.message) && !target.identity?.id;
}

/**
 * Resolve how an unkeyed live terminal projection is adopted by a durable row:
 * persistence receipts suppress directly (no inference needed), position
 * matches keep a recovery record, and a later distinct final stays visible.
 */
export function resolveLiveAdoption<TRun extends { message?: unknown; status: string }>(
  entry: TerminalProjectionEntry,
  matched: TerminalProjectionEntry,
  runs: Readonly<Record<string, TRun>>,
): { runId: string | null; recovery: TRun | null; keepVisible: boolean } {
  const runId = entry.identity?.runId ?? null;
  if (sameAssistantPersistenceReceipt(matched.identity, entry.identity)) {
    return { runId, recovery: null, keepVisible: false };
  }
  const recovery = withTentativeRecovery(runId ? runs[runId] : undefined, entry, matched);
  return {
    runId,
    recovery,
    keepVisible:
      !recovery && shouldKeepLaterFinalVisible(entry, matched, runId ? runs[runId] : undefined),
  };
}
