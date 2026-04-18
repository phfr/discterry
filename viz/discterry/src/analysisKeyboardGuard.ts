function isTypingTarget(t: EventTarget | null): boolean {
  if (!t || !(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return true;
  return t.isContentEditable;
}

/** When true, do not handle R/F/A view shortcuts. */
export function analysisKeyboardGuard(e: KeyboardEvent): boolean {
  if (e.defaultPrevented) return true;
  if (e.metaKey || e.ctrlKey || e.altKey) return true;
  return isTypingTarget(e.target);
}
