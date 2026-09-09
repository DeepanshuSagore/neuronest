"use client";

import { createScope, type Scope } from "animejs";
import { useEffect, useRef, type RefObject } from "react";

/**
 * Mechanism only — no timings, easings or motion vocabulary.
 *
 * Each design defines its own motion language; anything shared here beyond
 * lifecycle would quietly make two directions move alike, which is exactly
 * what the comparison is meant to surface.
 */

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Binds an anime.js scope to a React subtree.
 *
 * `createScope` is what makes selector strings inside `setup` resolve against
 * this component rather than the document, and `revert()` on unmount restores
 * every property anime.js touched. Without it, inline transforms survive the
 * unmount and the next design mounts onto dirty DOM — which is visible as
 * elements stuck mid-animation when switching directions.
 *
 * `setup` is held in a ref so callers can close over fresh props without
 * re-running the effect and re-triggering entrance animations on every render.
 */
export function useAnimeScope<T extends HTMLElement = HTMLDivElement>(
  setup: (scope: Scope) => void,
): RefObject<T | null> {
  const root = useRef<T | null>(null);
  const latest = useRef(setup);

  useEffect(() => {
    latest.current = setup;
  });

  useEffect(() => {
    if (!root.current) return;

    const scope = createScope({ root: root.current }).add((self) => {
      if (self) latest.current(self);
    });

    return () => {
      scope.revert();
    };
  }, []);

  return root;
}
